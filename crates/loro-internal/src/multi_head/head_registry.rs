use std::cell::Cell;
use std::cmp::Ordering;
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::event::{DocDiff, EventTriggerKind};
use crate::subscription::{Observer, Subscriber};
use crate::sync::AtomicU8;
use crate::utils::subscription::Subscription;
use crate::version::{shrink_frontiers, Frontiers};
use crate::{DocOwner, LoroDoc, HEAD_MODE_PRIVATE};
use loro_common::{ContainerID, InternalString, LoroError, LoroResult};

/// The `(origin, by)` an emitted transition diff carries (from `ResolveCause`).
type EventTag = (InternalString, EventTriggerKind);

/// A branch's subscriptions (container target + callback), snapshotted under
/// the registry lock and dispatched to after it drops.
pub(super) type SubsSnapshot = Vec<(Option<ContainerID>, Subscriber)>;

/// Where a branch's advance sources its state from.
enum AdvanceSource {
    /// Advance the branch's uniquely-owned (`refs == 1`) bound head in place.
    InPlace(HeadId),
    /// Copy this head (a shared head, or the pinned root), rebind the branch
    /// onto the copy, then advance the copy.
    CopyOf(HeadId),
}

use super::base::{install_forwarders, HeadOwner};
use super::ROOT_HEAD_ID;
use super::*;

// `MultiHeadDoc::on_head_committed` (fired by the txn on-commit hook) takes the
// registry lock. A commit triggered inside a registry op (`flip_to_shared`
// flushing pending ops) would re-enter it on the same thread, so this depth
// counter lets the callback defer; the in-progress op re-keys `by_tip` itself.
thread_local! {
    static REG_DEPTH: Cell<u32> = const { Cell::new(0) };
}

pub(crate) fn in_registry_op() -> bool {
    REG_DEPTH.with(|c| c.get() > 0)
}

pub(super) struct RegOpGuard;
impl RegOpGuard {
    pub(super) fn enter() -> Self {
        REG_DEPTH.with(|c| c.set(c.get() + 1));
        RegOpGuard
    }
}
impl Drop for RegOpGuard {
    fn drop(&mut self) {
        REG_DEPTH.with(|c| c.set(c.get() - 1));
    }
}

/// A materialized state at one tip: a `LoroDoc` over the shared `OpLog`.
pub(super) struct Head {
    pub(super) doc: LoroDoc,
    pub(super) tip: Frontiers,
    /// Number of `bound` entries pointing here: `1` = uniquely owned
    /// (writable in place); `> 1` = shared (immutable); `0` = retired, except
    /// the pinned root.
    pub(super) refs: usize,
    /// A lens (copy-on-open) head: the `refs == 1` head an open branch pins. Never in
    /// `by_tip`, so no sibling can share into or retire it. Delivers via its own
    /// observer, not the scratch `dispatch_to_branch`.
    pub(super) unshared: bool,
    /// Forwarders into the doc-level subscriptions; dropped with the head.
    pub(super) _forward: Vec<Subscription>,
}

/// The head registry, guarded by one `LockKind::BranchRegistry` lock. Acquired
/// before any head lock (so `resolve` can copy/rebind under it) and never held
/// into a content op.
pub(super) struct Registry {
    pub(super) heads: FxHashMap<HeadId, Head>,
    /// Resting heads by tip: the sharing lookup. A freshly copied head is
    /// absent until its first commit re-keys it.
    pub(super) by_tip: FxHashMap<Frontiers, HeadId>,
    /// Branch -> the head it currently uses.
    pub(super) bound: FxHashMap<BranchId, HeadId>,
    /// Branch-scoped subscriptions, re-installed on the new head across a rebind and
    /// the delivery target of a transition's synthesized diff.
    pub(super) subs: FxHashMap<BranchId, Vec<BranchSub>>,
    /// Branches open (lensed) from `open_lens` until unbound; they never take the
    /// rebind-to-existing arm.
    pub(super) open_branches: FxHashSet<BranchId>,
    /// Whether the most recent resolve held the branch behind a non-forward
    /// (ancestor) target. See `MultiHeadDoc<Delegated>::lens_held`.
    pub(super) guard_held: FxHashMap<BranchId, bool>,
    /// Bumped on every `rebind`; see `lens_generation`.
    pub(super) generation: FxHashMap<BranchId, u64>,
    pub(super) next_id: HeadId,
    pub(super) next_sub_id: u64,
}

/// A registry-owned branch subscription, re-installed on the branch's head on
/// each rebind. `installed` is the handle on the current head.
pub(super) struct BranchSub {
    pub(super) id: u64,
    /// The container to watch, or `None` for a root subscription.
    pub(super) target: Option<ContainerID>,
    pub(super) cb: Subscriber,
    pub(super) installed: Option<Subscription>,
}

impl Registry {
    fn alloc_id(&mut self) -> HeadId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

impl<P: HeadPolicy> MultiHeadDoc<P> {
    /// Copy a head: a new `LoroDocInner` over the same oplog, arena, config, and lock
    /// group, with forked state (fresh peer, `Private`). `refs == 0` and absent from
    /// `by_tip` until bound/committed.
    pub(super) fn copy_head(&self, reg: &mut Registry, src_id: HeadId) -> HeadId {
        let id = reg.alloc_id();
        let src_state = reg.heads[&src_id].doc.state.clone();
        let head_mode = Arc::new(AtomicU8::new(HEAD_MODE_PRIVATE));
        let arena = self.arena.clone();
        let config = self.config.clone();
        let lg = self.lock_group.clone();
        let owner: Arc<dyn DocOwner> = Arc::new(HeadOwner {
            inner: Arc::downgrade(&self.inner),
            head_id: id,
        });
        let doc = LoroDoc::build_head(
            self.oplog.clone(),
            arena.clone(),
            config.clone(),
            &self.lock_group,
            self.visible_op_count.clone(),
            head_mode,
            Some(owner),
            move |cyclic, hm| {
                src_state.lock().fork_in_group(
                    cyclic.clone(),
                    arena.clone(),
                    config.clone(),
                    &lg,
                    hm.clone(),
                )
            },
            true,
        );
        let forward = install_forwarders(&Arc::downgrade(&self.inner), &doc);
        let tip = doc.state_frontiers();
        reg.heads.insert(
            id,
            Head {
                doc,
                tip,
                refs: 0,
                unshared: false,
                _forward: forward,
            },
        );
        id
    }

    fn inc_refs(&self, reg: &mut Registry, id: HeadId) {
        let refs = {
            let h = reg.heads.get_mut(&id).expect("head exists");
            h.refs += 1;
            h.refs
        };
        if refs == 2 {
            self.flip_to_shared(reg, id);
        }
    }

    pub(super) fn dec_refs(&self, reg: &mut Registry, id: HeadId) {
        let refs = {
            let h = reg.heads.get_mut(&id).expect("head exists");
            h.refs -= 1;
            h.refs
        };
        if refs == 1 {
            self.flip_to_private(reg, id);
        } else if refs == 0 {
            self.retire(reg, id);
        }
    }

    /// `1 -> 2`: commit any pending transaction (a shared head never has an
    /// open one), re-key `by_tip` if that moved the tip, then mark the head
    /// `Shared`. The commit hook defers inside a registry op, so we re-key here.
    fn flip_to_shared(&self, reg: &mut Registry, id: HeadId) {
        let h = reg.heads.get_mut(&id).expect("head exists");
        let old_tip = h.tip.clone();
        let (_opts, guard) = h.doc.implicit_commit_then_stop();
        let new_tip = h.doc.state_frontiers();
        h.doc.set_head_shared(true);
        let unshared = h.unshared;
        h.tip = new_tip.clone();
        drop(guard);
        if old_tip != new_tip {
            if reg.by_tip.get(&old_tip) == Some(&id) {
                reg.by_tip.remove(&old_tip);
            }
            // Defensive: a lens head is `refs == 1` and never reaches here.
            if !unshared {
                reg.by_tip.insert(new_tip, id);
            }
        }
    }

    /// `2 -> 1`: uniquely owned again; make it writable.
    fn flip_to_private(&self, reg: &mut Registry, id: HeadId) {
        let h = reg.heads.get(&id).expect("head exists");
        h.doc.set_head_shared(false);
        h.doc.renew_txn_if_auto_commit(None);
    }

    /// Drop a `refs == 0` head. The root is pinned: it anchors `by_tip`, `materialize` and
    /// import/export. Asserts no pending ops instead of letting `Drop` implicit-commit them:
    /// `write` commits before returning and rebinds happen at op boundaries, so a pending op
    /// is out of contract.
    fn retire(&self, reg: &mut Registry, id: HeadId) {
        if id == ROOT_HEAD_ID {
            return;
        }
        if let Some(h) = reg.heads.get(&id) {
            debug_assert_eq!(
                h.doc.get_pending_txn_len(),
                0,
                "a head reaching refs == 0 must have no pending local ops \
                 (they would spuriously commit on drop)"
            );
            let tip = h.tip.clone();
            if reg.by_tip.get(&tip) == Some(&id) {
                reg.by_tip.remove(&tip);
            }
        }
        reg.heads.remove(&id);
    }

    /// Bind branch `b` to `new_id`, adjusting refs (and thus shared/private
    /// mode) on the old and new heads.
    pub(super) fn rebind(&self, reg: &mut Registry, b: &BranchId, new_id: HeadId) {
        if let Some(&old) = reg.bound.get(b) {
            if old == new_id {
                return;
            }
            self.dec_refs(reg, old);
        }
        reg.bound.insert(b.clone(), new_id);
        // Invalidates the JS lens handle cache.
        *reg.generation.entry(b.clone()).or_insert(0) += 1;
        self.inc_refs(reg, new_id);
        self.reinstall_subs(reg, b, new_id);
    }

    /// Move branch `b`'s subscriptions from its old head to `new_id`, so they
    /// keep firing after a copy-on-divergence.
    fn reinstall_subs(&self, reg: &mut Registry, b: &BranchId, new_id: HeadId) {
        let head = match reg.heads.get(&new_id) {
            Some(h) => h.doc.clone(),
            None => return,
        };
        if let Some(subs) = reg.subs.get_mut(b) {
            for s in subs.iter_mut() {
                s.installed = None;
                let installed = match &s.target {
                    Some(cid) => head.subscribe(cid, s.cb.clone()),
                    None => head.subscribe_root(s.cb.clone()),
                };
                s.installed = Some(installed);
            }
        }
    }

    /// Snapshot `b`'s subscriptions for delivery after the registry lock drops.
    pub(super) fn snapshot_subs(&self, reg: &Registry, b: &BranchId) -> SubsSnapshot {
        reg.subs
            .get(b)
            .map(|v| v.iter().map(|s| (s.target.clone(), s.cb.clone())).collect())
            .unwrap_or_default()
    }

    /// Deliver `events` to `subs` through a scratch observer: it reproduces a plain doc's
    /// container filtering (`Observer::emit_inner`), and holds only this branch's
    /// callbacks, so a shared head's co-owner never sees the jump. Runs after the
    /// registry lock drops, so callbacks may re-enter the registry. Handles are held
    /// until the last `emit` (dropping one unsubscribes it).
    pub(super) fn dispatch_to_branch(&self, subs: SubsSnapshot, events: Vec<DocDiff>) {
        if subs.is_empty() || events.is_empty() {
            return;
        }
        let observer = Observer::new(self.arena.clone());
        let mut handles = Vec::with_capacity(subs.len());
        for (target, cb) in subs {
            let handle = match &target {
                Some(cid) => observer.subscribe(cid, cb),
                None => observer.subscribe_root(cb),
            };
            handles.push(handle);
        }
        for ev in events {
            observer.emit(ev);
        }
        drop(handles);
    }

    /// Resolve branch `b` to its head, copying and rebinding when a `Write` reaches a
    /// shared head. A resolve that moves `b`'s tip delivers `diff(old -> new)` to
    /// `b`'s subscribers only, after the registry lock drops.
    ///
    /// Low-level: returns a raw `LoroDoc`, protected only by the `OwnedHeadOp` gates
    /// and the sink guard. Consumers go through `Branch`.
    #[doc(hidden)]
    pub fn resolve(&self, b: &BranchId, intent: Intent) -> LoroResult<(HeadId, LoroDoc)> {
        self.resolve_with(b, intent, ResolveCause::Access)
    }

    /// `resolve`, tagging the delivered transition event with `cause`.
    #[doc(hidden)]
    pub fn resolve_with(
        &self,
        b: &BranchId,
        intent: Intent,
        cause: ResolveCause,
    ) -> LoroResult<(HeadId, LoroDoc)> {
        let (out, pending, subs, lens_delivery) = self.resolve_collecting(b, intent, cause)?;
        self.deliver(&out.1, subs, pending, lens_delivery);
        Ok(out)
    }

    /// Deliver a transition's collected diffs after the registry lock drops. A lens
    /// head emits through its own observer (the branch's subscriptions are installed
    /// there, so its `UndoManager` and raw `subscribe`s hear it too, without double
    /// delivery). A cold or shared head uses the scratch dispatch: co-owners must not fire.
    fn deliver(
        &self,
        doc: &LoroDoc,
        subs: SubsSnapshot,
        pending: Vec<DocDiff>,
        lens_delivery: bool,
    ) {
        if pending.is_empty() {
            return;
        }
        if lens_delivery {
            doc.emit_collected_diffs(pending);
        } else {
            self.dispatch_to_branch(subs, pending);
        }
    }

    /// The registry-locked core of `resolve`: picks an arm, moves `b`, and
    /// returns the head, the pending diff, a subscription snapshot, and whether
    /// the head is a lens head, for `resolve` to deliver after the lock drops.
    ///
    /// A fast-forward-only guard runs above arm selection for a bound head,
    /// comparing its tip to `target` on the shared DAG. `Greater` (target is an
    /// ancestor: a regress) or an unheld frontier HOLDS: the head is returned
    /// unmoved, nothing is delivered, and the hold is recorded for `lens_held`.
    /// `None` (parallel) advances to the join `shrink(tip ∪ target)`.
    /// `Equal`/`Less` fall through. `ResolveCause::Advance` is exempt: a
    /// backward `advance` is the one intentional backward path.
    fn resolve_collecting(
        &self,
        b: &BranchId,
        intent: Intent,
        cause: ResolveCause,
    ) -> LoroResult<((HeadId, LoroDoc), Vec<DocDiff>, SubsSnapshot, bool)> {
        let ev = cause.to_event();
        self.with_reg(|this, reg| {
            let mut target = this.inner.policy.target(this, b)?;
            let mut pending: Vec<DocDiff> = Vec::new();

            let bound_head = reg.bound.get(b).copied();
            let mut hold = false;
            if !matches!(cause, ResolveCause::Advance) {
                if let Some(h) = bound_head {
                    let tip = reg.heads[&h].tip.clone();
                    let cmp = {
                        let ol = this.oplog.lock();
                        ol.dag.cmp_frontiers(&tip, &target)
                    };
                    match cmp {
                        // Regress or unheld op: HOLD.
                        Ok(Some(Ordering::Greater)) | Err(_) => {
                            hold = true;
                        }
                        // Parallel: advance to the join.
                        Ok(None) => {
                            let union: Frontiers = tip.iter().chain(target.iter()).collect();
                            let ol = this.oplog.lock();
                            target = shrink_frontiers(&union, &ol.dag)
                                .map_err(LoroError::FrontiersNotFound)?;
                        }
                        Ok(Some(Ordering::Equal)) | Ok(Some(Ordering::Less)) => {}
                    }
                }
            }
            reg.guard_held.insert(b.clone(), hold);

            let mut id = if hold {
                // No move. The Write-copy tail below still privatizes a shared head.
                bound_head.expect("hold implies a bound head")
            } else {
            match reg.bound.get(b).copied() {
                Some(h) if reg.heads[&h].tip == target => h,
                cur => match if reg.open_branches.contains(b) {
                    // An open branch advances its own unshared head in place
                    // instead of rebinding onto a sibling's resting head.
                    None
                } else {
                    reg.by_tip.get(&target).copied()
                } {
                    Some(h2) => {
                        // h2 already rests at `target`; `rebind` delivers nothing, so
                        // synthesize `diff(old -> target)` on a scratch copy for `b` only.
                        if let Some(old) = cur {
                            let subs_nonempty =
                                reg.subs.get(b).map(|v| !v.is_empty()).unwrap_or(false);
                            if subs_nonempty && old != h2 {
                                // Copy before `rebind`, which may retire `old`. Diff on
                                // the scratch, not via `advance_in_place`, whose `by_tip`
                                // re-key would clobber h2's key.
                                let scratch = this.copy_head(reg, old);
                                let scratch_doc = reg.heads[&scratch].doc.clone();
                                pending = scratch_doc.checkout_collecting_events(
                                    &target,
                                    ev.0.clone(),
                                    ev.1,
                                )?;
                                this.retire(reg, scratch);
                            }
                        }
                        this.rebind(reg, b, h2);
                        h2
                    }
                    None => match cur {
                        // Catch-up arm: advance the bound `refs == 1` head in place.
                        Some(h) if reg.heads[&h].refs == 1 => {
                            let (id2, evs) =
                                this.advance_via(reg, b, AdvanceSource::InPlace(h), &target, None, &ev)?;
                            pending = evs;
                            id2
                        }
                        // Copy+advance arm: shared head whose target moved off its tip.
                        Some(h) => {
                            let (c, evs) = this.advance_via(
                                reg,
                                b,
                                AdvanceSource::CopyOf(h),
                                &target,
                                Some(h),
                                &ev,
                            )?;
                            pending = evs;
                            c
                        }
                        // Materialize arm: cold branch, from a copy of the pinned root.
                        None => {
                            let (m, evs) = this.advance_via(
                                reg,
                                b,
                                AdvanceSource::CopyOf(ROOT_HEAD_ID),
                                &target,
                                None,
                                &ev,
                            )?;
                            pending = evs;
                            m
                        }
                    },
                },
            }
            };

            if intent == Intent::Write && reg.heads[&id].refs > 1 {
                debug_assert!(
                    P::COPY == CopyMode::OnDivergence,
                    "an Eager head is born refs == 1 and never reaches the Write copy arm"
                );
                let c = this.copy_head(reg, id);
                this.rebind(reg, b, c);
                id = c;
            }

            let subs = this.snapshot_subs(reg, b);
            let lens_delivery = reg.heads[&id].unshared;
            Ok(((id, reg.heads[&id].doc.clone()), pending, subs, lens_delivery))
        })
    }

    /// Check a head out to `target`, returning the resulting `DocDiff`s instead of
    /// emitting them (works even for a fresh, never-subscribed copy).
    fn advance_in_place(
        &self,
        reg: &mut Registry,
        id: HeadId,
        target: &Frontiers,
        ev: &EventTag,
    ) -> LoroResult<Vec<DocDiff>> {
        let (old_tip, doc) = {
            let h = reg.heads.get(&id).expect("head exists");
            (h.tip.clone(), h.doc.clone())
        };
        if old_tip == *target {
            return Ok(Vec::new());
        }
        // `checkout` leaves the head detached with its txn stopped. Callers that hand
        // back a writable head must clear that (see `advance_via`), else the next
        // write fails `AutoCommitNotStarted`.
        let events = doc.checkout_collecting_events(target, ev.0.clone(), ev.1)?;
        let new_tip = doc.state_frontiers();
        let unshared = {
            let h = reg.heads.get_mut(&id).expect("head exists");
            h.tip = new_tip.clone();
            h.unshared
        };
        if reg.by_tip.get(&old_tip) == Some(&id) {
            reg.by_tip.remove(&old_tip);
        }
        // `or_insert`, not `insert`: a `read_at` scratch head advanced to a frontier a
        // live head rests at must not displace that key (`retire(scratch)` would drop
        // the live head from `by_tip`). A `resolve` only gets here on a `by_tip` miss, so
        // it still installs the moved head. A lens (unshared) head never enters `by_tip`.
        if !unshared {
            reg.by_tip.entry(new_tip).or_insert(id);
        }
        Ok(events)
    }

    /// Advance `b` to `target` via `src`, returning the collected diff. A `CopyOf` copy
    /// is rebound to `b` first so its subscriptions are installed. The result is
    /// writable (detached cleared, txn renewed). On error `b` is rebound to
    /// `fallback` (or unbound), retiring the copy.
    fn advance_via(
        &self,
        reg: &mut Registry,
        b: &BranchId,
        src: AdvanceSource,
        target: &Frontiers,
        fallback: Option<HeadId>,
        ev: &EventTag,
    ) -> LoroResult<(HeadId, Vec<DocDiff>)> {
        let (id, pending) = match src {
            AdvanceSource::InPlace(id) => {
                let pending = self.advance_in_place(reg, id, target, ev)?;
                (id, pending)
            }
            AdvanceSource::CopyOf(src_id) => {
                let c = self.copy_head(reg, src_id);
                self.rebind(reg, b, c);
                match self.advance_in_place(reg, c, target, ev) {
                    Ok(pending) => (c, pending),
                    Err(e) => {
                        match fallback {
                            Some(old) => self.rebind(reg, b, old),
                            None => {
                                reg.bound.remove(b);
                                self.dec_refs(reg, c);
                            }
                        }
                        return Err(e);
                    }
                }
            }
        };
        let doc = &reg.heads[&id].doc;
        doc.set_detached(false);
        doc.renew_txn_if_auto_commit(None);
        Ok((id, pending))
    }

    /// Build a head for a frontier no live head sits at, by copying the pinned root
    /// and checking out to `target`. Correct but not cheap.
    fn materialize(
        &self,
        reg: &mut Registry,
        target: &Frontiers,
        ev: &EventTag,
    ) -> LoroResult<(HeadId, Vec<DocDiff>)> {
        let c = self.copy_head(reg, ROOT_HEAD_ID);
        let events = self.advance_in_place(reg, c, target, ev)?;
        Ok((c, events))
    }

    /// Advance `b`'s bound `refs == 1` head to `target` in place, leaving it writable.
    /// For an index merge, whose `target` bypasses the policy and which then commits
    /// a marker on the head. Errors if `b` is unbound. Takes the registry lock: must
    /// not be called inside another registry op.
    pub(super) fn advance_bound_writable(
        &self,
        b: &BranchId,
        target: &Frontiers,
    ) -> LoroResult<LoroDoc> {
        let (doc, pending, subs, lens_delivery) = self.with_reg(|this, reg| {
            let id = *reg.bound.get(b).ok_or_else(|| {
                LoroError::ArgErr(format!("cannot advance: branch '{b}' is not bound").into_boxed_str())
            })?;
            debug_assert_eq!(
                reg.heads[&id].refs, 1,
                "advance_bound_writable requires a uniquely-owned (refs == 1) head"
            );
            let ev = ResolveCause::Advance.to_event();
            let pending = this.advance_in_place(reg, id, target, &ev)?;
            let doc = reg.heads[&id].doc.clone();
            // The following merge-marker commit must not fail `AutoCommitNotStarted`.
            doc.set_detached(false);
            doc.renew_txn_if_auto_commit(None);
            let subs = this.snapshot_subs(reg, b);
            let lens_delivery = reg.heads[&id].unshared;
            Ok::<_, LoroError>((doc, pending, subs, lens_delivery))
        })?;
        self.deliver(&doc, subs, pending, lens_delivery);
        Ok(doc)
    }

    /// Run `f` on a throwaway scratch head checked out to `target`, without
    /// disturbing any bound head. `f` must not mutate (retire asserts no pending
    /// ops).
    pub(super) fn read_at<R>(
        &self,
        target: &Frontiers,
        f: impl FnOnce(&LoroDoc) -> R,
    ) -> LoroResult<R> {
        self.with_reg(|this, reg| {
            // No branch moved: the collected diff is discarded.
            let ev = ResolveCause::Access.to_event();
            let (scratch, _events) = this.materialize(reg, target, &ev)?;
            let out = f(&reg.heads[&scratch].doc);
            this.retire(reg, scratch);
            Ok(out)
        })
    }

    /// Copy-on-open: resolve `b` for write (privatizing a shared head), then pin it as
    /// a lens head (`unshared`, evicted from `by_tip`). Later resolves advance this
    /// same head in place, fast-forward-only.
    ///
    /// Exposes the full `LoroDoc` surface that `BranchingDocHead` withholds, but the head
    /// is owned: `checkout`/`import`/`set_peer_id` return `OwnedHeadOp`, and
    /// `detach`/`checkout_to_latest` are no-ops.
    pub fn open_lens(&self, b: &BranchId) -> LoroResult<LoroDoc> {
        let (id, doc) = self.resolve(b, Intent::Write)?;
        self.with_reg(|_, reg| {
            reg.open_branches.insert(b.clone());
            if let Some(h) = reg.heads.get_mut(&id) {
                h.unshared = true;
            }
            // The resolve may have inserted this head into `by_tip`.
            let tip = reg.heads[&id].tip.clone();
            if reg.by_tip.get(&tip) == Some(&id) {
                reg.by_tip.remove(&tip);
            }
        });
        Ok(doc)
    }

    /// Branch `b`'s identity generation; the JS lens cache re-mints only when it moves.
    pub fn lens_generation(&self, b: &BranchId) -> u64 {
        self.with_reg(|_, reg| reg.generation.get(b).copied().unwrap_or(0))
    }

    /// Whether the fast-forward guard held `b` on its last resolve.
    pub(super) fn guard_held_flag(&self, b: &BranchId) -> bool {
        self.with_reg(|_, reg| reg.guard_held.get(b).copied().unwrap_or(false))
    }

    /// Release a branch's open flag and guard-hold flag and invalidate its lens
    /// cache generation. Called from `unbind`.
    pub(super) fn release_open(&self, reg: &mut Registry, b: &BranchId) {
        reg.open_branches.remove(b);
        reg.guard_held.remove(b);
        *reg.generation.entry(b.clone()).or_insert(0) += 1;
    }
}
