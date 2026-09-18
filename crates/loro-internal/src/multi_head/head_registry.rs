use std::cell::Cell;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::event::{DocDiff, EventTriggerKind};
use crate::subscription::{Observer, Subscriber};
use crate::sync::AtomicU8;
use crate::utils::subscription::Subscription;
use crate::version::Frontiers;
use crate::{DocOwner, LoroDoc, HEAD_MODE_PRIVATE};
use loro_common::{ContainerID, LoroError, LoroResult};

/// A snapshot of a branch's subscriptions (container target + callback), taken
/// under the registry lock and dispatched to AFTER the lock drops.
pub(super) type SubsSnapshot = Vec<(Option<ContainerID>, Subscriber)>;

/// Where a branch's advance sources its state from. The three advancing arms are
/// ONE operation ("advance branch `b` to `target` via source `S`") with three
/// sources: the branch's own uniquely-owned head in place, a copy of its shared
/// head, or a copy of the pinned root (materialize).
enum AdvanceSource {
    /// Advance the branch's uniquely-owned (`refs == 1`) bound head in place.
    InPlace(HeadId),
    /// Copy this head first (a shared head, or the pinned root for materialize),
    /// rebind the branch onto the copy, then advance the copy.
    CopyOf(HeadId),
}

use super::base::{install_forwarders, HeadOwner};
use super::ROOT_HEAD_ID;
use super::*;

// A head commit fires the injected txn on-commit hook (see `DocOwner`), which
// calls `MultiHeadDoc::on_head_committed`. That callback needs the registry
// lock. When the commit was itself triggered from INSIDE a registry operation
// (a `flip_to_shared` that flushes a head's pending ops before marking it
// immutable), taking the lock again on the same thread would be reentrant. This
// thread-local depth counter lets the callback detect that case and defer: the
// in-progress registry op re-keys `by_tip` itself, so nothing is lost.
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
    /// Number of `bound` entries pointing here. The whole sharing rule:
    /// `1` = uniquely owned (writable in place); `> 1` = shared (immutable);
    /// `0` = retired (removed) -- except the pinned root, which rests at `0`.
    pub(super) refs: usize,
    /// Per-head forwarders that feed this head's local commits / first-commits
    /// into the doc-level `history_subs` / `first_commit_subs`. Kept alive for
    /// the head's lifetime; dropped (unsubscribed) when the head is evicted.
    pub(super) _forward: Vec<Subscription>,
}

/// The head registry. Guarded by a single `LockKind::BranchRegistry` lock,
/// acquired before any head lock so `resolve` can copy/rebind (taking head
/// `Txn`/`DocState` locks) while holding it, and never held into an actual
/// content op.
pub(super) struct Registry {
    pub(super) heads: FxHashMap<HeadId, Head>,
    /// Resting heads by tip: the sharing lookup. A freshly copied head is
    /// ABSENT until its first commit re-keys it (invariant I3).
    pub(super) by_tip: FxHashMap<Frontiers, HeadId>,
    /// Branch -> the head it currently uses.
    pub(super) bound: FxHashMap<BranchId, HeadId>,
    /// Branch-scoped subscriptions, registry-owned so they can be RE-INSTALLED
    /// on the branch's new head across a rebind (copy-on-divergence / merge),
    /// instead of going silent on the old head.
    pub(super) subs: FxHashMap<BranchId, Vec<BranchSub>>,
    pub(super) next_id: HeadId,
    pub(super) next_sub_id: u64,
}

/// A registry-owned branch subscription: its callback re-installed on the
/// branch's head whenever the branch rebinds. `installed` is the handle on the
/// CURRENT head (dropped -> unsubscribed) and is replaced on each rebind.
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
    // --- head construction -------------------------------------------------

    /// Structurally copy a head: a new `LoroDocInner` over the SAME
    /// `Arc<OpLog>`, arena, config, and lock group, with `fork_in_group` for
    /// the state (fresh peer, fresh `DiffCalculator`, `Private`), its registry
    /// back-pointer and history forwarders installed. Inserted with `refs == 0`
    /// and absent from `by_tip` (invariant I3) until a caller binds/commits it.
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
                _forward: forward,
            },
        );
        id
    }

    // --- refs / mode flips -------------------------------------------------

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

    /// `1 -> 2`: commit any pending transaction (so a shared head never has an
    /// open transaction, invariant I2), re-key `by_tip` if that commit moved
    /// the tip, then mark the head `Shared`. The flush's on-commit hook defers
    /// (we are inside a registry op), so we re-key here.
    fn flip_to_shared(&self, reg: &mut Registry, id: HeadId) {
        let h = reg.heads.get_mut(&id).expect("head exists");
        let old_tip = h.tip.clone();
        let (_opts, guard) = h.doc.implicit_commit_then_stop();
        let new_tip = h.doc.state_frontiers();
        h.doc.set_head_shared(true);
        h.tip = new_tip.clone();
        drop(guard);
        if old_tip != new_tip {
            if reg.by_tip.get(&old_tip) == Some(&id) {
                reg.by_tip.remove(&old_tip);
            }
            reg.by_tip.insert(new_tip, id);
        }
    }

    /// `2 -> 1`: the head is uniquely owned again; make it writable and renew
    /// its auto-commit transaction.
    fn flip_to_private(&self, reg: &mut Registry, id: HeadId) {
        let h = reg.heads.get(&id).expect("head exists");
        h.doc.set_head_shared(false);
        h.doc.renew_txn_if_auto_commit(None);
    }

    /// A non-root head reached `refs == 0`: retire it (remove from `by_tip` and
    /// `heads`, dropping its materialized state). The PINNED ROOT is never
    /// dropped -- it stays a resting head so `by_tip` / `materialize` / `import`
    /// / `export` always have an anchor over the shared op log. The head's
    /// already-COMMITTED ops remain in history, so its tip stays reachable by
    /// `materialize` replay.
    ///
    /// Drop-commit hazard: removing the head drops its `LoroDoc`, whose `Drop`
    /// implicit-commits an open auto-commit transaction. We ASSERT the head has
    /// no pending local ops here rather than clearing them: the base commits
    /// every `write()` before returning and only rebinds at operation
    /// boundaries, so a head reaching `refs == 0` always has an empty
    /// auto-commit txn. An empty commit inserts NO change (`Transaction::_commit`
    /// aborts on empty `local_ops`), so the ensuing drop is a clean no-op and
    /// appends nothing to shared history. Asserting (rather than committing)
    /// keeps a spurious out-of-contract pending op loud instead of silently
    /// polluting history.
    fn retire(&self, reg: &mut Registry, id: HeadId) {
        if id == ROOT_HEAD_ID {
            return; // pinned: keep the root as a resting head at refs == 0
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
        reg.heads.remove(&id); // drops the LoroDoc; empty txn => clean no-op commit
    }

    // --- binding -----------------------------------------------------------

    /// Bind branch `b` to `new_id`, adjusting refs (and thus shared/private
    /// mode) on both the old and new heads.
    pub(super) fn rebind(&self, reg: &mut Registry, b: &BranchId, new_id: HeadId) {
        if let Some(&old) = reg.bound.get(b) {
            if old == new_id {
                return;
            }
            self.dec_refs(reg, old);
        }
        reg.bound.insert(b.clone(), new_id);
        self.inc_refs(reg, new_id);
        self.reinstall_subs(reg, b, new_id);
    }

    /// Re-install branch `b`'s registry-owned subscriptions on head `new_id`
    /// (its new bound head after a rebind): drop each handle on the old head and
    /// re-subscribe the same callback on the new head, so a subscription placed
    /// before a copy-on-divergence keeps firing on the branch's live head.
    fn reinstall_subs(&self, reg: &mut Registry, b: &BranchId, new_id: HeadId) {
        let head = match reg.heads.get(&new_id) {
            Some(h) => h.doc.clone(),
            None => return,
        };
        if let Some(subs) = reg.subs.get_mut(b) {
            for s in subs.iter_mut() {
                s.installed = None; // unsubscribe from the old head
                let installed = match &s.target {
                    Some(cid) => head.subscribe(cid, s.cb.clone()),
                    None => head.subscribe_root(s.cb.clone()),
                };
                s.installed = Some(installed);
            }
        }
    }

    // --- branch-scoped dispatch --------------------------------------------

    /// Snapshot branch `b`'s subscriptions (container target + callback) so the
    /// diff of a transition can be delivered to them AFTER the registry lock
    /// drops. Taken under the lock; dispatched to outside it.
    pub(super) fn snapshot_subs(&self, reg: &Registry, b: &BranchId) -> SubsSnapshot {
        reg.subs
            .get(b)
            .map(|v| v.iter().map(|s| (s.target.clone(), s.cb.clone())).collect())
            .unwrap_or_default()
    }

    /// Deliver `events` to `subs` through a SCRATCH observer, reproducing a plain
    /// doc's root / ancestor-match filtering (`Observer::emit_inner`). Branch-
    /// scoped by construction: only this branch's callbacks are subscribed, so a
    /// shared destination head's co-owner never receives the jump (constraint
    /// C2). Runs AFTER the registry lock drops (constraint C3), so a callback may
    /// safely re-enter the registry. The `Subscription` handles are held until
    /// the last `emit` returns (dropping one unsubscribes its callback).
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

    // --- resolution --------------------------------------------------------

    /// Resolve branch `b` to the head it should use, copying-and-rebinding when
    /// a `Write` reaches a shared head. Returns the head id and a handle.
    ///
    /// A transition that moves `b`'s tip delivers exactly one `DocDiff` batch for
    /// `diff(old_tip -> new_tip)` to `b`'s subscribers (and nobody else), through
    /// `dispatch_to_branch` AFTER the registry lock drops. A resolve that does
    /// not move `b`'s tip delivers nothing.
    ///
    /// Low-level: hands back a raw `LoroDoc`; only the `OwnedHeadOp` gates and
    /// the sink guard protect it. Consumers go through `Branch`.
    #[doc(hidden)]
    pub fn resolve(&self, b: &BranchId, intent: Intent) -> LoroResult<(HeadId, LoroDoc)> {
        let (out, pending, subs) = self.resolve_collecting(b, intent)?;
        self.dispatch_to_branch(subs, pending);
        Ok(out)
    }

    /// The registry-locked core of `resolve`: picks an arm, moves `b`, and
    /// returns the head handle PLUS the pending diff and a subscription snapshot
    /// for the outer `resolve` to dispatch after the lock drops.
    fn resolve_collecting(
        &self,
        b: &BranchId,
        intent: Intent,
    ) -> LoroResult<((HeadId, LoroDoc), Vec<DocDiff>, SubsSnapshot)> {
        self.with_reg(|this, reg| {
            let target = this.inner.policy.target(this, b)?;
            let mut pending: Vec<DocDiff> = Vec::new();

            let mut id = match reg.bound.get(b).copied() {
                Some(h) if reg.heads[&h].tip == target => h,
                cur => match reg.by_tip.get(&target).copied() {
                    Some(h2) => {
                        // Rebind-to-existing arm: a head (h2) already rests at
                        // `target`, so the branch SHARES it instead of computing a
                        // diff. `rebind` is a pointer swap that delivers nothing,
                        // so synthesize `diff(X -> Y)` (X = the branch's old tip,
                        // Y = target) from a PRIVATE scratch copy of the old head
                        // and dispatch it to `b`'s subscribers ONLY. Emission
                        // through h2's shared observer is never correct here: h2's
                        // co-owner (a sibling branch) did not move (constraint C2).
                        if let Some(old) = cur {
                            let subs_nonempty =
                                reg.subs.get(b).map(|v| !v.is_empty()).unwrap_or(false);
                            if subs_nonempty && old != h2 {
                                // Copy `old` BEFORE `rebind`, which may retire it
                                // (`dec_refs` -> 0). Compute the diff on the scratch
                                // doc DIRECTLY, never via `advance_in_place`, whose
                                // `by_tip` re-key would clobber h2's key. The
                                // scratch is a private copy (refs == 0, absent from
                                // `by_tip`); `checkout_collecting_events` forces
                                // recording on it so the diff survives.
                                let scratch = this.copy_head(reg, old);
                                let scratch_doc = reg.heads[&scratch].doc.clone();
                                pending = scratch_doc.checkout_collecting_events(
                                    &target,
                                    "checkout".into(),
                                    EventTriggerKind::Checkout,
                                )?;
                                this.retire(reg, scratch);
                            }
                        }
                        this.rebind(reg, b, h2);
                        h2
                    }
                    None => match cur {
                        // Catch-up arm: the branch's bound LIVE head (refs == 1)
                        // advances IN PLACE to its policy-computed target (e.g.
                        // `after_import` moving a branch to its lineage join). The
                        // head becomes the branch's live writable head at the
                        // policy frontier; `advance_via` clears detached and
                        // renews the txn so a subsequent local write does not fail
                        // `AutoCommitNotStarted` (the import-then-record data-loss
                        // path). Registry-internal, so it bypasses the external
                        // E1-A `attach`/`checkout_to_latest` gate.
                        Some(h) if reg.heads[&h].refs == 1 => {
                            let (id2, ev) =
                                this.advance_via(reg, b, AdvanceSource::InPlace(h), &target, None)?;
                            pending = ev;
                            id2
                        }
                        // Copy+advance arm: a SHARED head (refs > 1) whose branch
                        // target moved off the shared tip. Copy it, rebind `b`
                        // onto the copy FIRST (re-installs subs, leaves the copy
                        // recording from the old tip), then advance the copy and
                        // deliver its diff. On error `b` is rebound back to `h`.
                        Some(h) => {
                            let (c, ev) = this.advance_via(
                                reg,
                                b,
                                AdvanceSource::CopyOf(h),
                                &target,
                                Some(h),
                            )?;
                            pending = ev;
                            c
                        }
                        // Materialize arm: a cold/unbound branch resolves to its
                        // policy frontier via a fresh copy of the pinned ROOT
                        // (the shared advance helper with `CopyOf(ROOT)`). Same
                        // rebind-first delivery; a cold content branch's first
                        // write on the materialized head must succeed, so it is
                        // handed back writable.
                        None => {
                            let (m, ev) = this.advance_via(
                                reg,
                                b,
                                AdvanceSource::CopyOf(ROOT_HEAD_ID),
                                &target,
                                None,
                            )?;
                            pending = ev;
                            m
                        }
                    },
                },
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
            Ok(((id, reg.heads[&id].doc.clone()), pending, subs))
        })
    }

    /// Move a head to `target` by applying `diff(tip, target)` (a checkout on the
    /// shared history), COLLECTING the resulting `DocDiff`s instead of emitting
    /// them through the head's observer. The caller decides whether to dispatch
    /// the returned events to the branch's subscribers (a `resolve` that moved
    /// the branch does; a scratch `read_at` discards them). Because the diff is
    /// collected with recording forced on at the pre-checkout frontier, this is
    /// correct even for a fresh, never-subscribed copy (the copy+advance and
    /// materialize arms), whose diff the old emit path discarded.
    fn advance_in_place(
        &self,
        reg: &mut Registry,
        id: HeadId,
        target: &Frontiers,
    ) -> LoroResult<Vec<DocDiff>> {
        let (old_tip, doc) = {
            let h = reg.heads.get(&id).expect("head exists");
            (h.tip.clone(), h.doc.clone())
        };
        if old_tip == *target {
            return Ok(Vec::new());
        }
        // NOTE(claude-opus-4-8/branchingdocrepo-multiheaddoc): `checkout` leaves
        // the head's `detached` flag set (state != the shared union) with its
        // auto-commit txn stopped. Clearing that is the CALLER's decision, not
        // this shared helper's, because it depends on whether the resulting head
        // is the branch's live writable head or a read-only view:
        //   - the resolve CATCH-UP arm (refs==1 bound live head) CLEARS it after
        //     this returns: the head is the branch's writable head at its policy
        //     target, and a local write must not fail `AutoCommitNotStarted`
        //     (the import-then-record data-loss path);
        //   - the resolve copy+advance arm (refs>1 shared head advanced onto a
        //     fresh copy) CLEARS after this returns, by the same live-writable
        //     reasoning (the copy is the branch's writable head at its policy
        //     target);
        //   - the resolve materialize arm (fresh head for a cold/unbound branch)
        //     ALSO CLEARS: a cold content branch's first write materializes to
        //     its live policy frontier and must be writable. (An explicit
        //     read-only history VIEW is `create_branch_at` / the checkout RFP,
        //     not this live-frontier resolve.)
        // In short: EVERY arm that hands back the branch's bound head at its
        // policy target clears; `advance_in_place` itself stays neutral.
        let events =
            doc.checkout_collecting_events(target, "checkout".into(), EventTriggerKind::Checkout)?;
        let new_tip = doc.state_frontiers();
        reg.heads.get_mut(&id).expect("head exists").tip = new_tip.clone();
        if reg.by_tip.get(&old_tip) == Some(&id) {
            reg.by_tip.remove(&old_tip);
        }
        // `or_insert`, NOT `insert`: a `read_at` scratch head advanced to a
        // frontier a LIVE head already rests at must not displace that head's
        // `by_tip` key (the following `retire(scratch)` would then remove it,
        // silently dropping the live head from `by_tip`). A `resolve` advance
        // only ever reaches here on a `by_tip` MISS for `target`, so `or_insert`
        // still installs the moved head as the resting owner in that case.
        reg.by_tip.entry(new_tip).or_insert(id);
        Ok(events)
    }

    /// Advance branch `b` to `target` via `src`, deliver-collecting the diff.
    /// Unifies the catch-up, copy+advance, and materialize arms.
    ///
    /// For a `CopyOf` source the copy is REBOUND to `b` BEFORE the advance: that
    /// re-installs `b`'s subscriptions on the copy (so its `DocState` records
    /// from the old tip) and moves refs; the advance then collects `diff(X -> Y)`
    /// even on a fresh, never-subscribed copy. Every advancing arm hands back a
    /// head that is the branch's live writable bound head at its policy target,
    /// so detached is cleared and the auto-commit txn renewed. On an advance
    /// error for a `CopyOf`, `b` is rebound to `fallback` (or unbound if it had
    /// no prior head), which retires the copy.
    fn advance_via(
        &self,
        reg: &mut Registry,
        b: &BranchId,
        src: AdvanceSource,
        target: &Frontiers,
        fallback: Option<HeadId>,
    ) -> LoroResult<(HeadId, Vec<DocDiff>)> {
        let (id, pending) = match src {
            AdvanceSource::InPlace(id) => {
                let pending = self.advance_in_place(reg, id, target)?;
                (id, pending)
            }
            AdvanceSource::CopyOf(src_id) => {
                let c = self.copy_head(reg, src_id);
                // Rebind FIRST: moves refs and re-installs subs on the copy.
                self.rebind(reg, b, c);
                match self.advance_in_place(reg, c, target) {
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

    /// SECONDARY path: a frontier no live head sits at. Build a head by copying
    /// the PINNED ROOT and replaying (checkout) to `target`. Logically correct,
    /// not cheap: no nearest-source scan and no warm/orphan reuse.
    ///
    /// NOTE(claude-opus-4-8/branchingdocrepo-multiheaddoc): nearest-source
    /// selection and periodic checkpoints that make this path cheap are DEFERRED
    /// to the persistent-DocState follow-up SoW
    /// (`cdocs/proposals/2026-09-14-multiheaddoc-api-snapshots-persistent-state.md`).
    /// The root always exists (pinned), so this is always available.
    fn materialize(
        &self,
        reg: &mut Registry,
        target: &Frontiers,
    ) -> LoroResult<(HeadId, Vec<DocDiff>)> {
        let c = self.copy_head(reg, ROOT_HEAD_ID);
        let events = self.advance_in_place(reg, c, target)?;
        Ok((c, events))
    }

    /// Advance branch `b`'s BOUND, uniquely-owned (`refs == 1`) head to `target`
    /// in place and leave it WRITABLE (detached cleared, auto-commit txn
    /// renewed), returning its `LoroDoc` so the caller can commit ops on it. This
    /// is exactly the move the resolve catch-up arm makes, exposed for a caller
    /// that computed `target` OUTSIDE the normal policy path -- specifically a
    /// repo-wide index merge, whose target is `join(tips[into], tips[from])` and
    /// whose next act is to commit one merge marker on the advanced head.
    ///
    /// Errors if `b` is unbound; the caller resolves `b` first (an eager index
    /// head is bound at `refs == 1` once resolved). Runs under the registry lock,
    /// so it must NOT be called from inside another registry op.
    pub(super) fn advance_bound_writable(
        &self,
        b: &BranchId,
        target: &Frontiers,
    ) -> LoroResult<LoroDoc> {
        let (doc, pending, subs) = self.with_reg(|this, reg| {
            let id = *reg.bound.get(b).ok_or_else(|| {
                LoroError::ArgErr(format!("cannot advance: branch '{b}' is not bound").into_boxed_str())
            })?;
            debug_assert_eq!(
                reg.heads[&id].refs, 1,
                "advance_bound_writable requires a uniquely-owned (refs == 1) head"
            );
            let pending = this.advance_in_place(reg, id, target)?;
            let doc = reg.heads[&id].doc.clone();
            // Same reasoning as the resolve catch-up arm: the head is now the
            // branch's live writable head at the merge join, so a following local
            // commit (the merge marker) must not fail `AutoCommitNotStarted`.
            doc.set_detached(false);
            doc.renew_txn_if_auto_commit(None);
            let subs = this.snapshot_subs(reg, b);
            Ok::<_, LoroError>((doc, pending, subs))
        })?;
        // Deliver the advance's diff after the registry lock drops.
        self.dispatch_to_branch(subs, pending);
        Ok(doc)
    }

    /// Read the state at an ARBITRARY frontier `target` without disturbing any
    /// bound head: materialize a throwaway scratch head (copy the pinned root,
    /// check it out to `target`), run `f` on it, then retire the scratch. Used by
    /// derived reads that must observe a doc's state at a frontier other than a
    /// branch's current tip (e.g. a fork-point diff). Read-only: `f` must not
    /// mutate (the scratch retires at `refs == 0`, which asserts no pending ops).
    pub(super) fn read_at<R>(
        &self,
        target: &Frontiers,
        f: impl FnOnce(&LoroDoc) -> R,
    ) -> LoroResult<R> {
        self.with_reg(|this, reg| {
            // A scratch read discards the collected diff: no branch moved, so
            // nothing is dispatched.
            let (scratch, _events) = this.materialize(reg, target)?;
            let out = f(&reg.heads[&scratch].doc);
            this.retire(reg, scratch);
            Ok(out)
        })
    }
}
