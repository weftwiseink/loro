//! `MultiHeadDoc<P>`: the shared base of `BranchingDocRepo`.
//!
//! One `OpLog` backs N materialized HEADS (each a `LoroDoc` / `LoroDocInner`
//! over that shared op log, with its own `DocState`, `DiffCalculator`, `Txn`
//! lock, observer, and peer). Branches are BOUND to heads, not owners of them:
//! a head bound to more than one branch is IMMUTABLE, and the first divergent
//! write COPIES it (copy-on-divergence). The registry (`heads` / `by_tip` /
//! `bound` / `refs`) and the copy/guard machinery are written ONCE here; the
//! two variants (`SelfRooted` index, `Delegated` content) differ only in a
//! [`HeadPolicy`], which is implemented in later phases. A test-only [`Manual`]
//! policy exercises this base standalone.
//!
//! The immutability of a shared head is enforced at loro's two `DocState`
//! mutation sinks (`apply_local_op`, `apply_diff`) via `head_mode` +
//! `ensure_private`, NOT by the closure API, so a mutation reaching a shared
//! head errors with [`LoroError::HeadShared`] instead of corrupting a
//! co-owner. See `crates/loro-internal/src/state.rs`.
//!
//! # Phase 1 scope
//!
//! This module is the Phase 1 foundational unit. It deliberately does NOT
//! implement the index (`SelfRooted`), content docs (`Delegated`), the
//! `BranchingDoc`/`Branch` wrapper, `__fs__`, wasm, or persistent DocState.
//! Branch-scoped subscription rebinding (`Registry::subs`) and true
//! history-only `import` belong to later phases and are flagged where stubbed.

use std::collections::VecDeque;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::arena::SharedArena;
use crate::configure::Configure;
use crate::encoding::ImportStatus;
use crate::lock::{LockKind, LoroLockGroup, LoroMutex};
use crate::oplog::OpLog;
use crate::state::DocState;
use crate::sync::{AtomicU8, AtomicUsize};
use crate::version::Frontiers;
use crate::{LoroDoc, HEAD_MODE_PRIVATE};
use loro_common::{InternalString, LoroError, LoroResult, ID};

pub type BranchId = InternalString;
pub type HeadId = u64;

/// When a `MultiHeadDoc` copies a head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyMode {
    /// Content docs: copy only when a shared head must diverge (lazy).
    OnDivergence,
    /// The index: copy at branch creation, so every head is born `refs == 1`.
    Eager,
}

/// Resolution intent: a `Write` on a shared head triggers copy-on-divergence; a
/// `Read` never copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    Read,
    Write,
}

/// The only axis on which the two `MultiHeadDoc` variants differ.
///
/// `target` says where a branch should be, `after_commit` publishes a new tip,
/// and `after_import` says which branches to rebind after history lands. The
/// base calls these; it never branches on which policy it holds.
pub trait HeadPolicy: Send + Sync + 'static + Sized {
    /// Copy discipline for this variant.
    const COPY: CopyMode;

    /// The frontier branch `b` should currently resolve to. Called while the
    /// registry lock is held, so it MUST NOT lock the registry.
    fn target(&self, this: &MultiHeadDoc<Self>, b: &BranchId) -> LoroResult<Frontiers>;

    /// Publish the new tip after `b`'s head commits `last`. May be a no-op (the
    /// index's tip IS its record).
    fn after_commit(&self, this: &MultiHeadDoc<Self>, b: &BranchId, last: ID);

    /// Given the spans that just landed on import, return the branches whose
    /// binding should be re-resolved.
    fn after_import(&self, this: &MultiHeadDoc<Self>, status: &ImportStatus) -> Vec<BranchId>;
}

/// A materialized state at one tip: a `LoroDoc` over the shared `OpLog`.
struct Head {
    doc: LoroDoc,
    tip: Frontiers,
    /// Number of `bound` entries pointing here. The whole sharing rule:
    /// `1` = uniquely owned (writable in place); `> 1` = shared (immutable);
    /// `0` = orphan.
    refs: usize,
}

/// The head registry. Guarded by a single `LockKind::BranchRegistry` lock,
/// acquired before any head lock so `resolve` can copy/rebind (taking head
/// `Txn`/`DocState` locks) while holding it, and never held into an actual
/// content op.
struct Registry {
    heads: FxHashMap<HeadId, Head>,
    /// Resting heads by tip: the sharing lookup. A freshly copied head is
    /// ABSENT until its first commit re-keys it (invariant I3).
    by_tip: FxHashMap<Frontiers, HeadId>,
    /// Branch -> the head it currently uses.
    bound: FxHashMap<BranchId, HeadId>,
    /// `refs == 0` heads, bounded by `warm_budget`, kept as nearest-source
    /// candidates then dropped.
    orphans: VecDeque<HeadId>,
    warm_budget: usize,
    next_id: HeadId,
}

impl Registry {
    fn alloc_id(&mut self) -> HeadId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

/// The shared base: one op log, the head registry, the copy machinery, the sink
/// guard. Written once; both variants build on it.
#[allow(missing_debug_implementations)] // holds LoroMutex<OpLog>/live heads; no useful Debug
pub struct MultiHeadDoc<P: HeadPolicy> {
    oplog: Arc<LoroMutex<OpLog>>,
    arena: SharedArena,
    config: Configure,
    lock_group: LoroLockGroup,
    visible_op_count: Arc<AtomicUsize>,
    reg: LoroMutex<Registry>,
    policy: P,
}

impl<P: HeadPolicy> MultiHeadDoc<P> {
    /// Open a fresh `MultiHeadDoc` with one empty root head. Nothing is bound
    /// yet; the consumer (or a policy) binds branches.
    pub fn new(policy: P) -> Self {
        let visible_op_count = Arc::new(AtomicUsize::new(0));
        let oplog = OpLog::new(visible_op_count.clone());
        let arena = oplog.arena.clone();
        let config: Configure = oplog.configure.clone();
        let lock_group = LoroLockGroup::new();
        let oplog = Arc::new(lock_group.new_lock(oplog, LockKind::OpLog));

        let head_mode = Arc::new(AtomicU8::new(HEAD_MODE_PRIVATE));
        let arena_c = arena.clone();
        let config_c = config.clone();
        let lg = lock_group.clone();
        let root = LoroDoc::build_head(
            oplog.clone(),
            arena.clone(),
            config.clone(),
            &lock_group,
            visible_op_count.clone(),
            head_mode,
            move |w, hm| {
                DocState::new_arc(
                    w.clone(),
                    arena_c.clone(),
                    config_c.clone(),
                    &lg,
                    hm.clone(),
                )
            },
            true,
        );
        let tip = root.state_frontiers();

        let mut heads = FxHashMap::default();
        let mut by_tip = FxHashMap::default();
        heads.insert(
            0,
            Head {
                doc: root,
                tip: tip.clone(),
                refs: 0,
            },
        );
        by_tip.insert(tip, 0);

        let reg = Registry {
            heads,
            by_tip,
            bound: FxHashMap::default(),
            orphans: VecDeque::new(),
            warm_budget: 8,
            next_id: 1,
        };

        MultiHeadDoc {
            oplog,
            arena,
            config,
            lock_group: lock_group.clone(),
            visible_op_count,
            reg: lock_group.new_lock(reg, LockKind::BranchRegistry),
            policy,
        }
    }

    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// The id of the empty root head created by [`new`](Self::new).
    pub fn root_head_id(&self) -> HeadId {
        // The root is always id 0.
        0
    }

    pub fn set_warm_budget(&self, orphans: usize) {
        self.reg.lock().warm_budget = orphans;
    }

    // --- head construction -------------------------------------------------

    /// Structurally copy a head: a new `LoroDocInner` over the SAME
    /// `Arc<OpLog>`, arena, config, and lock group, with `fork_in_group` for
    /// the state (fresh peer, fresh `DiffCalculator`, `Private`). The copy is
    /// inserted with `refs == 0` and is absent from `by_tip` (invariant I3)
    /// until a caller binds and commits it.
    fn copy_head(&self, reg: &mut Registry, src_id: HeadId) -> HeadId {
        let src_state = reg.heads[&src_id].doc.state.clone();
        let head_mode = Arc::new(AtomicU8::new(HEAD_MODE_PRIVATE));
        let arena = self.arena.clone();
        let config = self.config.clone();
        let lg = self.lock_group.clone();
        let doc = LoroDoc::build_head(
            self.oplog.clone(),
            arena.clone(),
            config.clone(),
            &self.lock_group,
            self.visible_op_count.clone(),
            head_mode,
            move |w, hm| {
                src_state.lock().fork_in_group(
                    w.clone(),
                    arena.clone(),
                    config.clone(),
                    &lg,
                    hm.clone(),
                )
            },
            true,
        );
        let tip = doc.state_frontiers();
        let id = reg.alloc_id();
        reg.heads.insert(id, Head { doc, tip, refs: 0 });
        id
    }

    // --- refs / mode flips -------------------------------------------------

    fn inc_refs(&self, reg: &mut Registry, id: HeadId) {
        // No longer an orphan if it was one.
        reg.orphans.retain(|o| *o != id);
        let refs = {
            let h = reg.heads.get_mut(&id).expect("head exists");
            h.refs += 1;
            h.refs
        };
        if refs == 2 {
            self.flip_to_shared(reg, id);
        }
    }

    fn dec_refs(&self, reg: &mut Registry, id: HeadId) {
        let refs = {
            let h = reg.heads.get_mut(&id).expect("head exists");
            h.refs -= 1;
            h.refs
        };
        if refs == 1 {
            self.flip_to_private(reg, id);
        } else if refs == 0 {
            self.orphan(reg, id);
        }
    }

    /// `1 -> 2`: commit any pending transaction (so a shared head never has an
    /// open transaction, invariant I2), re-key `by_tip` if that commit moved
    /// the tip, then mark the head `Shared`.
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

    fn orphan(&self, reg: &mut Registry, id: HeadId) {
        if let Some(h) = reg.heads.get(&id) {
            let tip = h.tip.clone();
            if reg.by_tip.get(&tip) == Some(&id) {
                reg.by_tip.remove(&tip);
            }
        }
        reg.orphans.push_back(id);
        while reg.orphans.len() > reg.warm_budget {
            if let Some(evict) = reg.orphans.pop_front() {
                reg.heads.remove(&evict);
            }
        }
    }

    // --- binding -----------------------------------------------------------

    /// Bind branch `b` to `new_id`, adjusting refs (and thus shared/private
    /// mode) on both the old and new heads. Sharing a head (`refs` rising to 2)
    /// flips it immutable; leaving one (`refs` falling to 1) flips it writable.
    fn rebind(&self, reg: &mut Registry, b: &BranchId, new_id: HeadId) {
        if let Some(&old) = reg.bound.get(b) {
            if old == new_id {
                return;
            }
            self.dec_refs(reg, old);
        }
        reg.bound.insert(b.clone(), new_id);
        self.inc_refs(reg, new_id);
    }

    /// Bind a branch directly to an existing head (test / low-level entry).
    pub fn bind(&self, b: &BranchId, head_id: HeadId) {
        let mut reg = self.reg.lock();
        self.rebind(&mut reg, b, head_id);
    }

    /// Create branch `new` bound to wherever `from` currently resolves: the
    /// free, O(1) branch-from-a-live-head path (share `from`'s head). For an
    /// `Eager` policy this is where the caller would instead copy; the base
    /// provides the shared-head bind and the policy decides.
    pub fn create_branch(&self, new: &BranchId, from: &BranchId) -> LoroResult<()> {
        let (from_head, _) = self.resolve(from, Intent::Read)?;
        let mut reg = self.reg.lock();
        if P::COPY == CopyMode::Eager {
            // The index's degenerate case: every head born refs == 1.
            let copy = self.copy_head(&mut reg, from_head);
            self.rebind(&mut reg, new, copy);
        } else {
            self.rebind(&mut reg, new, from_head);
        }
        Ok(())
    }

    // --- resolution --------------------------------------------------------

    /// Resolve branch `b` to the head it should use, copying-and-rebinding when
    /// a `Write` reaches a shared head. Returns the head id and a handle. The
    /// registry lock is released before the handle is used.
    pub fn resolve(&self, b: &BranchId, intent: Intent) -> LoroResult<(HeadId, LoroDoc)> {
        let mut reg = self.reg.lock();
        let target = self.policy.target(self, b)?;

        let mut id = match reg.bound.get(b).copied() {
            Some(h) if reg.heads[&h].tip == target => h,
            cur => match reg.by_tip.get(&target).copied() {
                // Another branch already rests at `target`: SHARE it, O(1).
                Some(h2) => {
                    self.rebind(&mut reg, b, h2);
                    h2
                }
                None => match cur {
                    Some(h) if reg.heads[&h].refs == 1 => {
                        self.advance_in_place(&mut reg, h, &target)?;
                        h
                    }
                    Some(h) => {
                        let c = self.copy_head(&mut reg, h);
                        self.advance_in_place(&mut reg, c, &target)?;
                        self.rebind(&mut reg, b, c);
                        c
                    }
                    // SECONDARY: nearest source + diff.
                    None => {
                        let m = self.materialize(&mut reg, &target)?;
                        self.rebind(&mut reg, b, m);
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
            let c = self.copy_head(&mut reg, id);
            self.rebind(&mut reg, b, c);
            id = c;
        }

        let doc = reg.heads[&id].doc.clone();
        Ok((id, doc))
    }

    /// Move a UNIQUELY-owned head to `target` by applying `diff(tip, target)`
    /// (a checkout on the shared history). Precondition: `refs == 1` (the
    /// caller copies first otherwise).
    fn advance_in_place(
        &self,
        reg: &mut Registry,
        id: HeadId,
        target: &Frontiers,
    ) -> LoroResult<()> {
        let h = reg.heads.get_mut(&id).expect("head exists");
        let old_tip = h.tip.clone();
        if old_tip == *target {
            return Ok(());
        }
        h.doc.checkout(target)?;
        let new_tip = h.doc.state_frontiers();
        h.tip = new_tip.clone();
        if reg.by_tip.get(&old_tip) == Some(&id) {
            reg.by_tip.remove(&old_tip);
        }
        reg.by_tip.insert(new_tip, id);
        Ok(())
    }

    /// SECONDARY path: build a head at `target` from the nearest existing head
    /// (or the root), then advance it. Reached only for a frontier no head is
    /// at. No checkpoint code (deferred).
    fn materialize(&self, reg: &mut Registry, target: &Frontiers) -> LoroResult<HeadId> {
        let src = reg
            .by_tip
            .values()
            .next()
            .copied()
            .or_else(|| reg.heads.keys().next().copied())
            .ok_or(LoroError::HeadShared)?; // no heads at all is impossible (root always exists)
        let c = self.copy_head(reg, src);
        self.advance_in_place(reg, c, target)?;
        Ok(c)
    }

    // --- read / write ------------------------------------------------------

    /// Resolve `b` for read (never copies) and run `f` on the resolved head. A
    /// mutation inside `f` on a shared head errors from the sink; it does not
    /// corrupt a co-owner.
    pub fn read<R>(&self, b: &BranchId, f: impl FnOnce(&LoroDoc) -> R) -> LoroResult<R> {
        let (_, doc) = self.resolve(b, Intent::Read)?;
        Ok(f(&doc))
    }

    /// Resolve `b` for write (copies if shared, handing back a `Private` head),
    /// run `f`, then commit and re-key `by_tip` + notify the policy.
    pub fn write<R>(&self, b: &BranchId, f: impl FnOnce(&LoroDoc) -> R) -> LoroResult<R> {
        let (id, doc) = self.resolve(b, Intent::Write)?;
        let r = f(&doc);
        doc.commit_then_renew();
        self.on_commit(b, id);
        Ok(r)
    }

    /// After a head commits: re-key `by_tip` to its new tip and let the policy
    /// publish it. `last` (the fresh op id) cannot collide in `by_tip`.
    fn on_commit(&self, b: &BranchId, id: HeadId) {
        let mut reg = self.reg.lock();
        let (old_tip, new_tip) = {
            let doc = reg.heads[&id].doc.clone();
            let new_tip = doc.state_frontiers();
            let h = reg.heads.get_mut(&id).expect("head exists");
            let old = std::mem::replace(&mut h.tip, new_tip.clone());
            (old, new_tip)
        };
        if old_tip != new_tip {
            if reg.by_tip.get(&old_tip) == Some(&id) {
                reg.by_tip.remove(&old_tip);
            }
            reg.by_tip.insert(new_tip.clone(), id);
        }
        drop(reg);
        if let Some(last) = new_tip.as_single() {
            self.policy.after_commit(self, b, last);
        }
    }

    // --- history barrier / import / export ---------------------------------

    /// Run `f` with EVERY head's transaction committed and stopped (the
    /// all-heads import barrier). Each head's auto-commit txn is renewed
    /// afterward. Reusable by `import` / `replace_history`.
    pub fn with_all_heads_barrier<R>(&self, f: impl FnOnce() -> R) -> R {
        let reg = self.reg.lock();
        let docs: Vec<LoroDoc> = reg.heads.values().map(|h| h.doc.clone()).collect();
        drop(reg);
        // Every head's `Txn` lock is the SAME `LockKind` in one shared group, so
        // the order checker forbids holding two at once. We therefore
        // commit-and-stop each head sequentially, dropping its txn guard (the
        // transaction stays `None` because we do not renew here), run `f` with
        // no head holding an open transaction, then renew each head's
        // auto-commit transaction afterward.
        //
        // NOTE(claude-opus-4-8/branchingdocrepo-multiheaddoc): dropping the
        // guards means the barrier alone does not block a concurrent thread from
        // starting a fresh txn on a head during `f`. Under Phase 1 that cannot
        // happen (single-writer, single-threaded tests); a multi-threaded
        // history-only import that needs a hard barrier is a later-phase concern
        // (it also needs `import_to_history`, not present on this branch).
        let mut renew = Vec::with_capacity(docs.len());
        for d in &docs {
            let (opts, guard) = d.implicit_commit_then_stop();
            drop(guard);
            renew.push((d.clone(), opts));
        }
        let r = f();
        for (d, opts) in renew {
            d.renew_txn_if_auto_commit(opts);
        }
        r
    }

    /// The number of live heads (test / diagnostics).
    pub fn head_count(&self) -> usize {
        self.reg.lock().heads.len()
    }

    /// A clone of a head's `LoroDoc` handle, by id (test / diagnostics; models
    /// a handle "leaked" out of a `read` closure).
    pub fn head_doc(&self, id: HeadId) -> Option<LoroDoc> {
        self.reg.lock().heads.get(&id).map(|h| h.doc.clone())
    }

    /// The head id a branch is currently bound to.
    pub fn bound_head(&self, b: &BranchId) -> Option<HeadId> {
        self.reg.lock().bound.get(b).copied()
    }

    /// The recorded tip of a head (test / diagnostics).
    pub fn head_tip(&self, id: HeadId) -> Option<Frontiers> {
        self.reg.lock().heads.get(&id).map(|h| h.tip.clone())
    }

    /// Whether `by_tip` maps `tip` to `id` (test: on_commit re-keying).
    pub fn by_tip_maps(&self, tip: &Frontiers, id: HeadId) -> bool {
        self.reg.lock().by_tip.get(tip) == Some(&id)
    }

    /// The refcount of the head a branch is bound to.
    pub fn refs_of(&self, b: &BranchId) -> Option<usize> {
        let reg = self.reg.lock();
        reg.bound
            .get(b)
            .and_then(|id| reg.heads.get(id))
            .map(|h| h.refs)
    }

    /// Whether the head a branch is bound to is currently shared (immutable).
    pub fn is_head_shared(&self, b: &BranchId) -> Option<bool> {
        let reg = self.reg.lock();
        reg.bound
            .get(b)
            .and_then(|id| reg.heads.get(id))
            .map(|h| h.doc.is_head_shared())
    }
}

/// A test-only policy that exercises the base's registry/copy/guard mechanics
/// without the real index or content policies. `target` is whatever the test
/// last recorded (default: the empty root frontier); `after_commit` advances a
/// branch's recorded target to its head's new tip so resolution stays stable.
#[allow(missing_debug_implementations)]
pub struct Manual {
    targets: std::sync::Mutex<FxHashMap<BranchId, Frontiers>>,
}

impl Default for Manual {
    fn default() -> Self {
        Self::new()
    }
}

impl Manual {
    pub fn new() -> Self {
        Manual {
            targets: std::sync::Mutex::new(FxHashMap::default()),
        }
    }

    /// Record where a branch should resolve to (test control).
    pub fn set_target(&self, b: &BranchId, target: Frontiers) {
        self.targets.lock().unwrap().insert(b.clone(), target);
    }
}

impl HeadPolicy for Manual {
    const COPY: CopyMode = CopyMode::OnDivergence;

    fn target(&self, _this: &MultiHeadDoc<Self>, b: &BranchId) -> LoroResult<Frontiers> {
        Ok(self
            .targets
            .lock()
            .unwrap()
            .get(b)
            .cloned()
            .unwrap_or_default())
    }

    fn after_commit(&self, _this: &MultiHeadDoc<Self>, b: &BranchId, last: ID) {
        self.targets
            .lock()
            .unwrap()
            .insert(b.clone(), Frontiers::from_id(last));
    }

    fn after_import(&self, _this: &MultiHeadDoc<Self>, _status: &ImportStatus) -> Vec<BranchId> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lock::{LockKind, LoroLockGroup};
    use std::sync::{Arc, Mutex};

    fn b(s: &str) -> BranchId {
        s.into()
    }

    fn tlen(md: &MultiHeadDoc<Manual>, branch: &BranchId) -> usize {
        md.read(branch, |d| d.get_text("t").len_unicode()).unwrap()
    }

    // ------------------------------------------------------------------
    // Registry / copy mechanics
    // ------------------------------------------------------------------

    #[test]
    fn free_creation_shares_head_no_copy() {
        let md = MultiHeadDoc::new(Manual::new());
        let (main, review) = (b("main"), b("review"));
        md.bind(&main, md.root_head_id());
        // create_branch from a live head is a bind, not a copy.
        md.create_branch(&review, &main).unwrap();
        assert_eq!(md.bound_head(&review), Some(0));
        assert_eq!(md.bound_head(&main), Some(0));
        assert_eq!(md.refs_of(&main), Some(2));
        assert_eq!(md.head_count(), 1, "no head was copied on creation");
        assert_eq!(md.is_head_shared(&main), Some(true));
    }

    #[test]
    fn copy_new_inner_fresh_peer_same_oplog() {
        let md = MultiHeadDoc::new(Manual::new());
        let (main, review) = (b("main"), b("review"));
        md.bind(&main, 0);
        md.bind(&review, 0); // shared
        md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
            .unwrap();
        let copy_id = md.bound_head(&main).unwrap();
        assert_ne!(copy_id, 0, "a divergent write minted a new head");
        let copy = md.head_doc(copy_id).unwrap();
        let root = md.head_doc(0).unwrap();
        assert_ne!(copy.peer_id(), root.peer_id(), "copy has a fresh peer");
        assert!(
            Arc::ptr_eq(&copy.oplog, &root.oplog),
            "copy shares the same Arc<OpLog>"
        );
    }

    #[test]
    fn on_commit_rekeys_by_tip_and_shares_via_lookup() {
        let md = MultiHeadDoc::new(Manual::new());
        let (main, other) = (b("main"), b("other"));
        md.bind(&main, 0);
        // In-place write on a uniquely-owned head; on_commit must re-key by_tip.
        md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
            .unwrap();
        let t = md.head_tip(0).unwrap();
        assert!(!t.is_empty(), "tip advanced after the write");
        assert!(md.by_tip_maps(&t, 0), "by_tip re-keyed to the new tip");
        assert!(
            !md.by_tip_maps(&Frontiers::default(), 0),
            "the old empty tip was vacated"
        );
        // An unbound branch whose target is that tip SHARES head 0 via the
        // by_tip lookup arm (no new head).
        md.policy().set_target(&other, t.clone());
        let (id, _) = md.resolve(&other, Intent::Read).unwrap();
        assert_eq!(id, 0);
        assert_eq!(md.head_count(), 1);
        assert_eq!(md.refs_of(&other), Some(2));
    }

    // ------------------------------------------------------------------
    // FLOOR: lock-group membership discriminator (+ fresh-group contrast)
    // ------------------------------------------------------------------

    #[test]
    fn lock_group_shared_docstate_before_oplog_panics() {
        // A copied head's DocState lock lives in the SHARED group with the root
        // head's OpLog lock, so a DocState-before-OpLog acquisition ACROSS the
        // two heads is caught by the debug order checker (kind 4 then kind 3).
        // This is the whole point of `fork_in_group`: both heads share one
        // ordering domain.
        let md = MultiHeadDoc::new(Manual::new());
        let (main, review) = (b("main"), b("review"));
        md.bind(&main, 0);
        md.bind(&review, 0);
        md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
            .unwrap();
        let copy = md.head_doc(md.bound_head(&main).unwrap()).unwrap();
        let root = md.head_doc(0).unwrap();

        // Suppress the expected panic's backtrace noise, then restore.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let copy_c = copy.clone();
        let root_c = root.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ds = copy_c.state.lock(); // DocState (kind 4)
            let _ol = root_c.oplog.lock(); // OpLog (kind 3) after DocState -> panic
        }));
        std::panic::set_hook(prev);
        assert!(
            result.is_err(),
            "DocState-before-OpLog across two shared-group heads must panic"
        );
        // The order-violation panic poisoned the copy's DocState mutex while its
        // guard was held; `forget` the whole doc so no `Drop` re-locks it (which
        // would double-panic). The clones above drop harmlessly (strong_count>1).
        std::mem::forget(md);
    }

    #[test]
    fn lock_group_fresh_group_docstate_before_oplog_does_not_panic() {
        // CONTRAST: the hazard `fork_in_group` fixes. A DocState lock built in a
        // FRESH group (as `fork_with_new_peer_id`'s bare mutex / a per-handle
        // group would be) is in a different ordering domain, so acquiring it
        // before the shared OpLog lock does NOT panic -- the inversion goes
        // unseen. This is exactly why a head must join the shared group.
        let md = MultiHeadDoc::new(Manual::new());
        md.bind(&b("main"), 0);
        let root = md.head_doc(0).unwrap();

        let fresh = LoroLockGroup::new();
        let foreign_docstate = fresh.new_lock(0u8, LockKind::DocState);
        let _ds = foreign_docstate.lock(); // DocState (kind 4) in a DIFFERENT domain
        let _ol = root.oplog.lock(); // no panic: different threadlocal
                                     // Reaching here without a panic is the assertion.
    }

    // ------------------------------------------------------------------
    // FLOOR: sink guard
    // ------------------------------------------------------------------

    #[test]
    fn sink_guard_mutation_in_read_closure_refuses_and_baseline_corrupts() {
        // GUARDED: a mutation inside a `read` closure on a shared head returns
        // HeadShared; the co-owner is untouched.
        let md = MultiHeadDoc::new(Manual::new());
        let (main, review) = (b("main"), b("review"));
        md.bind(&main, 0);
        md.bind(&review, 0); // shared, refs 2
        assert_eq!(md.is_head_shared(&main), Some(true));

        let res: LoroResult<()> = md
            .read(&main, |d| d.get_text("t").insert_unicode(0, "X"))
            .unwrap();
        assert!(
            matches!(res, Err(LoroError::HeadShared)),
            "mutation inside read on a shared head is refused, got {res:?}"
        );
        assert_eq!(tlen(&md, &review), 0, "co-owner untouched under the guard");

        // BASELINE (single variable = the guard flag): flip the SAME head to
        // Private (the pre-guard state where a refs>1 head was writable) and
        // the identical mutation now lands and CORRUPTS the co-owner.
        let leaked = md.head_doc(0).unwrap();
        leaked.set_head_shared(false);
        leaked.renew_txn_if_auto_commit(None);
        leaked.get_text("t").insert_unicode(0, "X").unwrap();
        assert_eq!(
            tlen(&md, &review),
            1,
            "baseline: unguarded shared-head write corrupts the co-owner"
        );
    }

    #[test]
    fn sink_guard_leaked_handle_refused_after_flip() {
        let md = MultiHeadDoc::new(Manual::new());
        let (main, review) = (b("main"), b("review"));
        md.bind(&main, 0); // refs 1, private
        let leaked = md.head_doc(0).unwrap();
        // A write while private is fine.
        leaked.get_text("t").insert_unicode(0, "A").unwrap();

        md.bind(&review, 0); // refs 2 -> Private->Shared flip (commits "A")
        assert_eq!(md.is_head_shared(&main), Some(true));

        // The leaked handle, used AFTER the flip, is refused at both sinks.
        let r_op = leaked.get_text("t").insert_unicode(0, "B");
        assert!(
            matches!(r_op, Err(LoroError::HeadShared)),
            "leaked local op after flip refused, got {r_op:?}"
        );
        let r_checkout = leaked.checkout(&Frontiers::default());
        assert!(
            matches!(r_checkout, Err(LoroError::HeadShared)),
            "leaked checkout (apply_diff) after flip refused, got {r_checkout:?}"
        );
        // The co-owner still reads only the pre-flip content ("A", len 1).
        // (`bind` is a low-level entry that does not update the Manual target,
        // so point `review` at the head's committed tip before reading, or
        // `resolve` would treat the stale empty target as authoritative.)
        md.policy().set_target(&review, md.head_tip(0).unwrap());
        assert_eq!(tlen(&md, &review), 1);
    }

    #[test]
    fn sink_guard_nested_mutation_in_event_callback_refused() {
        let md = Arc::new(MultiHeadDoc::new(Manual::new()));
        let (main, review, writer) = (b("main"), b("review"), b("writer"));
        md.bind(&main, 0);
        md.bind(&review, 0);
        md.bind(&writer, 0); // refs 3, head 0 shared

        // Give `writer` its own private head by diverging once.
        md.write(&writer, |d| d.get_text("t").insert_unicode(0, "1").unwrap())
            .unwrap();
        assert_eq!(md.is_head_shared(&main), Some(true), "head 0 still shared");
        let writer_doc = md.head_doc(md.bound_head(&writer).unwrap()).unwrap();

        // A callback on the writer's (private) head that, when it fires, tries
        // to mutate the SHARED head 0 -- a nested mutation from inside an event.
        let shared_head = md.head_doc(0).unwrap();
        let captured: Arc<Mutex<Option<LoroResult<()>>>> = Arc::new(Mutex::new(None));
        let cap2 = captured.clone();
        let sub = writer_doc.subscribe_root(Arc::new(move |_ev: crate::event::DiffEvent| {
            let r = shared_head.get_text("t").insert_unicode(0, "Z");
            *cap2.lock().unwrap() = Some(r);
        }));

        // Fire the callback with another write on the writer's own head.
        md.write(&writer, |d| d.get_text("t").insert_unicode(0, "2").unwrap())
            .unwrap();
        drop(sub);

        let got = captured.lock().unwrap().take();
        assert!(
            matches!(got, Some(Err(LoroError::HeadShared))),
            "nested mutation of a shared head inside an event callback refused, got {got:?}"
        );
        assert_eq!(tlen(&md, &main), 0, "the shared co-owner was not corrupted");
    }

    // ------------------------------------------------------------------
    // FLOOR: copy-then-write correctness
    // ------------------------------------------------------------------

    #[test]
    fn copy_then_write_coowners_unaffected() {
        let md = MultiHeadDoc::new(Manual::new());
        let (main, review, draft) = (b("main"), b("review"), b("draft"));
        md.bind(&main, 0);
        md.bind(&review, 0);
        md.bind(&draft, 0); // refs 3, shared
        assert_eq!(md.refs_of(&main), Some(3));
        assert_eq!(md.is_head_shared(&main), Some(true));

        // draft diverges.
        md.write(&draft, |d| d.get_text("t").insert_unicode(0, "D").unwrap())
            .unwrap();
        assert_ne!(md.bound_head(&draft), Some(0), "draft copied off head 0");
        assert_eq!(
            md.refs_of(&main),
            Some(2),
            "head 0 still shared by main + review"
        );
        assert_eq!(md.is_head_shared(&main), Some(true));

        assert_eq!(tlen(&md, &draft), 1, "writer sees its own op");
        assert_eq!(tlen(&md, &main), 0, "co-owner main unaffected");
        assert_eq!(tlen(&md, &review), 0, "co-owner review unaffected");

        // A subsequent op by the writer still does not touch the co-owners.
        md.write(&draft, |d| d.get_text("t").insert_unicode(1, "E").unwrap())
            .unwrap();
        assert_eq!(tlen(&md, &draft), 2);
        assert_eq!(tlen(&md, &main), 0);
        assert_eq!(tlen(&md, &review), 0);
    }

    // ------------------------------------------------------------------
    // All-heads barrier
    // ------------------------------------------------------------------

    #[test]
    fn all_heads_barrier_runs_over_every_head() {
        let md = MultiHeadDoc::new(Manual::new());
        let (main, review) = (b("main"), b("review"));
        md.bind(&main, 0);
        md.bind(&review, 0);
        md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
            .unwrap(); // 2 heads now
        assert_eq!(md.head_count(), 2);
        let ran = md.with_all_heads_barrier(|| 42);
        assert_eq!(ran, 42);
        // Heads remain usable afterward (txns renewed).
        md.write(&main, |d| d.get_text("t").insert_unicode(1, "B").unwrap())
            .unwrap();
        assert_eq!(tlen(&md, &main), 2);
    }
}
