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
//! This module is the foundational unit. It deliberately does NOT implement
//! the index (`SelfRooted`), content docs (`Delegated`), the
//! `BranchingDoc`/`Branch` wrapper, `__fs__`, wasm, or persistent DocState.
//! Branch-scoped subscription rebinding on `rebind` belongs to a later phase
//! (the base aggregates history / first-commit at the doc level instead).
//!
//! On-commit re-keying of `by_tip` and `HeadPolicy::after_commit` run through
//! the INJECTED txn on-commit hook (see [`DocOwner`] / `on_head_committed`),
//! not a synchronous call from `write`. `import` is history-only: ops enter the
//! shared op log via `LoroDoc::import_to_history` under the all-heads barrier,
//! never materializing into a live head.

use std::cell::Cell;
use std::cmp::Ordering;
use std::ops::Deref;
use std::sync::{Arc, OnceLock, Weak};

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::arena::SharedArena;
use crate::change::Timestamp;
use crate::configure::Configure;
use crate::container::IntoContainerId;
use crate::encoding::{ExportMode, ImportStatus};
use crate::event::Index;
#[cfg(feature = "counter")]
use crate::handler::counter::CounterHandler;
use crate::handler::{
    ListHandler, MapHandler, MovableListHandler, TextHandler, TreeHandler, ValueOrHandler,
};
use crate::lock::{LockKind, LoroLockGroup, LoroMutex};
use crate::loro::CommitOptions;
use crate::oplog::OpLog;
use crate::pre_commit::{
    FirstCommitFromPeerCallback, FirstCommitFromPeerPayload, PreCommitCallback,
};
use crate::state::DocState;
use crate::subscription::Subscriber;
use crate::sync::{AtomicU64, AtomicU8, AtomicUsize};
use crate::undo::DiffBatch;
use crate::utils::subscription::{SubscriberSetWithQueue, Subscription};
use crate::version::{shrink_frontiers, Frontiers, VersionVector};
use crate::{DocOwner, LoroDoc, HEAD_MODE_PRIVATE};
use loro_common::{
    ContainerID, Counter, IdSpan, InternalString, LoroEncodeError, LoroError, LoroResult,
    LoroValue, PeerID, ID,
};

pub type BranchId = InternalString;
pub type DocId = InternalString;
pub type HeadId = u64;

/// The pinned root head's id. Created at `refs == 0` and NEVER dropped: it is
/// the anchor `import` / `export` / `materialize` lean on (they need at least
/// one live head over the shared op log to operate through).
///
/// Refcount invariant for every OTHER head: a head is live while `refs >= 1`,
/// and is retired (removed from `heads` and `by_tip`) the moment `refs` reaches
/// `0`. There is no `refs == 0` resting/orphan state except the pinned root.
const ROOT_HEAD_ID: HeadId = 0;

/// One event per new change landing in the shared history (a head commit or an
/// import), carrying that change's update bytes. Mirrors `LocalUpdateCallback`.
pub type HistoryCallback = Box<dyn Fn(&Vec<u8>) -> bool + Send + Sync + 'static>;

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
    /// index's tip IS its record). Runs OUTSIDE the registry lock.
    fn after_commit(&self, this: &MultiHeadDoc<Self>, b: &BranchId, last: ID);

    /// Given the spans that just landed on import, return the branches whose
    /// binding should be re-resolved.
    fn after_import(&self, this: &MultiHeadDoc<Self>, status: &ImportStatus) -> Vec<BranchId>;
}

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

struct RegOpGuard;
impl RegOpGuard {
    fn enter() -> Self {
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
struct Head {
    doc: LoroDoc,
    tip: Frontiers,
    /// Number of `bound` entries pointing here. The whole sharing rule:
    /// `1` = uniquely owned (writable in place); `> 1` = shared (immutable);
    /// `0` = retired (removed) -- except the pinned root, which rests at `0`.
    refs: usize,
    /// Per-head forwarders that feed this head's local commits / first-commits
    /// into the doc-level `history_subs` / `first_commit_subs`. Kept alive for
    /// the head's lifetime; dropped (unsubscribed) when the head is evicted.
    _forward: Vec<Subscription>,
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
    /// Branch-scoped subscriptions, registry-owned so they can be RE-INSTALLED
    /// on the branch's new head across a rebind (copy-on-divergence / merge),
    /// instead of going silent on the old head.
    subs: FxHashMap<BranchId, Vec<BranchSub>>,
    next_id: HeadId,
    next_sub_id: u64,
}

/// A registry-owned branch subscription: its callback re-installed on the
/// branch's head whenever the branch rebinds. `installed` is the handle on the
/// CURRENT head (dropped -> unsubscribed) and is replaced on each rebind.
struct BranchSub {
    id: u64,
    /// The container to watch, or `None` for a root subscription.
    target: Option<ContainerID>,
    cb: Subscriber,
    installed: Option<Subscription>,
}

impl Registry {
    fn alloc_id(&mut self) -> HeadId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

/// The shared innards of a `MultiHeadDoc`, held behind an `Arc` so a head's
/// injected on-commit hook can hold a `Weak` back to it (see `HeadOwner`).
#[allow(missing_debug_implementations)] // holds LoroMutex<OpLog>/live heads; no useful Debug
pub struct MultiHeadInner<P: HeadPolicy> {
    oplog: Arc<LoroMutex<OpLog>>,
    arena: SharedArena,
    config: Configure,
    lock_group: LoroLockGroup,
    visible_op_count: Arc<AtomicUsize>,
    reg: LoroMutex<Registry>,
    policy: P,
    /// Head whose materialized state a `Snapshot` export carries (default: the
    /// root head). Ops/history export is head-independent (shared op log).
    snapshot_head: AtomicU64,
    /// One event per new change in the shared history, aggregated over heads and
    /// imports.
    history_subs: SubscriberSetWithQueue<(), HistoryCallback, Vec<u8>>,
    /// One event per first commit from a peer, aggregated over heads. Every head
    /// copy mints a fresh peer, so this fires once per write slot.
    first_commit_subs:
        SubscriberSetWithQueue<(), FirstCommitFromPeerCallback, FirstCommitFromPeerPayload>,
}

/// The shared base: one op log, the head registry, the copy machinery, the sink
/// guard. Written once; both variants build on it. Cheap to clone (an `Arc`).
#[allow(missing_debug_implementations)]
pub struct MultiHeadDoc<P: HeadPolicy> {
    inner: Arc<MultiHeadInner<P>>,
}

impl<P: HeadPolicy> Clone for MultiHeadDoc<P> {
    fn clone(&self) -> Self {
        MultiHeadDoc {
            inner: self.inner.clone(),
        }
    }
}

impl<P: HeadPolicy> Deref for MultiHeadDoc<P> {
    type Target = MultiHeadInner<P>;
    fn deref(&self) -> &MultiHeadInner<P> {
        &self.inner
    }
}

/// The registry back-pointer installed on each head. The txn commit path calls
/// `on_head_commit` after the head's locks drop; we re-key the tip index and
/// notify the policy from there.
struct HeadOwner<P: HeadPolicy> {
    inner: Weak<MultiHeadInner<P>>,
    head_id: HeadId,
}

impl<P: HeadPolicy> DocOwner for HeadOwner<P> {
    fn on_head_commit(&self, _id_span: IdSpan) {
        if let Some(inner) = self.inner.upgrade() {
            MultiHeadDoc { inner }.on_head_committed(self.head_id);
        }
    }
}

/// Install the per-head forwarders that feed a head's local commits and
/// first-commit-from-peer events into the doc-level aggregated subscriptions.
fn install_forwarders<P: HeadPolicy>(
    inner: &Weak<MultiHeadInner<P>>,
    doc: &LoroDoc,
) -> Vec<Subscription> {
    let w1 = inner.clone();
    let s1 = doc.subscribe_local_update(Box::new(move |bytes| {
        if let Some(inner) = w1.upgrade() {
            inner.history_subs.emit(&(), bytes.clone());
        }
        true
    }));
    let w2 = inner.clone();
    let s2 = doc.subscribe_first_commit_from_peer(Box::new(move |payload| {
        if let Some(inner) = w2.upgrade() {
            inner.first_commit_subs.emit(&(), payload.clone());
        }
        true
    }));
    vec![s1, s2]
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

        let inner = Arc::new_cyclic(|w: &Weak<MultiHeadInner<P>>| {
            // Build the empty root head (id 0) over the shared op log, with its
            // registry back-pointer and history forwarders installed.
            let head_mode = Arc::new(AtomicU8::new(HEAD_MODE_PRIVATE));
            let arena_c = arena.clone();
            let config_c = config.clone();
            let lg = lock_group.clone();
            let owner: Arc<dyn DocOwner> = Arc::new(HeadOwner {
                inner: w.clone(),
                head_id: 0,
            });
            let root = LoroDoc::build_head(
                oplog.clone(),
                arena.clone(),
                config.clone(),
                &lock_group,
                visible_op_count.clone(),
                head_mode,
                Some(owner),
                move |cyclic, hm| {
                    DocState::new_arc(
                        cyclic.clone(),
                        arena_c.clone(),
                        config_c.clone(),
                        &lg,
                        hm.clone(),
                    )
                },
                true,
            );
            let forward = install_forwarders(w, &root);
            let tip = root.state_frontiers();

            let mut heads = FxHashMap::default();
            let mut by_tip = FxHashMap::default();
            // The pinned root head: born at refs == 0 and never retired.
            heads.insert(
                ROOT_HEAD_ID,
                Head {
                    doc: root,
                    tip: tip.clone(),
                    refs: 0,
                    _forward: forward,
                },
            );
            by_tip.insert(tip, ROOT_HEAD_ID);

            MultiHeadInner {
                oplog: oplog.clone(),
                arena: arena.clone(),
                config: config.clone(),
                lock_group: lock_group.clone(),
                visible_op_count: visible_op_count.clone(),
                reg: lock_group.new_lock(
                    Registry {
                        heads,
                        by_tip,
                        bound: FxHashMap::default(),
                        subs: FxHashMap::default(),
                        next_id: ROOT_HEAD_ID + 1,
                        next_sub_id: 0,
                    },
                    LockKind::BranchRegistry,
                ),
                policy,
                snapshot_head: AtomicU64::new(0),
                history_subs: SubscriberSetWithQueue::new(),
                first_commit_subs: SubscriberSetWithQueue::new(),
            }
        });

        MultiHeadDoc { inner }
    }

    pub fn policy(&self) -> &P {
        &self.inner.policy
    }

    /// The id of the pinned root head created by [`new`](Self::new).
    pub fn root_head_id(&self) -> HeadId {
        ROOT_HEAD_ID
    }

    /// The head whose state a `Snapshot` export carries.
    pub fn set_snapshot_head(&self, id: HeadId) {
        self.snapshot_head
            .store(id, std::sync::atomic::Ordering::Release);
    }

    /// Run `f` under the registry lock, marking the current thread as inside a
    /// registry op for the duration (so an on-commit hook fired by a
    /// `flip_to_shared` flush defers rather than re-entering the lock).
    fn with_reg<R>(&self, f: impl FnOnce(&Self, &mut Registry) -> R) -> R {
        let _g = RegOpGuard::enter();
        let mut reg = self.reg.lock();
        f(self, &mut reg)
    }

    // --- head construction -------------------------------------------------

    /// Structurally copy a head: a new `LoroDocInner` over the SAME
    /// `Arc<OpLog>`, arena, config, and lock group, with `fork_in_group` for
    /// the state (fresh peer, fresh `DiffCalculator`, `Private`), its registry
    /// back-pointer and history forwarders installed. Inserted with `refs == 0`
    /// and absent from `by_tip` (invariant I3) until a caller binds/commits it.
    fn copy_head(&self, reg: &mut Registry, src_id: HeadId) -> HeadId {
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

    fn dec_refs(&self, reg: &mut Registry, id: HeadId) {
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
    fn rebind(&self, reg: &mut Registry, b: &BranchId, new_id: HeadId) {
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

    /// Bind a branch directly to an existing head (test / low-level entry).
    pub fn bind(&self, b: &BranchId, head_id: HeadId) {
        self.with_reg(|this, reg| this.rebind(reg, b, head_id));
    }

    /// Unbind a branch: drop its `bound` entry and decrement its head's refs
    /// (retiring the head if it reaches 0, unless it is the pinned root). Leaves
    /// no dangling `bound`/`by_tip` entry. No-op if the branch is not bound
    /// (a content doc binds a branch only lazily on first access).
    pub fn unbind(&self, b: &BranchId) {
        self.with_reg(|this, reg| {
            if let Some(old) = reg.bound.remove(b) {
                this.dec_refs(reg, old);
            }
        });
    }

    /// Install a registry-owned, rebind-surviving subscription on branch `b`
    /// (`target = None` for a root subscription). It fires on `b`'s CURRENT head
    /// and is re-installed on the new head whenever `b` rebinds. Dropping the
    /// returned [`BranchSubscription`] removes it.
    #[doc(hidden)]
    pub fn subscribe_branch(
        &self,
        b: &BranchId,
        target: Option<ContainerID>,
        cb: Subscriber,
    ) -> LoroResult<BranchSubscription> {
        let (_, head) = self.resolve(b, Intent::Read)?;
        let installed = match &target {
            Some(cid) => head.subscribe(cid, cb.clone()),
            None => head.subscribe_root(cb.clone()),
        };
        let sub_id = self.with_reg(|_, reg| {
            let sub_id = reg.next_sub_id;
            reg.next_sub_id += 1;
            reg.subs.entry(b.clone()).or_default().push(BranchSub {
                id: sub_id,
                target,
                cb,
                installed: Some(installed),
            });
            sub_id
        });
        let weak = Arc::downgrade(&self.inner);
        let branch = b.clone();
        Ok(BranchSubscription {
            remove: Some(Box::new(move || {
                if let Some(inner) = weak.upgrade() {
                    MultiHeadDoc { inner }.with_reg(|_, reg| {
                        if let Some(v) = reg.subs.get_mut(&branch) {
                            v.retain(|s| s.id != sub_id);
                        }
                    });
                }
            })),
        })
    }

    /// Create branch `new` bound to wherever `from` currently resolves: the
    /// free, O(1) branch-from-a-live-head path (share `from`'s head).
    ///
    /// This is the `OnDivergence` (content-doc) path only. An `Eager` policy
    /// (the index) provides its OWN creation that additionally seeds lineage,
    /// so the eager copy never leaks into this generic / `Delegated` path (see
    /// `MultiHeadDoc<SelfRooted>::create_index_branch`).
    pub fn create_branch(&self, new: &BranchId, from: &BranchId) -> LoroResult<()> {
        debug_assert!(
            P::COPY == CopyMode::OnDivergence,
            "eager policies must use their own branch creation (e.g. create_index_branch)"
        );
        let (from_head, _) = self.resolve(from, Intent::Read)?;
        self.with_reg(|this, reg| this.rebind(reg, new, from_head));
        Ok(())
    }

    // --- resolution --------------------------------------------------------

    /// Resolve branch `b` to the head it should use, copying-and-rebinding when
    /// a `Write` reaches a shared head. Returns the head id and a handle.
    ///
    /// Low-level: hands back a raw `LoroDoc`; only the `OwnedHeadOp` gates and
    /// the sink guard protect it. Consumers go through `Branch`.
    #[doc(hidden)]
    pub fn resolve(&self, b: &BranchId, intent: Intent) -> LoroResult<(HeadId, LoroDoc)> {
        self.with_reg(|this, reg| {
            let target = this.inner.policy.target(this, b)?;

            let mut id = match reg.bound.get(b).copied() {
                Some(h) if reg.heads[&h].tip == target => h,
                cur => match reg.by_tip.get(&target).copied() {
                    Some(h2) => {
                        this.rebind(reg, b, h2);
                        h2
                    }
                    None => match cur {
                        Some(h) if reg.heads[&h].refs == 1 => {
                            // Catch-up arm: the branch's bound LIVE head (refs==1)
                            // advances IN PLACE to its policy-computed target
                            // (e.g. `after_import` moving a branch to its lineage
                            // join). `advance_in_place` checks out, which leaves
                            // the head `detached` with its auto-commit txn stopped
                            // -> a subsequent local write (record_head after a
                            // remote import) would fail `AutoCommitNotStarted` and
                            // silently drop the record. The head is now the
                            // branch's live writable head sitting at the frontier
                            // the policy chose, so re-enable editing. This is a
                            // registry-internal move, so it correctly bypasses the
                            // external E1-A `attach`/`checkout_to_latest` gate.
                            this.advance_in_place(reg, h, &target)?;
                            let doc = &reg.heads[&h].doc;
                            doc.set_detached(false);
                            doc.renew_txn_if_auto_commit(None);
                            h
                        }
                        Some(h) => {
                            // Copy+advance arm: a SHARED head (refs>1) whose
                            // branch target moved off the shared tip. Copy it,
                            // advance the copy to the policy target, and hand it
                            // to the branch as its live writable head. Like the
                            // catch-up arm, `advance_in_place` checks out and
                            // leaves the copy detached with its txn stopped, so a
                            // subsequent local write would fail
                            // `AutoCommitNotStarted` (data loss). Clear it here in
                            // the caller (registry-internal, bypasses the E1-A
                            // gate). See the `advance_in_place` NOTE.
                            let c = this.copy_head(reg, h);
                            this.advance_in_place(reg, c, &target)?;
                            let doc = &reg.heads[&c].doc;
                            doc.set_detached(false);
                            doc.renew_txn_if_auto_commit(None);
                            this.rebind(reg, b, c);
                            c
                        }
                        None => {
                            // Materialize arm: a cold/unbound branch resolves to
                            // its policy frontier via a fresh head. `materialize`
                            // checks out (detached). This head becomes the
                            // branch's live bound head at its policy target, so a
                            // first WRITE on it (a cold content branch created
                            // then diverged after its parent moved on) must
                            // succeed. Clear detached here too, unifying all three
                            // advancing arms: any head handed back as the branch's
                            // bound head at its policy target is writable. (This
                            // extends the round-6 determination, which found
                            // "materialize read-only" only because the index/base
                            // never wrote to a materialized head; content branches
                            // do.)
                            let m = this.materialize(reg, &target)?;
                            let doc = &reg.heads[&m].doc;
                            doc.set_detached(false);
                            doc.renew_txn_if_auto_commit(None);
                            this.rebind(reg, b, m);
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

            Ok((id, reg.heads[&id].doc.clone()))
        })
    }

    /// Move a UNIQUELY-owned head to `target` by applying `diff(tip, target)`
    /// (a checkout on the shared history). Precondition: `refs == 1`.
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
        h.doc.checkout(target)?;
        let new_tip = h.doc.state_frontiers();
        h.tip = new_tip.clone();
        if reg.by_tip.get(&old_tip) == Some(&id) {
            reg.by_tip.remove(&old_tip);
        }
        reg.by_tip.insert(new_tip, id);
        Ok(())
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
    fn materialize(&self, reg: &mut Registry, target: &Frontiers) -> LoroResult<HeadId> {
        let c = self.copy_head(reg, ROOT_HEAD_ID);
        self.advance_in_place(reg, c, target)?;
        Ok(c)
    }

    // --- read / write ------------------------------------------------------

    /// Resolve `b` for read (never copies) and run `f` on the resolved head. A
    /// mutation inside `f` on a shared head errors from the sink; on a Private
    /// head it is a normal local edit, COMMITTED on exit (proposal L375). The
    /// exit-commit makes the retire-time "no pending ops" assertion a true
    /// invariant: no uncommitted op survives to a `retire`.
    ///
    /// Low-level: hands back a raw `LoroDoc`. The consumer surface is
    /// `Branch::read`, which wraps this and hands a `BranchingDocHead`.
    #[doc(hidden)]
    pub fn read<R>(&self, b: &BranchId, f: impl FnOnce(&LoroDoc) -> R) -> LoroResult<R> {
        let (_, doc) = self.resolve(b, Intent::Read)?;
        let r = f(&doc);
        // Commit-on-exit: a mutation inside a read closure on a Private head is
        // committed as a normal local edit (the injected on-commit hook re-keys
        // `by_tip`). A Shared head refused the op at the sink, so nothing is
        // pending there.
        if doc.get_pending_txn_len() > 0 {
            doc.commit_then_renew();
        }
        Ok(r)
    }

    /// Resolve `b` for write (copies if shared, handing back a `Private` head),
    /// run `f`, then commit. The commit fires the injected on-commit hook
    /// (`on_head_committed`), which re-keys `by_tip` and notifies the policy;
    /// there is no synchronous re-keying here.
    ///
    /// Low-level: hands back a raw `LoroDoc`. The consumer surface is
    /// `Branch::write`.
    #[doc(hidden)]
    pub fn write<R>(&self, b: &BranchId, f: impl FnOnce(&LoroDoc) -> R) -> LoroResult<R> {
        let (_, doc) = self.resolve(b, Intent::Write)?;
        let r = f(&doc);
        doc.commit_then_renew();
        Ok(r)
    }

    /// The injected txn on-commit hook target. Runs after the head's locks drop
    /// (so it may lock the registry), unless the commit was triggered from
    /// inside a registry op (a `flip_to_shared` flush), in which case that op
    /// re-keys `by_tip` itself and this defers.
    fn on_head_committed(&self, id: HeadId) {
        if in_registry_op() {
            return;
        }
        let (new_tip, branch) = {
            let _g = RegOpGuard::enter();
            let mut reg = self.reg.lock();
            let Some(head) = reg.heads.get(&id) else {
                return;
            };
            let new_tip = head.doc.state_frontiers();
            let old_tip = std::mem::replace(
                &mut reg.heads.get_mut(&id).expect("head exists").tip,
                new_tip.clone(),
            );
            if old_tip != new_tip {
                if reg.by_tip.get(&old_tip) == Some(&id) {
                    reg.by_tip.remove(&old_tip);
                }
                reg.by_tip.insert(new_tip.clone(), id);
            }
            // Reverse lookup: only a refs == 1 head commits, so at most one
            // branch is bound here.
            let branch = reg
                .bound
                .iter()
                .find(|(_, h)| **h == id)
                .map(|(b, _)| b.clone());
            (new_tip, branch)
        };
        if let (Some(b), Some(last)) = (branch, new_tip.as_single()) {
            self.inner.policy.after_commit(self, &b, last);
        }
    }

    // --- history barrier / import / export ---------------------------------

    /// Run `f` with EVERY head's transaction committed and stopped (the
    /// all-heads import barrier), then renew each head's auto-commit txn.
    ///
    /// Every head's `Txn` lock is the SAME `LockKind` in one shared group, so
    /// the order checker forbids holding two at once; we therefore commit-stop
    /// each head sequentially (dropping its guard; the txn stays `None` because
    /// we do not renew yet), run `f`, then renew.
    pub fn with_all_heads_barrier<R>(&self, f: impl FnOnce() -> R) -> R {
        let docs: Vec<LoroDoc> = {
            let reg = self.reg.lock();
            reg.heads.values().map(|h| h.doc.clone()).collect()
        };
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

    /// History-only import: land ops into the shared `OpLog` WITHOUT moving any
    /// head's state or binding (all heads barriered). No co-owner of a shared
    /// head can be corrupted; heads observe the ops only when the registry
    /// advances them. Emits a history event with the imported bytes.
    pub fn import(&self, bytes: &[u8]) -> LoroResult<ImportStatus> {
        let head = {
            let reg = self.reg.lock();
            reg.heads
                .values()
                .next()
                .map(|h| h.doc.clone())
                .expect("the root head always exists")
        };
        let status = self.with_all_heads_barrier(|| head.import_to_history(bytes))?;
        // Drive policy-directed rebinding of the branches whose recorded history
        // just landed (the ingest). For `SelfRooted` this discovers remote
        // branches from the lineage scan and advances affected index heads; for
        // policies that track no remote binding (e.g. `Manual`) it is empty.
        // A branch whose ids are not yet fully held is skipped and picked up on
        // the next import (resolve errors are non-fatal here).
        let touched = self.inner.policy.after_import(self, &status);
        for b in touched {
            let _ = self.resolve(&b, Intent::Read);
        }
        if !self.history_subs.inner().is_empty() {
            self.history_subs.emit(&(), bytes.to_vec());
        }
        Ok(status)
    }

    /// Export the shared history. `Snapshot` additionally carries the
    /// `snapshot_head`'s materialized state (default: the root head); all other
    /// modes are head-independent (the op log is shared).
    pub fn export(&self, mode: ExportMode) -> Result<Vec<u8>, LoroEncodeError> {
        let want = self
            .snapshot_head
            .load(std::sync::atomic::Ordering::Acquire);
        let doc = {
            let reg = self.reg.lock();
            reg.heads
                .get(&want)
                .or_else(|| reg.heads.values().next())
                .map(|h| h.doc.clone())
                .expect("the root head always exists")
        };
        doc.export(mode)
    }

    /// One event per new change in the shared history (any head commit or an
    /// import), carrying that change's update bytes.
    pub fn subscribe_history(&self, callback: HistoryCallback) -> Subscription {
        let (sub, enable) = self.history_subs.inner().insert((), callback);
        enable();
        sub
    }

    /// One event per first commit from a peer, aggregated over all heads. Every
    /// head copy mints a fresh peer, so this fires once per write slot.
    pub fn subscribe_first_commit_from_peer(
        &self,
        callback: FirstCommitFromPeerCallback,
    ) -> Subscription {
        let (sub, enable) = self.first_commit_subs.inner().insert((), callback);
        enable();
        sub
    }

    // --- accessors (test / diagnostics) ------------------------------------

    pub fn head_count(&self) -> usize {
        self.reg.lock().heads.len()
    }

    /// Low-level: a raw `LoroDoc` head handle (test / diagnostics; models a
    /// handle "leaked" out of a closure). Consumers go through `Branch`.
    #[doc(hidden)]
    pub fn head_doc(&self, id: HeadId) -> Option<LoroDoc> {
        self.reg.lock().heads.get(&id).map(|h| h.doc.clone())
    }

    pub fn bound_head(&self, b: &BranchId) -> Option<HeadId> {
        self.reg.lock().bound.get(b).copied()
    }

    pub fn head_tip(&self, id: HeadId) -> Option<Frontiers> {
        self.reg.lock().heads.get(&id).map(|h| h.tip.clone())
    }

    pub fn by_tip_maps(&self, tip: &Frontiers, id: HeadId) -> bool {
        self.reg.lock().by_tip.get(tip) == Some(&id)
    }

    pub fn refs_of(&self, b: &BranchId) -> Option<usize> {
        let reg = self.reg.lock();
        reg.bound
            .get(b)
            .and_then(|id| reg.heads.get(id))
            .map(|h| h.refs)
    }

    pub fn is_head_shared(&self, b: &BranchId) -> Option<bool> {
        let reg = self.reg.lock();
        reg.bound
            .get(b)
            .and_then(|id| reg.heads.get(id))
            .map(|h| h.doc.is_head_shared())
    }

    /// A handle to branch `name`: a NAME, resolved per operation. This is the
    /// consumer surface; every op re-resolves because the head a branch uses
    /// can change underneath the handle (a copy, a rebind).
    pub fn branch(&self, name: impl Into<BranchId>) -> Branch<'_, P> {
        Branch {
            doc: self,
            name: name.into(),
        }
    }
}

// ======================================================================
// The consumer surface: BranchingDocHead + Branch.
// ======================================================================

/// The head-safe, branch-facing surface handed to a `Branch::read` / `write`
/// closure: a pass-through newtype over a resolved head's `LoroDoc` exposing
/// ONLY the methods that read or write THIS head's materialized state or
/// transaction and leave the registry's view consistent.
///
/// It deliberately has NO `Deref`, NO `From`/inner accessor, and does NOT
/// forward the history / attachment / identity ops (`import*`, `export*`,
/// `checkout`, `attach`, `checkout_to_latest`, `detach`, `oplog_*`, `fork`,
/// `set_peer_id`, `diff`, config setters, ...). Those touch the shared op log or
/// the head's attachment and are incoherent on a registry head, so they are
/// ABSENT AT THE TYPE LEVEL: `head.import(..)` / `head.checkout(..)` /
/// `head.attach()` do not compile. `subscribe*` and `checkout` are head-safe but
/// NOT branch-correct (they must re-key/re-install on rebind) and so live on
/// `Branch`, not here.
///
/// > NOTE(claude-opus-4-8/branchingdocrepo-multiheaddoc): This is the RUST
/// > overlay over `loro_internal::LoroDoc`. The API-shape report recommends a
/// > sibling `loro`-crate `BranchingDocHead(loro::LoroDoc)` with the wasm class
/// > + `.d.ts` `Equals<Omit<LoroDoc, HistoryKeys>>` drift check; that public
/// > mirror is a Phase-4 artifact and can wrap or forward to this one. Placement
/// > here is what lets `Branch::read`/`write` be generic over
/// > `MultiHeadDoc<P>` and tested now against `Manual`/`SelfRooted`.
///
/// A head-safe method is present and works:
///
/// ```
/// use loro_internal::multi_head::{Manual, MultiHeadDoc};
/// let md = MultiHeadDoc::new(Manual::new());
/// md.bind(&"main".into(), md.root_head_id());
/// let text = md
///     .branch("main")
///     .write(|head| {
///         head.get_text("t").insert_unicode(0, "hi").unwrap();
///         head.get_text("t").to_string()
///     })
///     .unwrap();
/// assert_eq!(text, "hi");
/// ```
///
/// A history / attachment op is ABSENT at the type level (does not compile):
///
/// ```compile_fail
/// use loro_internal::multi_head::{Manual, MultiHeadDoc};
/// let md = MultiHeadDoc::new(Manual::new());
/// md.bind(&"main".into(), md.root_head_id());
/// md.branch("main")
///     .read(|head| {
///         // no method `import` / `checkout` / `attach` on BranchingDocHead
///         head.import(&[]).unwrap();
///     })
///     .unwrap();
/// ```
#[allow(missing_debug_implementations)]
pub struct BranchingDocHead(LoroDoc);

impl BranchingDocHead {
    pub(crate) fn from_head(doc: LoroDoc) -> Self {
        Self(doc)
    }

    // --- container access ---
    pub fn get_text<I: IntoContainerId>(&self, id: I) -> TextHandler {
        self.0.get_text(id)
    }
    pub fn get_map<I: IntoContainerId>(&self, id: I) -> MapHandler {
        self.0.get_map(id)
    }
    pub fn get_list<I: IntoContainerId>(&self, id: I) -> ListHandler {
        self.0.get_list(id)
    }
    pub fn get_movable_list<I: IntoContainerId>(&self, id: I) -> MovableListHandler {
        self.0.get_movable_list(id)
    }
    pub fn get_tree<I: IntoContainerId>(&self, id: I) -> TreeHandler {
        self.0.get_tree(id)
    }
    #[cfg(feature = "counter")]
    pub fn get_counter<I: IntoContainerId>(&self, id: I) -> CounterHandler {
        self.0.get_counter(id)
    }
    pub fn try_get_text<I: IntoContainerId>(&self, id: I) -> Option<TextHandler> {
        self.0.try_get_text(id)
    }
    pub fn try_get_map<I: IntoContainerId>(&self, id: I) -> Option<MapHandler> {
        self.0.try_get_map(id)
    }
    pub fn try_get_list<I: IntoContainerId>(&self, id: I) -> Option<ListHandler> {
        self.0.try_get_list(id)
    }
    pub fn try_get_movable_list<I: IntoContainerId>(&self, id: I) -> Option<MovableListHandler> {
        self.0.try_get_movable_list(id)
    }
    pub fn try_get_tree<I: IntoContainerId>(&self, id: I) -> Option<TreeHandler> {
        self.0.try_get_tree(id)
    }
    #[cfg(feature = "counter")]
    pub fn try_get_counter<I: IntoContainerId>(&self, id: I) -> Option<CounterHandler> {
        self.0.try_get_counter(id)
    }
    pub fn get_by_path(&self, path: &[Index]) -> Option<ValueOrHandler> {
        self.0.get_by_path(path)
    }
    pub fn get_by_str_path(&self, path: &str) -> Option<ValueOrHandler> {
        self.0.get_by_str_path(path)
    }
    pub fn has_container(&self, id: &ContainerID) -> bool {
        self.0.has_container(id)
    }

    // --- state reads ---
    pub fn get_value(&self) -> LoroValue {
        self.0.get_value()
    }
    pub fn get_deep_value(&self) -> LoroValue {
        self.0.get_deep_value()
    }
    pub fn get_deep_value_with_id(&self) -> LoroValue {
        self.0.get_deep_value_with_id()
    }
    pub fn state_frontiers(&self) -> Frontiers {
        self.0.state_frontiers()
    }
    pub fn state_vv(&self) -> VersionVector {
        self.0.state_vv()
    }
    pub fn cmp_with_frontiers(&self, other: &Frontiers) -> Ordering {
        self.0.cmp_with_frontiers(other)
    }
    pub fn get_path_to_container(&self, id: &ContainerID) -> Option<Vec<(ContainerID, Index)>> {
        self.0.get_path_to_container(id)
    }
    pub fn get_pending_txn_len(&self) -> usize {
        self.0.get_pending_txn_len()
    }
    pub fn peer_id(&self) -> PeerID {
        self.0.peer_id()
    }
    pub fn len_ops(&self) -> usize {
        self.0.len_ops()
    }
    pub fn len_changes(&self) -> usize {
        self.0.len_changes()
    }

    // --- local write ---
    /// Commit this head's pending transaction and start the next (the head-safe
    /// commit; `commit_with`'s internal form leaks a lock guard and is omitted).
    pub fn commit(&self) -> Option<CommitOptions> {
        self.0.commit_then_renew()
    }
    pub fn set_next_commit_message(&self, message: &str) {
        self.0.set_next_commit_message(message)
    }
    pub fn set_next_commit_origin(&self, origin: &str) {
        self.0.set_next_commit_origin(origin)
    }
    pub fn set_next_commit_timestamp(&self, timestamp: Timestamp) {
        self.0.set_next_commit_timestamp(timestamp)
    }
    pub fn set_next_commit_options(&self, options: CommitOptions) {
        self.0.set_next_commit_options(options)
    }
    pub fn clear_next_commit_options(&self) {
        self.0.clear_next_commit_options()
    }
    pub fn apply_diff(&self, diff: DiffBatch) -> LoroResult<()> {
        self.0.apply_diff(diff)
    }
    pub fn revert_to(&self, target: &Frontiers) -> LoroResult<()> {
        self.0.revert_to(target)
    }

    // --- per-head hooks ---
    pub fn subscribe_pre_commit(&self, callback: PreCommitCallback) -> Subscription {
        self.0.subscribe_pre_commit(callback)
    }
    pub fn free_diff_calculator(&self) {
        self.0.free_diff_calculator()
    }
}

/// Handle for a registry-owned branch subscription (from `Branch::subscribe` /
/// `subscribe_root`). Unlike a raw `Subscription` on one head, this one follows
/// the branch across rebinds (copy-on-divergence / merge). Dropping it removes
/// the subscription from the registry (and unsubscribes its current head
/// handle).
#[allow(missing_debug_implementations)]
pub struct BranchSubscription {
    remove: Option<Box<dyn FnOnce() + Send + Sync>>,
}

impl BranchSubscription {
    /// Stop the subscription immediately (same as dropping it).
    pub fn unsubscribe(self) {
        drop(self)
    }
}

impl Drop for BranchSubscription {
    fn drop(&mut self) {
        if let Some(f) = self.remove.take() {
            f();
        }
    }
}

/// A branch handle: a NAME plus its `MultiHeadDoc`. Every operation re-resolves
/// the branch's head at call time. This is the frozen consumer contract: no
/// public signature here names `LoroDoc` except `fork` (the escape hatch).
#[allow(missing_debug_implementations)]
pub struct Branch<'a, P: HeadPolicy> {
    doc: &'a MultiHeadDoc<P>,
    name: BranchId,
}

impl<P: HeadPolicy> Branch<'_, P> {
    /// Resolve the branch's head for read and run `f` on its `BranchingDocHead`.
    /// A mutation inside `f` on a Private head is committed on exit (see
    /// `MultiHeadDoc::read`); on a shared head it errors at the sink.
    pub fn read<R>(&self, f: impl FnOnce(&BranchingDocHead) -> R) -> LoroResult<R> {
        self.doc
            .read(&self.name, |d| f(&BranchingDocHead::from_head(d.clone())))
    }

    /// Resolve the branch's head for write (copy-on-divergence if shared), run
    /// `f`, then commit.
    pub fn write<R>(&self, f: impl FnOnce(&BranchingDocHead) -> R) -> LoroResult<R> {
        self.doc
            .write(&self.name, |d| f(&BranchingDocHead::from_head(d.clone())))
    }

    // NOTE(claude-opus-4-8/branchingdocrepo-multiheaddoc): `Branch::checkout`
    // (and `attach`/`checkout_to_latest`) are DEFERRED to an RFP and NOT exposed
    // here. They are a bundled "move the head, then reset from the oplog" surface
    // that does not map to branch scope: a branch has no single "latest", and
    // loro's `attach` snaps to the shared UNION, which corrupts an owned head.
    // The whole trio is prevented for now (the raw `LoroDoc` seam gates them);
    // the writable-after-checkout contract is the RFP's to define, so there is
    // nothing to freeze on `Branch`. `fork` / `fork_at` (eject) stay as the
    // history-adjacent affordance we keep.

    /// Branch-scoped container subscription. A registry op: installed on the
    /// branch's current head.
    ///
    /// > NOTE(claude-opus-4-8/branchingdocrepo-multiheaddoc): re-installing the
    /// > subscription on the new head across a copy-on-divergence / merge rebind
    /// > (registry-owned `subs`, proposal L183/L411) is the Phase-3 BODY; this
    /// > unit freezes the signature and installs on the current head.
    pub fn subscribe(&self, cid: &ContainerID, cb: Subscriber) -> LoroResult<BranchSubscription> {
        self.doc.subscribe_branch(&self.name, Some(cid.clone()), cb)
    }

    /// Branch-scoped root subscription (see `subscribe`).
    pub fn subscribe_root(&self, cb: Subscriber) -> LoroResult<BranchSubscription> {
        self.doc.subscribe_branch(&self.name, None, cb)
    }

    /// The gated escape hatch: EJECT a standalone `LoroDoc` (independent op log,
    /// full surface) that aliases no registry head. This is the only `Branch`
    /// signature that names `LoroDoc`.
    ///
    /// Ejects via `fork_at(&head.state_frontiers())`, NOT `LoroDoc::fork()`:
    /// `fork()` assumes `!is_detached()` implies `state == oplog.frontiers()`,
    /// but a registry head is behind the shared union while attached, so `fork()`
    /// would label the ejected snapshot with the union frontier while its state
    /// is only this head's -- an incoherent doc that panics on first `checkout`
    /// (`richtext_state.rs`). `fork_at` snapshots AT the head's own frontier, so
    /// the ejected doc's state matches its frontier label and is checkout-able.
    pub fn fork(&self) -> LoroResult<LoroDoc> {
        let (_, head) = self.doc.resolve(&self.name, Intent::Read)?;
        head.fork_at(&head.state_frontiers())
    }
}

// ======================================================================
// The index variant: MultiHeadDoc<SelfRooted>.
// ======================================================================

/// Root-container name prefix for a branch's lineage list (`lineage:<b>`).
const LINEAGE_PREFIX: &str = "lineage:";

fn lineage_name(b: &BranchId) -> String {
    format!("{LINEAGE_PREFIX}{b}")
}

/// If `idx` names a `lineage:<b>` root container, return `b`. Pure op-log
/// discovery: the branch is encoded in the (name-addressable) container id, so
/// `after_import` never needs a materialized state to identify it.
fn lineage_branch_of(ol: &OpLog, idx: crate::container::idx::ContainerIdx) -> Option<BranchId> {
    match ol.arena.idx_to_id(idx)? {
        ContainerID::Root { name, .. } => name
            .as_str()
            .strip_prefix(LINEAGE_PREFIX)
            .map(InternalString::from),
        _ => None,
    }
}

/// The index's resolution policy: its OWN per-branch lineage IS the root of
/// truth for where each branch is, so it consults no other index -- this breaks
/// the `BranchingDoc`-depends-on-index circularity. Eager copy at branch
/// creation means every index head is born `refs == 1` (never shared), so the
/// sink guard never fires on an index head.
///
/// `lineage` maps a branch to the peers of its index heads. It is rebuilt from
/// the index's own op log: each index head writes under exactly one branch with
/// a unique peer, so "the index ops of branch `b`" is exactly "the ops of the
/// peers in `lineage:<b>`", and `b`'s index frontier is the join of those peers'
/// latest ids.
#[allow(missing_debug_implementations)]
pub struct SelfRooted {
    // Created lazily in the doc's lock group (`LockKind::Lineage`, the leaf
    // acquired only after `OpLog`) on first use, since the group exists only
    // once `MultiHeadDoc::new` has run.
    lineage: OnceLock<LoroMutex<FxHashMap<BranchId, SmallVec<[PeerID; 2]>>>>,
}

impl Default for SelfRooted {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfRooted {
    pub fn new() -> Self {
        SelfRooted {
            lineage: OnceLock::new(),
        }
    }

    fn lineage(
        &self,
        this: &MultiHeadDoc<SelfRooted>,
    ) -> &LoroMutex<FxHashMap<BranchId, SmallVec<[PeerID; 2]>>> {
        self.lineage.get_or_init(|| {
            this.lock_group
                .new_lock(FxHashMap::default(), LockKind::Lineage)
        })
    }
}

impl HeadPolicy for SelfRooted {
    const COPY: CopyMode = CopyMode::Eager;

    /// Branch `b`'s index frontier = the join of the latest ids of the peers in
    /// `lineage:<b>`, shrunk against the op-log DAG. The causal past does the
    /// rest: a peer's ops depend on the tip its head forked from, so this brings
    /// the inherited parent record and `b`'s own writes, never a sibling's later
    /// writes (which no `b` peer depends on).
    fn target(&self, this: &MultiHeadDoc<Self>, b: &BranchId) -> LoroResult<Frontiers> {
        let ol = this.oplog.lock(); // OpLog before Lineage
        let peers: SmallVec<[PeerID; 2]> = self
            .lineage(this)
            .lock()
            .get(b)
            .cloned()
            .unwrap_or_default();
        let ids: Vec<ID> = peers
            .iter()
            .filter_map(|p| ol.vv().get_last(*p).map(|c| ID::new(*p, c)))
            .collect();
        shrink_frontiers(&Frontiers::from(ids), &ol.dag).map_err(LoroError::FrontiersNotFound)
    }

    /// The index tip IS the record; nothing extra to publish.
    fn after_commit(&self, _this: &MultiHeadDoc<Self>, _b: &BranchId, _last: ID) {}

    /// Walk the just-imported spans: an op in a `lineage:<b>` container names a
    /// (possibly remote) peer of `b`; any op by a known lineage peer marks its
    /// branch touched. Returns the branches whose index head should be rebound.
    fn after_import(&self, this: &MultiHeadDoc<Self>, st: &ImportStatus) -> Vec<BranchId> {
        // Collect under the OpLog lock, then fold into the lineage map (leaf
        // lock, acquired alone) -- never nesting the two here.
        let mut lineage_ops: Vec<(BranchId, PeerID)> = Vec::new();
        let mut change_peers: Vec<PeerID> = Vec::new();
        {
            let ol = this.oplog.lock();
            for (peer, (start, end)) in st.success.iter() {
                for ch in ol.iter_changes(IdSpan::new(*peer, *start, *end)) {
                    let cp = ch.peer();
                    change_peers.push(cp);
                    for op in ch.ops().iter() {
                        if let Some(b) = lineage_branch_of(&ol, op.container) {
                            lineage_ops.push((b, cp));
                        }
                    }
                }
            }
        }
        let mut touched: FxHashSet<BranchId> = FxHashSet::default();
        let mut lineage = self.lineage(this).lock();
        for (b, p) in lineage_ops {
            let entry = lineage.entry(b.clone()).or_default();
            if !entry.contains(&p) {
                entry.push(p);
            }
            touched.insert(b);
        }
        for cp in change_peers {
            for (b, peers) in lineage.iter() {
                if peers.contains(&cp) {
                    touched.insert(b.clone());
                }
            }
        }
        drop(lineage);
        touched.into_iter().collect()
    }
}

/// The repo's index doc. Its heads hold only frontiers: a root map
/// `docs: LoroMap<DocId, LoroMap<"heads", LoroMap<PeerID, Counter>>>` plus one
/// root list per branch, `lineage:<b>`. No weft filesystem metadata (that is a
/// TS-side content doc). The only types that appear are `ID`/`Frontiers`/
/// `DocId`/`BranchId`.
pub type IndexDoc = MultiHeadDoc<SelfRooted>;

impl MultiHeadDoc<SelfRooted> {
    /// Establish the first (genesis) branch, bound to the pinned root head, and
    /// seed its lineage with the root's peer. Must be called once before other
    /// branches are created.
    pub fn init_genesis(&self, genesis: &BranchId) -> LoroResult<()> {
        let root = self.head_doc(ROOT_HEAD_ID).expect("root head exists");
        let peer = root.peer_id();
        self.with_reg(|this, reg| this.rebind(reg, genesis, ROOT_HEAD_ID));
        // The genesis lineage op, authored by the root's peer, written directly
        // on the root head; then recorded, so target(genesis) == root's tip.
        root.get_list(lineage_name(genesis).as_str())
            .push(peer as i64)?;
        root.commit_then_renew();
        self.policy()
            .lineage(self)
            .lock()
            .entry(genesis.clone())
            .or_default()
            .push(peer);
        Ok(())
    }

    /// Create `new` from `from` by EAGER copy: fork `from`'s index head (a state
    /// of a few map entries), then write the copy's first op -- appending the
    /// copy's fresh peer to `lineage:<new>` -- directly on the copy so its tip is
    /// `[peer]` and `target(new)` is consistent. Every index head is thus born
    /// `refs == 1`.
    pub fn create_index_branch(&self, new: &BranchId, from: &BranchId) -> LoroResult<()> {
        let (from_head, _) = self.resolve(from, Intent::Read)?;
        let copy_doc = self.with_reg(|this, reg| {
            let c = this.copy_head(reg, from_head);
            this.rebind(reg, new, c);
            reg.heads[&c].doc.clone()
        });
        let peer = copy_doc.peer_id();
        copy_doc
            .get_list(lineage_name(new).as_str())
            .push(peer as i64)?;
        copy_doc.commit_then_renew();
        self.policy()
            .lineage(self)
            .lock()
            .entry(new.clone())
            .or_default()
            .push(peer);
        Ok(())
    }

    /// Record `docs[doc].heads[peer] = counter` on `b`'s index head: the
    /// per-writer head record the frontier reduction reads back. One local op.
    pub fn record_head(
        &self,
        b: &BranchId,
        doc: &DocId,
        peer: PeerID,
        counter: Counter,
    ) -> LoroResult<()> {
        self.write(b, |d| {
            // Mergeable (map-key) child containers so two sessions writing the
            // same doc/heads path converge to one container across import.
            let docs = d.get_map("docs");
            let per_doc = docs.ensure_mergeable_map(doc.as_str())?;
            let heads = per_doc.ensure_mergeable_map("heads")?;
            heads.insert(&peer.to_string(), counter as i64)
        })?
    }

    /// Record every id of frontier `f` into `docs[doc].heads` on `b`'s index
    /// head (advance / merge / ingest: the durable per-branch content frontier).
    pub fn record_frontier(&self, b: &BranchId, doc: &DocId, f: &Frontiers) -> LoroResult<()> {
        self.write(b, |d| {
            let docs = d.get_map("docs");
            let per_doc = docs.ensure_mergeable_map(doc.as_str())?;
            let heads = per_doc.ensure_mergeable_map("heads")?;
            for id in f.iter() {
                heads.insert(&id.peer.to_string(), id.counter as i64)?;
            }
            Ok(())
        })?
    }

    /// The raw `(peer, counter)` ids recorded in `docs[doc].heads` on `b`'s index
    /// head (before any reduction against a content doc's history). Empty if the
    /// branch or doc is unknown. `Delegated::target` reduces these against the
    /// content doc's DAG.
    pub fn recorded_ids(&self, b: &BranchId, doc: &DocId) -> LoroResult<Vec<ID>> {
        self.read(b, |d| {
            let mut ids = Vec::new();
            let heads = d
                .get_deep_value()
                .as_map()
                .and_then(|root| root.get("docs").cloned())
                .and_then(|v| v.into_map().ok())
                .and_then(|docs| docs.get(doc.as_str()).cloned())
                .and_then(|v| v.into_map().ok())
                .and_then(|per| per.get("heads").cloned())
                .and_then(|v| v.into_map().ok());
            if let Some(heads) = heads {
                for (peer_str, counter) in heads.iter() {
                    if let (Ok(peer), Some(c)) = (peer_str.parse::<PeerID>(), counter.as_i64()) {
                        ids.push(ID::new(peer, *c as Counter));
                    }
                }
            }
            ids
        })
    }

    /// The branches this session knows, from the lineage map.
    pub fn branches(&self) -> Vec<BranchId> {
        self.policy().lineage(self).lock().keys().cloned().collect()
    }

    /// Remove a branch from the index: unbind its index head (retiring it) and
    /// drop its lineage entry, so `branches()` no longer lists it and no
    /// `bound`/`by_tip` entry dangles. (The durable cross-peer "discard" is the
    /// wrapper's lifecycle log; this is the local registry cleanup.)
    pub fn delete_index_branch(&self, name: &BranchId) {
        self.unbind(name);
        self.policy().lineage(self).lock().remove(name);
    }

    /// The registry HeadId of branch `b`'s (self-rooted) index head, resolving
    /// it from lineage if not yet bound. (`Head` itself is registry-internal, so
    /// this hands back the id rather than the proposal's `Arc<Head>`.)
    pub fn head_of(&self, b: &BranchId) -> LoroResult<HeadId> {
        Ok(self.resolve(b, Intent::Read)?.0)
    }
}

// ======================================================================
// The content variant: BranchingDoc = MultiHeadDoc<Delegated>, and the repo.
// ======================================================================

/// The outcome of a `merge`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeOutcome {
    /// `into` already contains `from` (equal or ahead); nothing moved.
    AlreadyContained,
    /// `from` is strictly ahead of `into`; a fast-forward (no divergent diff).
    FastForward,
    /// `into` and `from` diverged; the join was computed and applied.
    Merged,
}

/// A content doc's resolution policy: DELEGATE to the repo's index for "where is
/// branch `b` for this doc", copying lazily on divergence. The index is the
/// source of truth; a content head materializes at the recorded frontier.
///
/// NOTE(claude-opus-4-8/branchingdocrepo-multiheaddoc): the proposal's
/// `frontier_cache` (a per-branch mirror of the index frontier refreshed by the
/// index's history subscription) is a HOT-PATH OPTIMIZATION and is deferred:
/// `target` reads the index directly each call (the source of truth), which is
/// correct, just not cached. The cache is Phase-3 follow-up scope.
#[allow(missing_debug_implementations)]
pub struct Delegated {
    id: DocId,
    index: Arc<IndexDoc>,
}

impl Delegated {
    pub fn new(id: DocId, index: Arc<IndexDoc>) -> Self {
        Delegated { id, index }
    }
}

impl HeadPolicy for Delegated {
    const COPY: CopyMode = CopyMode::OnDivergence;

    /// Where branch `b` is for this doc: the index's recorded ids reduced against
    /// THIS doc's history -- drop ids the content history does not yet hold
    /// ("skipped until held"), then shrink. Reads the index first (its locks
    /// released), then the content DAG; the two docs' locks never nest.
    fn target(&self, this: &MultiHeadDoc<Self>, b: &BranchId) -> LoroResult<Frontiers> {
        let raw = self.index.recorded_ids(b, &self.id)?;
        let ol = this.oplog.lock();
        let held: Vec<ID> = raw
            .into_iter()
            .filter(|id| ol.vv().get_last(id.peer).is_some_and(|c| c >= id.counter))
            .collect();
        shrink_frontiers(&Frontiers::from(held), &ol.dag).map_err(LoroError::FrontiersNotFound)
    }

    /// Publish this commit's new tip into the index: `docs[doc].heads[peer] = c`.
    fn after_commit(&self, _this: &MultiHeadDoc<Self>, b: &BranchId, last: ID) {
        let _ = self.index.record_head(b, &self.id, last.peer, last.counter);
    }

    /// After a content import, re-resolve every branch the index knows: a branch
    /// whose recorded ids for this doc just became held advances to them (the
    /// ingest); ids still unheld are dropped by `target` and picked up next time.
    fn after_import(&self, _this: &MultiHeadDoc<Self>, _status: &ImportStatus) -> Vec<BranchId> {
        self.index.branches()
    }
}

/// One content document: a `MultiHeadDoc` with delegated (index-backed) branch
/// resolution and copy-on-divergence.
pub type BranchingDoc = MultiHeadDoc<Delegated>;

impl MultiHeadDoc<Delegated> {
    fn doc_id(&self) -> &DocId {
        &self.inner.policy.id
    }
    fn index(&self) -> &IndexDoc {
        &self.inner.policy.index
    }

    /// This doc's content frontier for branch `b` (the reduced index record).
    pub fn frontier_of(&self, b: &BranchId) -> LoroResult<Frontiers> {
        self.inner.policy.target(self, b)
    }

    /// Whether branch `target` contains branch `source` (both of this doc).
    pub fn contains(&self, target: &BranchId, source: &BranchId) -> LoroResult<bool> {
        let ft = self.frontier_of(target)?;
        let fs = self.frontier_of(source)?;
        let ol = self.oplog.lock();
        Ok(matches!(
            ol.dag.cmp_frontiers(&fs, &ft).map_err(LoroError::from)?,
            Some(Ordering::Less) | Some(Ordering::Equal)
        ))
    }

    /// Move branch `b` to frontier `to`: record it in the index (durable state
    /// leads the cache), then re-resolve to advance/rebind this doc's head.
    pub fn advance(&self, b: &BranchId, to: &Frontiers) -> LoroResult<()> {
        self.index().record_frontier(b, self.doc_id(), to)?;
        let _ = self.resolve(b, Intent::Read)?;
        Ok(())
    }

    /// Create branch `name` as a WRITABLE fork of THIS doc at a chosen historical
    /// frontier `at` (the `forkAt`-shaped affordance). The branch is registered
    /// in the index (off genesis if new) and `at` is recorded as its content
    /// frontier for this doc, overriding the inherited record; it is cold until
    /// first access, when it materializes to `at` as a live writable head.
    ///
    /// NOTE(claude-opus-4-8/branchingdocrepo-multiheaddoc): this is the RFP's
    /// use-case 3 ("branch off a past version to start work" = a writable NEW
    /// branch), NOT a read-only history VIEW (RFP use-cases 1/2). Since the
    /// materialized fork is a live writable head (iter-1 unify), no read-only
    /// detached-owned head is produced here -- so the `attach`/`checkout_to_latest`
    /// no-op gates (N2) stay DEFENSE-IN-DEPTH, unexercised, until the separate
    /// read-only-history-view affordance lands.
    pub fn create_branch_at(&self, name: &BranchId, at: &Frontiers) -> LoroResult<()> {
        if !self.index().branches().contains(name) {
            self.index()
                .create_index_branch(name, &GENESIS_BRANCH.into())?;
        }
        self.index().record_frontier(name, self.doc_id(), at)?;
        Ok(())
    }

    /// Merge `from` into `into` as frontier advancement -- no new ops are created
    /// and NO op is dropped: the applied frontier is the join (shrink of the
    /// union of both branches' ids), so `into` advances to include every op of
    /// `from`.
    pub fn merge(&self, into: &BranchId, from: &BranchId) -> LoroResult<MergeOutcome> {
        let fi = self.frontier_of(into)?;
        let ff = self.frontier_of(from)?;
        let (outcome, join) = {
            let ol = self.oplog.lock();
            let outcome = match ol.dag.cmp_frontiers(&fi, &ff).map_err(LoroError::from)? {
                Some(Ordering::Equal) | Some(Ordering::Greater) => {
                    return Ok(MergeOutcome::AlreadyContained)
                }
                Some(Ordering::Less) => MergeOutcome::FastForward,
                None => MergeOutcome::Merged,
            };
            // The join: shrink(union of both frontiers' ids). Every id of `from`
            // is in the union, so nothing is dropped.
            let mut u = fi.clone();
            for id in ff.iter() {
                u.push(id);
            }
            (
                outcome,
                shrink_frontiers(&u, &ol.dag).map_err(LoroError::FrontiersNotFound)?,
            )
        };
        self.advance(into, &join)?;
        Ok(outcome)
    }
}

/// The repo: one loro index doc plus its content docs. Branch existence and
/// "where each branch is for each doc" live in the index; content docs delegate.
#[allow(missing_debug_implementations)]
pub struct BranchingDocRepo {
    index: Arc<IndexDoc>,
    docs: std::sync::Mutex<FxHashMap<DocId, Arc<BranchingDoc>>>,
}

/// The genesis branch every repo opens with.
pub const GENESIS_BRANCH: &str = "main";

impl BranchingDocRepo {
    /// Open a fresh repo with the genesis branch.
    pub fn open() -> LoroResult<Self> {
        let index = Arc::new(IndexDoc::new(SelfRooted::new()));
        index.init_genesis(&GENESIS_BRANCH.into())?;
        Ok(BranchingDocRepo {
            index,
            docs: std::sync::Mutex::new(FxHashMap::default()),
        })
    }

    /// The index doc (pure ids / frontiers).
    pub fn index(&self) -> &IndexDoc {
        &self.index
    }

    /// The branches the repo knows (from the index lineage).
    pub fn branches(&self) -> Vec<BranchId> {
        self.index.branches()
    }

    /// Open (or get) a content doc. Idempotent per id.
    pub fn open_doc(&self, id: DocId) -> Arc<BranchingDoc> {
        let mut docs = self.docs.lock().unwrap();
        if let Some(d) = docs.get(&id) {
            return d.clone();
        }
        let bd = Arc::new(MultiHeadDoc::new(Delegated::new(
            id.clone(),
            self.index.clone(),
        )));
        docs.insert(id, bd.clone());
        bd
    }

    /// Drop the in-memory content doc (its recorded frontiers stay in the index;
    /// reopening re-materializes lazily).
    pub fn close_doc(&self, id: &DocId) {
        self.docs.lock().unwrap().remove(id);
    }

    /// Repo-wide branch creation: eager-copy the index head for `new` from
    /// `from`; content docs bind `new` lazily on first access (free creation).
    pub fn create_branch(&self, new: &BranchId, from: &BranchId) -> LoroResult<()> {
        self.index.create_index_branch(new, from)
    }

    /// Repo-wide branch deletion: unbind the branch from every open content doc
    /// AND the index, and drop its index lineage. `branches()` then excludes it
    /// with no dangling `bound`/`by_tip` entry anywhere; the pinned root is never
    /// removed. Its committed ops stay in each doc's history (unreferenced).
    pub fn delete_branch(&self, name: &BranchId) -> LoroResult<()> {
        for doc in self.docs.lock().unwrap().values() {
            doc.unbind(name);
        }
        self.index.delete_index_branch(name);
        Ok(())
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

    /// The recorded target for a branch (test: observe `after_commit`).
    pub fn target_of(&self, b: &BranchId) -> Option<Frontiers> {
        self.targets.lock().unwrap().get(b).cloned()
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
    use crate::handler::HandlerTrait;
    use crate::lock::{LockKind, LoroLockGroup};
    use loro_common::LoroError;
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
        // A leaked raw checkout is now refused earlier, by the owned-head
        // checkout gate (before it can reach the apply_diff sink), so the error
        // is `OwnedHeadOp` rather than `HeadShared`. Either way: refused.
        let r_checkout = leaked.checkout(&Frontiers::default());
        assert!(
            matches!(r_checkout, Err(LoroError::OwnedHeadOp("checkout"))),
            "leaked checkout after flip refused, got {r_checkout:?}"
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

    // ------------------------------------------------------------------
    // Injected on-commit hook (migration off the synchronous write() path)
    // ------------------------------------------------------------------

    #[test]
    fn injected_on_commit_hook_rekeys_and_after_commits() {
        // Commit DIRECTLY on a head's own handle, bypassing MultiHeadDoc::write
        // entirely. by_tip re-keying and Manual's after_commit can now come ONLY
        // from the injected txn on-commit hook (the synchronous write()-driven
        // path is gone), so their effects prove the hook fired.
        let md = MultiHeadDoc::new(Manual::new());
        let main = b("main");
        md.bind(&main, 0); // refs 1, private, tip empty
        assert!(md.by_tip_maps(&Frontiers::default(), 0));

        let doc = md.head_doc(0).unwrap();
        doc.get_text("t").insert_unicode(0, "A").unwrap();
        doc.commit_then_renew(); // fires the injected on_commit hook

        let t = md.head_tip(0).unwrap();
        assert!(!t.is_empty(), "tip advanced");
        assert!(md.by_tip_maps(&t, 0), "hook re-keyed by_tip to the new tip");
        assert!(
            !md.by_tip_maps(&Frontiers::default(), 0),
            "hook vacated the old empty tip"
        );
        assert_eq!(
            md.policy().target_of(&main),
            Some(t),
            "hook ran after_commit (Manual recorded the new target)"
        );
    }

    // ------------------------------------------------------------------
    // History-only import: no head materialization, co-owner safe, visible
    // to a head that advances; sink guard backstops a shared head.
    // ------------------------------------------------------------------

    fn external_updates(text: &str) -> Vec<u8> {
        let ext = LoroDoc::new();
        ext.start_auto_commit();
        ext.get_text("t").insert_unicode(0, text).unwrap();
        ext.commit_then_renew();
        ext.export(crate::encoding::ExportMode::all_updates())
            .unwrap()
    }

    fn external_updates_frontier(text: &str) -> (Vec<u8>, Frontiers) {
        let ext = LoroDoc::new();
        ext.start_auto_commit();
        ext.get_text("t").insert_unicode(0, text).unwrap();
        ext.commit_then_renew();
        let f = ext.state_frontiers();
        let bytes = ext
            .export(crate::encoding::ExportMode::all_updates())
            .unwrap();
        (bytes, f)
    }

    #[test]
    fn import_is_history_only_and_coowner_safe() {
        let md = MultiHeadDoc::new(Manual::new());
        let (main, review) = (b("main"), b("review"));
        md.bind(&main, 0);
        md.bind(&review, 0); // refs 2, head 0 shared
        assert_eq!(md.is_head_shared(&main), Some(true));

        let updates = external_updates("ABC");
        md.import(&updates).unwrap();

        // History-only: the shared head's materialized state is UNTOUCHED, so no
        // co-owner sees the imported ops. (The failure picture: an import that
        // silently materialized into head 0 would corrupt both main and review.)
        assert_eq!(
            tlen(&md, &main),
            0,
            "shared co-owner main untouched by import"
        );
        assert_eq!(
            tlen(&md, &review),
            0,
            "shared co-owner review untouched by import"
        );

        // The ops ARE in the shared history: a fresh branch that advances to
        // include them materializes them.
        let (updates2, f2) = external_updates_frontier("XYZ");
        md.import(&updates2).unwrap();
        let reader = b("reader");
        md.policy().set_target(&reader, f2);
        let seen = md.read(&reader, |d| d.get_text("t").to_string()).unwrap();
        assert_eq!(
            seen, "XYZ",
            "imported ops visible to a head advanced to them"
        );
        // Co-owners of the shared head still see nothing.
        assert_eq!(tlen(&md, &main), 0);
        assert_eq!(tlen(&md, &review), 0);
    }

    #[test]
    fn import_path_sink_guard_backstops_shared_head() {
        // The history-only import never materializes into a head. As a backstop,
        // even a DIRECT attempt to apply imported ops to the shared head (via
        // checkout -> apply_diff) is refused by the sink guard rather than
        // silently corrupting co-owners. This is the guard the history-only rule
        // makes it unnecessary to rely on, shown load-bearing.
        let md = MultiHeadDoc::new(Manual::new());
        let (main, review) = (b("main"), b("review"));
        md.bind(&main, 0);
        md.bind(&review, 0); // shared

        let (updates, f) = external_updates_frontier("ABC");
        md.import(&updates).unwrap();
        assert_eq!(
            tlen(&md, &main),
            0,
            "history-only import left the shared head empty"
        );

        // A direct checkout of the shared head to the imported frontier would be
        // a materialization; the owned-head checkout gate refuses it (before the
        // apply_diff sink guard would).
        let shared = md.head_doc(0).unwrap();
        let r = shared.checkout(&f);
        assert!(
            matches!(r, Err(LoroError::OwnedHeadOp("checkout"))),
            "checkout on a shared owned head refused, got {r:?}"
        );
        assert_eq!(
            tlen(&md, &main),
            0,
            "co-owner still uncorrupted after the refused attempt"
        );
    }

    #[test]
    fn owned_head_refuses_direct_import_and_set_peer_id() {
        // The owner gates: a registry head is mutated only through the registry.
        let md = MultiHeadDoc::new(Manual::new());
        md.bind(&b("main"), 0);
        let head = md.head_doc(0).unwrap();
        let updates = external_updates("ABC");
        assert!(
            matches!(head.import(&updates), Err(LoroError::OwnedHeadOp("import"))),
            "direct import on an owned head refused"
        );
        assert!(
            matches!(
                head.set_peer_id(12345),
                Err(LoroError::OwnedHeadOp("set_peer_id"))
            ),
            "set_peer_id on an owned head refused"
        );
    }

    // ------------------------------------------------------------------
    // Completed base methods: aggregated subscriptions and snapshot export.
    // ------------------------------------------------------------------

    #[test]
    fn subscribe_history_fires_on_commit_and_import() {
        let md = MultiHeadDoc::new(Manual::new());
        let main = b("main");
        md.bind(&main, 0);
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c2 = count.clone();
        let _sub = md.subscribe_history(Box::new(move |_bytes| {
            c2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            true
        }));
        md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
            .unwrap();
        let after_commit = count.load(std::sync::atomic::Ordering::SeqCst);
        assert!(after_commit >= 1, "history event fired on a head commit");
        md.import(&external_updates("Z")).unwrap();
        assert!(
            count.load(std::sync::atomic::Ordering::SeqCst) > after_commit,
            "history event fired on import"
        );
    }

    #[test]
    fn subscribe_first_commit_from_peer_aggregates_over_heads() {
        let md = MultiHeadDoc::new(Manual::new());
        let (main, review) = (b("main"), b("review"));
        let peers = Arc::new(Mutex::new(Vec::<u64>::new()));
        let p2 = peers.clone();
        let _sub = md.subscribe_first_commit_from_peer(Box::new(move |payload| {
            p2.lock().unwrap().push(payload.peer);
            true
        }));
        md.bind(&main, 0);
        md.bind(&review, 0); // shared
                             // main diverges -> a copy with a fresh peer -> a first-commit from it.
        md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
            .unwrap();
        // review diverges -> another fresh peer.
        md.write(&review, |d| d.get_text("t").insert_unicode(0, "B").unwrap())
            .unwrap();
        let seen = peers.lock().unwrap().clone();
        assert!(
            seen.len() >= 2,
            "one first-commit per write slot (fresh peer per copy), got {seen:?}"
        );
        assert_ne!(seen[0], seen[1], "distinct peers per head copy");
    }

    #[test]
    fn export_snapshot_roundtrips_a_head_state() {
        let md = MultiHeadDoc::new(Manual::new());
        let main = b("main");
        md.bind(&main, 0);
        md.write(&main, |d| {
            d.get_text("t").insert_unicode(0, "hello").unwrap()
        })
        .unwrap();
        // Snapshot carries the root head's state; a fresh doc restores it.
        let snap = md.export(crate::encoding::ExportMode::Snapshot).unwrap();
        let restored = LoroDoc::new();
        restored.import(&snap).unwrap();
        assert_eq!(restored.get_text("t").to_string(), "hello");
        // Updates export is head-independent (shared op log).
        let updates = md
            .export(crate::encoding::ExportMode::all_updates())
            .unwrap();
        let restored2 = LoroDoc::new();
        restored2.import(&updates).unwrap();
        assert_eq!(restored2.get_text("t").to_string(), "hello");
    }

    // ------------------------------------------------------------------
    // refs==0 retirement: non-root heads are dropped, the root is pinned.
    // ------------------------------------------------------------------

    #[test]
    fn dead_tip_rematerializes_via_replay_after_head_dropped() {
        let md = MultiHeadDoc::new(Manual::new());
        let (main, feature) = (b("main"), b("feature"));
        md.bind(&main, 0);
        md.bind(&feature, 0); // root shared, refs 2

        // feature diverges onto its OWN sole-owner (non-root) head H.
        md.write(&feature, |d| {
            d.get_text("t").insert_unicode(0, "F").unwrap()
        })
        .unwrap();
        let h = md.bound_head(&feature).unwrap();
        assert_ne!(h, md.root_head_id(), "feature is on a non-root head");
        assert_eq!(md.refs_of(&feature), Some(1), "H is feature's sole owner");
        let h_tip = md.head_tip(h).unwrap();
        let pre_value = md.read(&feature, |d| d.get_text("t").to_string()).unwrap();
        assert_eq!(pre_value, "F");

        // Rebind feature OFF H (target back to the root's empty tip): H reaches
        // refs 0 and is RETIRED (dropped), root is not.
        md.policy().set_target(&feature, Frontiers::default());
        md.resolve(&feature, Intent::Read).unwrap();
        assert!(
            md.head_doc(h).is_none(),
            "sole-owner head retired at refs 0"
        );
        assert!(
            md.head_doc(md.root_head_id()).is_some(),
            "root head is never retired"
        );

        // A later access to H's now-dead tip re-materializes correctly via
        // replay from the pinned root (value equality with the pre-drop state).
        let reader = b("reader");
        md.policy().set_target(&reader, h_tip.clone());
        let seen = md.read(&reader, |d| d.get_text("t").to_string()).unwrap();
        assert_eq!(
            seen, pre_value,
            "dead tip re-materialized to the same value via replay"
        );
        assert_ne!(
            md.bound_head(&reader).unwrap(),
            h,
            "re-materialization built a fresh head, not the dropped one"
        );
    }

    #[test]
    fn root_head_is_pinned_never_retired() {
        let md = MultiHeadDoc::new(Manual::new());
        let (main, x) = (b("main"), b("x"));
        md.bind(&main, 0);
        md.bind(&x, 0); // root shared, refs 2

        // x diverges onto its own head Hx; root drops to refs 1 (main only).
        md.write(&x, |d| d.get_text("t").insert_unicode(0, "X").unwrap())
            .unwrap();
        let hx = md.bound_head(&x).unwrap();
        assert_ne!(hx, 0);

        // Move main onto Hx too, driving the root head to refs 0.
        md.policy().set_target(&main, md.head_tip(hx).unwrap());
        md.resolve(&main, Intent::Read).unwrap();
        assert_ne!(md.bound_head(&main), Some(0), "main left the root head");

        // The root reached refs 0 but is PINNED: still present and still the
        // materialize/import/export anchor.
        assert!(
            md.head_doc(0).is_some(),
            "root head is pinned and survives refs == 0"
        );
        // It still anchors resolution: a branch targeting the root's (empty) tip
        // shares it rather than hitting a missing head.
        let cold = b("cold");
        md.policy().set_target(&cold, Frontiers::default());
        let seen = md.read(&cold, |d| d.get_text("t").to_string()).unwrap();
        assert_eq!(seen, "", "pinned root still resolvable at its empty tip");
    }

    // ------------------------------------------------------------------
    // Phase 2: IndexDoc = MultiHeadDoc<SelfRooted>
    // ------------------------------------------------------------------

    fn d(s: &str) -> DocId {
        s.into()
    }

    fn doc_has_key(md: &IndexDoc, branch: &BranchId, key: &str) -> bool {
        md.read(branch, |doc| doc.get_map("docs").get_(key).is_some())
            .unwrap()
    }

    #[test]
    fn index_eager_copy_born_refs_one_never_shared() {
        let idx = IndexDoc::new(SelfRooted::new());
        idx.init_genesis(&b("main")).unwrap();
        idx.create_index_branch(&b("feat"), &b("main")).unwrap();
        // Eager: feat is on its OWN head (a copy of main's), refs 1, private.
        assert_ne!(idx.bound_head(&b("feat")), idx.bound_head(&b("main")));
        assert_eq!(idx.refs_of(&b("feat")), Some(1));
        assert_eq!(idx.is_head_shared(&b("feat")), Some(false));
        assert!(idx.branches().contains(&b("main")));
        assert!(idx.branches().contains(&b("feat")));
    }

    #[test]
    fn index_remote_lineage_import_rebinds_branch() {
        // Session A creates `feat` and records content on it.
        let a = IndexDoc::new(SelfRooted::new());
        a.init_genesis(&b("main")).unwrap();
        a.create_index_branch(&b("feat"), &b("main")).unwrap();
        a.record_head(&b("feat"), &d("D"), 111, 7).unwrap();
        let updates = a.export(ExportMode::all_updates()).unwrap();

        // Session B knows only `main`.
        let bb = IndexDoc::new(SelfRooted::new());
        bb.init_genesis(&b("main")).unwrap();
        assert!(
            !bb.branches().contains(&b("feat")),
            "feat unknown before import"
        );

        // Importing A's history discovers `feat` via the lineage scan and
        // rebinds it (after_import -> resolve).
        bb.import(&updates).unwrap();
        assert!(
            bb.branches().contains(&b("feat")),
            "remote branch discovered"
        );
        assert!(
            doc_has_key(&bb, &b("feat"), "D"),
            "feat's index head materialized A's record"
        );
        assert!(
            !doc_has_key(&bb, &b("main"), "D"),
            "main did not absorb feat's record"
        );
    }

    #[test]
    fn index_target_is_join_of_lineage_peers() {
        // Session A creates `shared` and records doc "A" on it.
        let a = IndexDoc::new(SelfRooted::new());
        a.init_genesis(&b("main")).unwrap();
        a.create_index_branch(&b("shared"), &b("main")).unwrap();
        a.record_head(&b("shared"), &d("A"), 1, 1).unwrap();
        let a_updates = a.export(ExportMode::all_updates()).unwrap();

        // Session B independently creates the SAME branch `shared`, records "B".
        let bb = IndexDoc::new(SelfRooted::new());
        bb.init_genesis(&b("main")).unwrap();
        bb.create_index_branch(&b("shared"), &b("main")).unwrap();
        bb.record_head(&b("shared"), &d("B"), 2, 2).unwrap();
        assert!(doc_has_key(&bb, &b("shared"), "B"));
        assert!(!doc_has_key(&bb, &b("shared"), "A"), "B has not seen A yet");

        // Import A's history: `shared` now has TWO lineage peers on B, and its
        // frontier is their JOIN -> B's shared head shows BOTH records.
        bb.import(&a_updates).unwrap();
        assert!(
            doc_has_key(&bb, &b("shared"), "A"),
            "join of lineage peers brought session A's record"
        );
        assert!(
            doc_has_key(&bb, &b("shared"), "B"),
            "join of lineage peers kept session B's record"
        );
    }

    /// Number of `docs[doc].heads` entries visible on `branch` (0 if absent).
    fn head_count(md: &IndexDoc, branch: &BranchId, doc: &str) -> usize {
        md.read(branch, |d| {
            d.get_deep_value()
                .as_map()
                .and_then(|root| root.get("docs").cloned())
                .and_then(|v| v.into_map().ok())
                .and_then(|m| m.get(doc).cloned())
                .and_then(|v| v.as_map().cloned())
                .and_then(|per| per.get("heads").cloned())
                .and_then(|v| v.into_map().ok())
                .map(|h| h.len())
                .unwrap_or(0)
        })
        .unwrap()
    }

    #[test]
    fn index_record_head_same_doc_converges_across_sessions() {
        // Two sessions on the SAME branch record a head for the SAME doc "G"
        // under different head-peer keys. With mergeable map-key children,
        // docs["G"].heads is the SAME container on both sides, so both records
        // survive a cross-import. (Divergent op-id children would LWW-drop one.)
        let a = IndexDoc::new(SelfRooted::new());
        a.init_genesis(&b("main")).unwrap();
        a.create_index_branch(&b("shared"), &b("main")).unwrap();
        a.record_head(&b("shared"), &d("G"), 111, 1).unwrap();

        let bb = IndexDoc::new(SelfRooted::new());
        bb.init_genesis(&b("main")).unwrap();
        bb.create_index_branch(&b("shared"), &b("main")).unwrap();
        bb.record_head(&b("shared"), &d("G"), 222, 2).unwrap();

        let a_updates = a.export(ExportMode::all_updates()).unwrap();
        let b_updates = bb.export(ExportMode::all_updates()).unwrap();
        a.import(&b_updates).unwrap();
        bb.import(&a_updates).unwrap();

        // The discriminating assertion: no head record is lost -- the same
        // mergeable heads container carries BOTH sessions' records on both sides.
        assert_eq!(
            head_count(&a, &b("shared"), "G"),
            2,
            "session A: both head records converged (mergeable child, not LWW-dropped)"
        );
        assert_eq!(
            head_count(&bb, &b("shared"), "G"),
            2,
            "session B: both head records converged"
        );
        // The docs subtree is byte-identical across sessions.
        let docs_of = |md: &IndexDoc| {
            md.read(&b("shared"), |d| {
                d.get_deep_value()
                    .as_map()
                    .and_then(|r| r.get("docs").cloned())
            })
            .unwrap()
        };
        let a_docs = docs_of(&a);
        let b_docs = docs_of(&bb);
        assert_eq!(
            a_docs, b_docs,
            "docs subtree converges to an identical value"
        );
    }

    #[test]
    fn index_import_then_record_converges() {
        // The import-THEN-record ordering real peer sync produces (the existing
        // convergence test records BEFORE importing, which masked this). A
        // session first receives a remote import -- advancing its refs==1 bound
        // index head via the resolve CATCH-UP arm (checkout -> detached) -- THEN
        // records locally on that head. Without the catch-up detached-clear, the
        // local record fails `AutoCommitNotStarted` and its head record is
        // dropped (head_count wrong). With it, the record lands and converges.
        let a = IndexDoc::new(SelfRooted::new());
        a.init_genesis(&b("main")).unwrap();
        a.create_index_branch(&b("shared"), &b("main")).unwrap();
        a.record_head(&b("shared"), &d("G"), 111, 1).unwrap();
        let a_updates = a.export(ExportMode::all_updates()).unwrap();

        let bb = IndexDoc::new(SelfRooted::new());
        bb.init_genesis(&b("main")).unwrap();
        bb.create_index_branch(&b("shared"), &b("main")).unwrap();

        // IMPORT FIRST: advances bb's `shared` index head (refs==1) to the
        // lineage join through the catch-up arm.
        bb.import(&a_updates).unwrap();
        assert!(
            doc_has_key(&bb, &b("shared"), "G"),
            "import advanced the head to A's record"
        );

        // THEN record locally on that just-advanced head. This must succeed.
        bb.record_head(&b("shared"), &d("G"), 222, 2).unwrap();
        assert_eq!(
            head_count(&bb, &b("shared"), "G"),
            2,
            "import-then-record kept BOTH head records (no dropped record)"
        );
    }

    // ------------------------------------------------------------------
    // Phase-3 precondition: BranchingDocHead surface, Branch signatures,
    // read-exit-commit, and the two E1 closures.
    // ------------------------------------------------------------------

    #[test]
    fn branching_doc_head_surface_read_write() {
        // read/write hand the closure a &BranchingDocHead exposing head-safe
        // methods (container access, state reads, local write). The history /
        // attachment methods (import/export/checkout/attach/detach/oplog_*/
        // set_peer_id/fork/diff) are ABSENT at the type level -- see the
        // `compile_fail` doctest on `BranchingDocHead`.
        let md = MultiHeadDoc::new(Manual::new());
        md.bind(&b("main"), 0);
        let br = md.branch("main");
        br.write(|head: &BranchingDocHead| {
            head.get_text("t").insert_unicode(0, "hi").unwrap();
        })
        .unwrap();
        let (text, deep_is_map, frontier_nonempty) = br
            .read(|head: &BranchingDocHead| {
                // Exercise head-safe reads (peer_id/len_ops are callable too).
                let _ = head.peer_id();
                let _ = head.len_ops();
                (
                    head.get_text("t").to_string(),
                    head.get_deep_value().is_map(),
                    !head.state_frontiers().is_empty(),
                )
            })
            .unwrap();
        assert_eq!(text, "hi");
        assert!(deep_is_map, "get_deep_value returns the doc map");
        assert!(frontier_nonempty, "state advanced after the write");
    }

    #[test]
    fn read_exit_commits_pending_mutation() {
        let md = MultiHeadDoc::new(Manual::new());
        md.bind(&b("main"), 0);
        let br = md.branch("main");
        // A mutation inside `read` is committed as a normal local edit on exit.
        br.read(|head| head.get_text("t").insert_unicode(0, "X").unwrap())
            .unwrap();
        // After the read returns nothing is left pending, and the edit persists.
        let (pending, text) = br
            .read(|head| (head.get_pending_txn_len(), head.get_text("t").to_string()))
            .unwrap();
        assert_eq!(
            pending, 0,
            "read committed pending ops on exit (nothing left)"
        );
        assert_eq!(text, "X", "the mutation persisted");
    }

    #[test]
    fn e1_raw_checkout_seam_refused_on_owned_head() {
        // The `doc()` re-entry seam: `head.get_text("t").doc()` hands back the
        // raw owned `LoroDoc`. A raw `checkout` on it would move the head's state
        // WITHOUT re-keying the registry, desyncing `tip`/`by_tip` (a sibling
        // bound to the head would read the moved-away state while the registry
        // still says the head's tip). The gate refuses it.
        let md = MultiHeadDoc::new(Manual::new());
        md.bind(&b("main"), 0);
        md.write(&b("main"), |d| {
            d.get_text("t").insert_unicode(0, "M").unwrap()
        })
        .unwrap();

        // Drive the raw seam from inside a read closure.
        let r: LoroResult<()> = md
            .branch("main")
            .read(|head| {
                let raw = head.get_text("t").doc().expect("handler has a doc");
                raw.checkout(&Frontiers::default())
            })
            .unwrap();
        assert!(
            matches!(r, Err(LoroError::OwnedHeadOp("checkout"))),
            "raw checkout on an owned head refused, got {r:?}"
        );
        // The registry is NOT desynced: main still reads its own state.
        assert_eq!(
            md.branch("main")
                .read(|h| h.get_text("t").to_string())
                .unwrap(),
            "M",
            "the head was not moved out from under the registry"
        );
    }

    #[test]
    fn e1_detach_noop_on_owned_head() {
        // `detach()` is gated (no-op) on a registry-owned head -- part of the
        // deferred checkout/attach/detach trio: a branch head's attachment is the
        // registry's, not the raw doc's, to change.
        //
        // (The attach/checkout_to_latest no-op gates are defense-in-depth: after
        // the resolve arms all clear `detached`, and external checkout/detach are
        // gated, no reachable owned head is left detached to observe them on --
        // the discriminating case returns with the deferred create_branch_at
        // read-only-view head. The observable E1-A guard is the raw-checkout-seam
        // test above.)
        let md = MultiHeadDoc::new(Manual::new());
        md.bind(&b("main"), 0);
        md.write(&b("main"), |d| {
            d.get_text("t").insert_unicode(0, "M").unwrap()
        })
        .unwrap();
        let head = md.head_doc(0).unwrap();
        assert!(!head.is_detached(), "owned head starts attached");
        head.detach();
        assert!(!head.is_detached(), "detach refused (no-op) on owned head");
    }

    #[test]
    fn e1_site_b_branch_scoped_new_version() {
        let md = MultiHeadDoc::new(Manual::new());
        md.bind(&b("main"), 0);
        md.bind(&b("feat"), 0); // shared
        md.write(&b("feat"), |d| {
            d.get_text("t").insert_unicode(0, "F").unwrap()
        })
        .unwrap(); // feat -> own head; main keeps root
        md.write(&b("main"), |d| {
            d.get_text("t").insert_unicode(0, "M").unwrap()
        })
        .unwrap();

        let captured: Arc<Mutex<Option<Frontiers>>> = Arc::new(Mutex::new(None));
        let cap = captured.clone();
        let sub = md
            .branch("main")
            .subscribe_root(Arc::new(move |ev: crate::event::DiffEvent| {
                *cap.lock().unwrap() = Some(ev.event_meta.to.clone());
            }))
            .unwrap();

        // A sibling advances the shared union; main's subscriber must not adopt it.
        md.write(&b("feat"), |d| {
            d.get_text("t").insert_unicode(1, "2").unwrap()
        })
        .unwrap();
        // main commits -> its subscriber fires with a BRANCH-SCOPED new_version.
        md.write(&b("main"), |d| {
            d.get_text("t").insert_unicode(1, "2").unwrap()
        })
        .unwrap();
        drop(sub);

        let to = captured
            .lock()
            .unwrap()
            .take()
            .expect("main's subscriber fired");
        let main_frontier = md.branch("main").read(|h| h.state_frontiers()).unwrap();
        let union = md.head_doc(0).unwrap().oplog().lock().frontiers().clone();
        assert_eq!(to, main_frontier, "new_version is main's own frontier");
        assert_ne!(
            to, union,
            "new_version is NOT the shared union (feat excluded)"
        );
    }

    #[test]
    fn branch_fork_ejects_coherent_checkout_able_doc() {
        // main's head is behind the shared union (feat diverged), yet still
        // "attached" (detached flag false). `Branch::fork` must eject via
        // `fork_at(&state_frontiers())` so the ejected snapshot's state matches
        // its own frontier label -- not `LoroDoc::fork()`, which would label
        // main's content with the union frontier (incoherent -> panics on
        // checkout).
        let md = MultiHeadDoc::new(Manual::new());
        md.bind(&b("main"), 0);
        md.bind(&b("feat"), 0); // shared
        md.write(&b("feat"), |d| {
            d.get_text("t").insert_unicode(0, "F").unwrap()
        })
        .unwrap(); // feat diverges; union advances
        md.write(&b("main"), |d| {
            d.get_text("t").insert_unicode(0, "M").unwrap()
        })
        .unwrap(); // main behind the union

        let forked = md.branch("main").fork().unwrap();
        assert_eq!(forked.get_text("t").to_string(), "M");
        // The ejected doc is coherent and checkout-able (would panic if the
        // state/frontier label were mismatched).
        let f = forked.state_frontiers();
        forked.checkout(&Frontiers::default()).unwrap();
        assert_eq!(forked.get_text("t").to_string(), "");
        forked.checkout(&f).unwrap();
        assert_eq!(forked.get_text("t").to_string(), "M");
        // It aliases no registry head: main is untouched.
        assert_eq!(
            md.branch("main")
                .read(|h| h.get_text("t").to_string())
                .unwrap(),
            "M"
        );
    }

    // ------------------------------------------------------------------
    // Phase 3 body: refs>1 copy+advance detached-clear (base), and the
    // Delegated engine (repo create / write / read / merge / advance).
    // ------------------------------------------------------------------

    #[test]
    fn refs_gt1_copy_advance_clears_detached() {
        // The refs>1 copy+advance arm: a SHARED head whose branch target moved
        // off the shared tip is copied and advanced (checkout -> detached). The
        // copy is the branch's live writable head, so a subsequent write must
        // succeed (the clear); WITHOUT it the write fails on a detached head.
        let md = MultiHeadDoc::new(Manual::new());
        // main + feat SHARE the root head; bind `x` too, then `x` diverges (root
        // is shared, so its write copies off) onto its own head and builds "AB",
        // leaving root empty and shared by main + feat (refs 2).
        md.bind(&b("main"), 0);
        md.bind(&b("feat"), 0);
        md.bind(&b("x"), 0);
        md.write(&b("x"), |d| d.get_text("t").insert_unicode(0, "A").unwrap())
            .unwrap(); // x diverges off the shared root onto its own head
        let t_a = md.head_tip(md.bound_head(&b("x")).unwrap()).unwrap();
        md.write(&b("x"), |d| d.get_text("t").insert_unicode(1, "B").unwrap())
            .unwrap();
        assert_eq!(md.refs_of(&b("main")), Some(2));

        // main's target -> the intermediate frontier: resolve(Write) hits the
        // refs>1 copy+advance arm.
        md.policy().set_target(&b("main"), t_a.clone());
        let inner = md
            .write(&b("main"), |d| d.get_text("t").insert_unicode(0, "M"))
            .unwrap();
        assert!(
            inner.is_ok(),
            "write on the advanced shared-copy head succeeded (detached cleared): {inner:?}"
        );
        assert_eq!(
            md.read(&b("main"), |d| d.get_text("t").to_string())
                .unwrap(),
            "MA",
            "the advanced head materialized 'A' and accepted 'M'"
        );
        assert_eq!(
            md.read(&b("feat"), |d| d.get_text("t").to_string())
                .unwrap(),
            "",
            "feat, still on the shared root, is unaffected"
        );
    }

    #[test]
    fn repo_create_write_read_free_creation() {
        let repo = BranchingDocRepo::open().unwrap();
        let doc = repo.open_doc("G".into());
        doc.branch(GENESIS_BRANCH)
            .write(|h| h.get_text("t").insert_unicode(0, "main").unwrap())
            .unwrap();

        // Free creation: draft shares main's content head until it writes.
        repo.create_branch(&b("draft"), &b(GENESIS_BRANCH)).unwrap();
        assert!(repo.branches().contains(&b("draft")));
        assert_eq!(
            doc.branch("draft")
                .read(|h| h.get_text("t").to_string())
                .unwrap(),
            "main",
            "draft sees main's content (shared head, no copy)"
        );

        // draft diverges (copy-on-divergence); main is unaffected.
        doc.branch("draft")
            .write(|h| h.get_text("t").insert_unicode(4, "-draft").unwrap())
            .unwrap();
        assert_eq!(
            doc.branch("draft")
                .read(|h| h.get_text("t").to_string())
                .unwrap(),
            "main-draft"
        );
        assert_eq!(
            doc.branch(GENESIS_BRANCH)
                .read(|h| h.get_text("t").to_string())
                .unwrap(),
            "main",
            "main unaffected by draft's divergent write"
        );
    }

    #[test]
    fn merge_does_not_drop_ops() {
        let repo = BranchingDocRepo::open().unwrap();
        let doc = repo.open_doc("G".into());
        doc.branch(GENESIS_BRANCH)
            .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
            .unwrap();
        repo.create_branch(&b("feature"), &b(GENESIS_BRANCH))
            .unwrap();

        // Both branches diverge with a concurrent edit at the same position.
        doc.branch(GENESIS_BRANCH)
            .write(|h| h.get_text("t").insert_unicode(4, "M").unwrap())
            .unwrap();
        doc.branch("feature")
            .write(|h| h.get_text("t").insert_unicode(4, "F").unwrap())
            .unwrap();
        assert!(!doc.contains(&b(GENESIS_BRANCH), &b("feature")).unwrap());

        let outcome = doc.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap();
        assert_eq!(outcome, MergeOutcome::Merged);

        // main now contains BOTH edits -- no dropped op.
        let text = doc
            .branch(GENESIS_BRANCH)
            .read(|h| h.get_text("t").to_string())
            .unwrap();
        assert_eq!(
            text.chars().count(),
            6,
            "base + both 1-char edits, got {text:?}"
        );
        assert!(
            text.starts_with("base") && text.contains('M') && text.contains('F'),
            "merge kept both branches' ops, got {text:?}"
        );
        // feature is now contained in main.
        assert!(doc.contains(&b(GENESIS_BRANCH), &b("feature")).unwrap());
    }

    #[test]
    fn merge_fast_forward_and_already_contained() {
        let repo = BranchingDocRepo::open().unwrap();
        let doc = repo.open_doc("G".into());
        doc.branch(GENESIS_BRANCH)
            .write(|h| h.get_text("t").insert_unicode(0, "x").unwrap())
            .unwrap();
        repo.create_branch(&b("feature"), &b(GENESIS_BRANCH))
            .unwrap();
        // Only feature advances; main is behind -> fast-forward.
        doc.branch("feature")
            .write(|h| h.get_text("t").insert_unicode(1, "y").unwrap())
            .unwrap();
        assert_eq!(
            doc.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap(),
            MergeOutcome::FastForward
        );
        assert_eq!(
            doc.branch(GENESIS_BRANCH)
                .read(|h| h.get_text("t").to_string())
                .unwrap(),
            "xy"
        );
        // Merging again: already contained.
        assert_eq!(
            doc.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap(),
            MergeOutcome::AlreadyContained
        );
    }

    // ------------------------------------------------------------------
    // N1: two-peer Delegated content-sync convergence (the raison d'etre).
    // ------------------------------------------------------------------

    #[test]
    fn two_peer_content_sync_converges() {
        // Peer A edits branch main on doc G; export A's index + content; peer B
        // imports both. B's `after_import` (Delegated) must re-resolve main so its
        // branch head converges to A's ACTUAL content -- not merely a matching
        // index head_count.
        let a = BranchingDocRepo::open().unwrap();
        let ga = a.open_doc("G".into());
        ga.branch(GENESIS_BRANCH)
            .write(|h| h.get_text("t").insert_unicode(0, "hello").unwrap())
            .unwrap();
        let a_index = a.index().export(ExportMode::all_updates()).unwrap();
        let a_content = ga.export(ExportMode::all_updates()).unwrap();

        let bb = BranchingDocRepo::open().unwrap();
        let gb = bb.open_doc("G".into());
        assert_eq!(
            gb.branch(GENESIS_BRANCH)
                .read(|h| h.get_text("t").to_string())
                .unwrap(),
            "",
            "B starts empty"
        );

        // Sync: index first (B learns where main is), then content (the ops).
        bb.index().import(&a_index).unwrap();
        gb.import(&a_content).unwrap();

        // Inspect main's bound head DIRECTLY (no branch read/resolve), so this
        // isolates `after_import`'s EAGER re-resolve/advance -- a lazy read would
        // re-resolve and converge on its own, masking the ingest.
        let head_id = gb.bound_head(&GENESIS_BRANCH.into()).expect("main bound");
        let head_content = gb.head_doc(head_id).unwrap().get_text("t").to_string();
        assert_eq!(
            head_content, "hello",
            "after_import eagerly advanced main's head to A's actual content"
        );
    }

    #[test]
    fn delete_branch_no_dangling_registry() {
        let repo = BranchingDocRepo::open().unwrap();
        let doc = repo.open_doc("G".into());
        doc.branch(GENESIS_BRANCH)
            .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
            .unwrap();
        repo.create_branch(&b("draft"), &b(GENESIS_BRANCH)).unwrap();
        // Access draft on the content doc so it binds AND diverges onto its own
        // head (refs 1) -- the case a broken delete would leave dangling.
        doc.branch("draft")
            .write(|h| h.get_text("t").insert_unicode(4, "-d").unwrap())
            .unwrap();
        assert!(repo.branches().contains(&b("draft")));
        assert!(
            repo.index().bound_head(&b("draft")).is_some(),
            "index binds draft"
        );
        assert!(
            doc.bound_head(&b("draft")).is_some(),
            "content doc binds draft"
        );

        repo.delete_branch(&b("draft")).unwrap();

        // No dangling / desynced entry anywhere.
        assert!(
            !repo.branches().contains(&b("draft")),
            "draft removed from index lineage"
        );
        assert_eq!(
            repo.index().bound_head(&b("draft")),
            None,
            "index: draft unbound (no dangling bound entry)"
        );
        assert_eq!(
            doc.bound_head(&b("draft")),
            None,
            "content doc: draft unbound (no dangling bound entry)"
        );
        // Pinned root intact; the other branch is unaffected.
        assert!(doc.head_doc(0).is_some(), "pinned root intact");
        assert_eq!(
            doc.branch(GENESIS_BRANCH)
                .read(|h| h.get_text("t").to_string())
                .unwrap(),
            "base",
            "main unaffected by the delete"
        );
    }

    #[test]
    fn create_branch_at_writable_fork_at_historical_frontier() {
        let repo = BranchingDocRepo::open().unwrap();
        let doc = repo.open_doc("G".into());
        // main: "A" then "AB"; capture the frontier after "A".
        doc.branch(GENESIS_BRANCH)
            .write(|h| h.get_text("t").insert_unicode(0, "A").unwrap())
            .unwrap();
        let at_a = doc.frontier_of(&b(GENESIS_BRANCH)).unwrap();
        doc.branch(GENESIS_BRANCH)
            .write(|h| h.get_text("t").insert_unicode(1, "B").unwrap())
            .unwrap();

        // Fork a new branch at the historical "A" frontier.
        doc.create_branch_at(&b("hist"), &at_a).unwrap();
        assert!(repo.branches().contains(&b("hist")));
        // hist materializes to the CHOSEN frontier ("A"), not main's current "AB".
        assert_eq!(
            doc.branch("hist")
                .read(|h| h.get_text("t").to_string())
                .unwrap(),
            "A"
        );
        // head_of resolves hist's index head to a valid id.
        assert!(repo.index().head_of(&b("hist")).is_ok());
        // It is a WRITABLE fork (RFP use-case 3): a write succeeds and diverges.
        doc.branch("hist")
            .write(|h| h.get_text("t").insert_unicode(1, "X").unwrap())
            .unwrap();
        assert_eq!(
            doc.branch("hist")
                .read(|h| h.get_text("t").to_string())
                .unwrap(),
            "AX"
        );
        // main is unaffected.
        assert_eq!(
            doc.branch(GENESIS_BRANCH)
                .read(|h| h.get_text("t").to_string())
                .unwrap(),
            "AB"
        );
    }

    #[test]
    fn subscribe_survives_rebind() {
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
        let md = MultiHeadDoc::new(Manual::new());
        md.bind(&b("main"), 0);
        md.bind(&b("feat"), 0); // root shared, refs 2

        // Subscribe on feat BEFORE it diverges (installed on the shared root).
        let count = Arc::new(AtomicUsize::new(0));
        let c2 = count.clone();
        let sub = md
            .branch("feat")
            .subscribe_root(Arc::new(move |_ev: crate::event::DiffEvent| {
                c2.fetch_add(1, SeqCst);
            }))
            .unwrap();

        // feat writes -> copy-on-divergence -> feat rebinds to a NEW head. The
        // subscription must re-install on the new head and fire for this write.
        md.branch("feat")
            .write(|h| h.get_text("t").insert_unicode(0, "F").unwrap())
            .unwrap();
        assert!(
            count.load(SeqCst) >= 1,
            "subscription fired on feat's new head after copy-on-divergence rebind"
        );
        drop(sub);
    }
}
