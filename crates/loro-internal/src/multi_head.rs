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
use std::ops::Deref;
use std::sync::{Arc, Weak};

use rustc_hash::FxHashMap;

use crate::arena::SharedArena;
use crate::configure::Configure;
use crate::encoding::{ExportMode, ImportStatus};
use crate::lock::{LockKind, LoroLockGroup, LoroMutex};
use crate::oplog::OpLog;
use crate::pre_commit::{FirstCommitFromPeerCallback, FirstCommitFromPeerPayload};
use crate::state::DocState;
use crate::sync::{AtomicU64, AtomicU8, AtomicUsize};
use crate::utils::subscription::{SubscriberSetWithQueue, Subscription};
use crate::version::Frontiers;
use crate::{DocOwner, LoroDoc, HEAD_MODE_PRIVATE};
use loro_common::{IdSpan, InternalString, LoroEncodeError, LoroResult, ID};

pub type BranchId = InternalString;
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

fn in_registry_op() -> bool {
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
    next_id: HeadId,
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
                        next_id: ROOT_HEAD_ID + 1,
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
    }

    /// Bind a branch directly to an existing head (test / low-level entry).
    pub fn bind(&self, b: &BranchId, head_id: HeadId) {
        self.with_reg(|this, reg| this.rebind(reg, b, head_id));
    }

    /// Create branch `new` bound to wherever `from` currently resolves: the
    /// free, O(1) branch-from-a-live-head path (share `from`'s head), or an
    /// eager copy for an `Eager` policy.
    pub fn create_branch(&self, new: &BranchId, from: &BranchId) -> LoroResult<()> {
        let (from_head, _) = self.resolve(from, Intent::Read)?;
        self.with_reg(|this, reg| {
            if P::COPY == CopyMode::Eager {
                let copy = this.copy_head(reg, from_head);
                this.rebind(reg, new, copy);
            } else {
                this.rebind(reg, new, from_head);
            }
        });
        Ok(())
    }

    // --- resolution --------------------------------------------------------

    /// Resolve branch `b` to the head it should use, copying-and-rebinding when
    /// a `Write` reaches a shared head. Returns the head id and a handle.
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
                            this.advance_in_place(reg, h, &target)?;
                            h
                        }
                        Some(h) => {
                            let c = this.copy_head(reg, h);
                            this.advance_in_place(reg, c, &target)?;
                            this.rebind(reg, b, c);
                            c
                        }
                        None => {
                            let m = this.materialize(reg, &target)?;
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
    /// mutation inside `f` on a shared head errors from the sink.
    pub fn read<R>(&self, b: &BranchId, f: impl FnOnce(&LoroDoc) -> R) -> LoroResult<R> {
        let (_, doc) = self.resolve(b, Intent::Read)?;
        Ok(f(&doc))
    }

    /// Resolve `b` for write (copies if shared, handing back a `Private` head),
    /// run `f`, then commit. The commit fires the injected on-commit hook
    /// (`on_head_committed`), which re-keys `by_tip` and notifies the policy;
    /// there is no synchronous re-keying here.
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
        // a materialization; the sink guard refuses it.
        let shared = md.head_doc(0).unwrap();
        let r = shared.checkout(&f);
        assert!(
            matches!(r, Err(LoroError::HeadShared)),
            "checkout (apply_diff) on a shared head refused, got {r:?}"
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
}
