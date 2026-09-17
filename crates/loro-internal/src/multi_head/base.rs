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
use crate::subscription::Subscriber;
use crate::sync::{AtomicU64, AtomicU8, AtomicUsize};
use crate::utils::subscription::{SubscriberSetWithQueue, Subscription};
use crate::version::{Frontiers, VersionVector};
use crate::{DocOwner, LoroDoc, HEAD_MODE_PRIVATE};
use loro_common::{ContainerID, HasIdSpan, IdSpan, LoroEncodeError, LoroResult};

use super::head_registry::{in_registry_op, BranchSub, Head, RegOpGuard, Registry};
use super::ROOT_HEAD_ID;
use super::*;

/// The shared innards of a `MultiHeadDoc`, held behind an `Arc` so a head's
/// injected on-commit hook can hold a `Weak` back to it (see `HeadOwner`).
#[allow(missing_debug_implementations)] // holds LoroMutex<OpLog>/live heads; no useful Debug
pub struct MultiHeadInner<P: HeadPolicy> {
    pub(super) oplog: Arc<LoroMutex<OpLog>>,
    pub(super) arena: SharedArena,
    pub(super) config: Configure,
    pub(super) lock_group: LoroLockGroup,
    pub(super) visible_op_count: Arc<AtomicUsize>,
    pub(super) reg: LoroMutex<Registry>,
    pub(super) policy: P,
    /// Head whose materialized state a `Snapshot` export carries (default: the
    /// root head). Ops/history export is head-independent (shared op log).
    pub(super) snapshot_head: AtomicU64,
    /// One event per new change in the shared history, aggregated over heads and
    /// imports.
    pub(super) history_subs: SubscriberSetWithQueue<(), HistoryCallback, Vec<u8>>,
    /// One event per LOCAL new change in the shared history (a head commit on any
    /// branch), aggregated over heads. UNLIKE `history_subs`, this does NOT fire
    /// on `import`: it mirrors `LoroDoc::subscribe_local_update` exactly (local
    /// edits only), so a wire adaptor driven off it never re-broadcasts imported
    /// (remote) ops back onto the wire.
    pub(super) local_update_subs: SubscriberSetWithQueue<(), HistoryCallback, Vec<u8>>,
    /// One event per first commit from a peer, aggregated over heads. Every head
    /// copy mints a fresh peer, so this fires once per write slot.
    pub(super) first_commit_subs:
        SubscriberSetWithQueue<(), FirstCommitFromPeerCallback, FirstCommitFromPeerPayload>,
}

/// The shared base: one op log, the head registry, the copy machinery, the sink
/// guard. Written once; both variants build on it. Cheap to clone (an `Arc`).
#[allow(missing_debug_implementations)]
pub struct MultiHeadDoc<P: HeadPolicy> {
    pub(super) inner: Arc<MultiHeadInner<P>>,
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
pub(super) struct HeadOwner<P: HeadPolicy> {
    pub(super) inner: Weak<MultiHeadInner<P>>,
    pub(super) head_id: HeadId,
}

impl<P: HeadPolicy> DocOwner for HeadOwner<P> {
    fn on_head_commit(&self, id_span: IdSpan) {
        if let Some(inner) = self.inner.upgrade() {
            MultiHeadDoc { inner }.on_head_committed(self.head_id, id_span);
        }
    }
}

/// Install the per-head forwarders that feed a head's local commits and
/// first-commit-from-peer events into the doc-level aggregated subscriptions.
pub(super) fn install_forwarders<P: HeadPolicy>(
    inner: &Weak<MultiHeadInner<P>>,
    doc: &LoroDoc,
) -> Vec<Subscription> {
    let w1 = inner.clone();
    let s1 = doc.subscribe_local_update(Box::new(move |bytes| {
        if let Some(inner) = w1.upgrade() {
            inner.history_subs.emit(&(), bytes.clone());
            // Local-only stream: fed from the per-head local-update forwarder and
            // NEVER from `import` (which emits to `history_subs` alone). This is
            // what `MultiHeadDoc::subscribe_local_update` exposes.
            inner.local_update_subs.emit(&(), bytes.clone());
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
                local_update_subs: SubscriberSetWithQueue::new(),
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
    pub(super) fn with_reg<R>(&self, f: impl FnOnce(&Self, &mut Registry) -> R) -> R {
        let _g = RegOpGuard::enter();
        let mut reg = self.reg.lock();
        f(self, &mut reg)
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
    fn on_head_committed(&self, id: HeadId, committed: IdSpan) {
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
            debug_assert_eq!(
                last,
                committed.id_last(),
                "a head's post-commit tip is the committed change's last id"
            );
            self.inner.policy.after_commit(self, &b, committed);
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

    /// One event per LOCAL new change in the shared history (a head commit on ANY
    /// branch), carrying that change's update bytes. Mirrors
    /// `LoroDoc::subscribe_local_update`: it fires ONLY on this peer's local edits,
    /// NEVER on `import`, so a sync adaptor driven off it does not echo imported
    /// ops back onto the wire. Aggregated over every head, so an edit on any branch
    /// (including a copy-on-divergence head for a non-main branch) is delivered.
    pub fn subscribe_local_update(&self, callback: HistoryCallback) -> Subscription {
        let (sub, enable) = self.local_update_subs.inner().insert((), callback);
        enable();
        sub
    }

    /// The version vector of the SHARED op log: everything this peer holds across
    /// all branches. The op log is the sync unit (`export({mode:"update", from})`
    /// reads it), so this is the version a wire adaptor compares against a peer's
    /// version and exports the delta from. A `MultiHeadDoc` has no single
    /// materialized DocState (each head sits at its own frontier), so both the
    /// "state" and "oplog" version a `LoroDoc` distinguishes collapse to this one
    /// shared-oplog vv.
    pub fn oplog_vv(&self) -> VersionVector {
        self.oplog.lock().vv().clone()
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
}
