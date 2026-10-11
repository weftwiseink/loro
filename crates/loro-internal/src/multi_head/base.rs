use std::ops::Deref;
use std::sync::{Arc, Weak};

use rustc_hash::{FxHashMap, FxHashSet};

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

/// The shared innards of a `MultiHeadDoc`, behind an `Arc` so a head's on-commit
/// hook can hold a `Weak` back to it (see `HeadOwner`).
#[allow(missing_debug_implementations)]
pub struct MultiHeadInner<P: HeadPolicy> {
    pub(super) oplog: Arc<LoroMutex<OpLog>>,
    pub(super) arena: SharedArena,
    pub(super) config: Configure,
    pub(super) lock_group: LoroLockGroup,
    pub(super) visible_op_count: Arc<AtomicUsize>,
    pub(super) reg: LoroMutex<Registry>,
    pub(super) policy: P,
    pub(super) snapshot_head: AtomicU64,
    pub(super) history_subs: SubscriberSetWithQueue<(), HistoryCallback, Vec<u8>>,
    pub(super) local_update_subs: SubscriberSetWithQueue<(), HistoryCallback, Vec<u8>>,
    pub(super) first_commit_subs:
        SubscriberSetWithQueue<(), FirstCommitFromPeerCallback, FirstCommitFromPeerPayload>,
}

/// The shared base: one op log, the head registry, the copy machinery, the sink
/// guard. Cheap to clone (an `Arc`).
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

/// Registry back-pointer installed on each head.
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

/// Feed a head's local commits and first-commit events into the doc-level
/// subscriptions.
pub(super) fn install_forwarders<P: HeadPolicy>(
    inner: &Weak<MultiHeadInner<P>>,
    doc: &LoroDoc,
) -> Vec<Subscription> {
    let w1 = inner.clone();
    let s1 = doc.subscribe_local_update(Box::new(move |bytes| {
        if let Some(inner) = w1.upgrade() {
            inner.history_subs.emit(&(), bytes.clone());
            // Local-only: `import` emits to `history_subs` alone.
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
    /// Open a fresh `MultiHeadDoc` with one empty root head and no bindings.
    pub fn new(policy: P) -> Self {
        let visible_op_count = Arc::new(AtomicUsize::new(0));
        let oplog = OpLog::new(visible_op_count.clone());
        let arena = oplog.arena.clone();
        let config: Configure = oplog.configure.clone();
        let lock_group = LoroLockGroup::new();
        let oplog = Arc::new(lock_group.new_lock(oplog, LockKind::OpLog));

        let inner = Arc::new_cyclic(|w: &Weak<MultiHeadInner<P>>| {
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
            // Pinned root: born at refs == 0, never retired.
            heads.insert(
                ROOT_HEAD_ID,
                Head {
                    doc: root,
                    tip: tip.clone(),
                    refs: 0,
                    unshared: false,
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
                        open_branches: FxHashSet::default(),
                        guard_held: FxHashMap::default(),
                        generation: FxHashMap::default(),
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

    /// Run `f` under the registry lock, marking the thread as inside a registry
    /// op so on-commit hooks defer rather than re-enter the lock.
    pub(super) fn with_reg<R>(&self, f: impl FnOnce(&Self, &mut Registry) -> R) -> R {
        let _g = RegOpGuard::enter();
        let mut reg = self.reg.lock();
        f(self, &mut reg)
    }

    /// Bind a branch directly to an existing head (test / low-level entry).
    pub fn bind(&self, b: &BranchId, head_id: HeadId) {
        self.with_reg(|this, reg| this.rebind(reg, b, head_id));
    }

    /// Unbind a branch and decrement its head's refs (retiring the head at 0,
    /// unless pinned root). No-op if unbound (content docs bind lazily).
    pub fn unbind(&self, b: &BranchId) {
        self.with_reg(|this, reg| {
            this.release_open(reg, b);
            if let Some(old) = reg.bound.remove(b) {
                this.dec_refs(reg, old);
            }
        });
    }

    /// Subscribe on branch `b` (`target = None` for root), surviving rebinds.
    /// Dropping the returned [`BranchSubscription`] removes it.
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

    /// Create branch `new` sharing `from`'s head (O(1)). `OnDivergence` policies only.
    pub fn create_branch(&self, new: &BranchId, from: &BranchId) -> LoroResult<()> {
        debug_assert!(
            P::COPY == CopyMode::OnDivergence,
            "eager policies must use their own branch creation (e.g. create_index_branch)"
        );
        let (from_head, _) = self.resolve(from, Intent::Read)?;
        self.with_reg(|this, reg| {
            debug_assert!(
                !reg.heads.get(&from_head).map(|h| h.unshared).unwrap_or(false),
                "create_branch from a lensed head would share the pin (invariant 6); \
                 content branches must materialize lazily instead"
            );
            this.rebind(reg, new, from_head)
        });
        Ok(())
    }

    /// Resolve `b` for read (never copies) and run `f` on the head. A Private head's
    /// edits are committed on exit so no uncommitted op survives to a `retire`.
    /// Low-level; the consumer surface is `Branch::read`.
    #[doc(hidden)]
    pub fn read<R>(&self, b: &BranchId, f: impl FnOnce(&LoroDoc) -> R) -> LoroResult<R> {
        let (_, doc) = self.resolve(b, Intent::Read)?;
        let r = f(&doc);
        if doc.get_pending_txn_len() > 0 {
            doc.commit_then_renew();
        }
        Ok(r)
    }

    /// Resolve `b` for write (copies if shared), run `f`, then commit.
    /// Low-level; the consumer surface is `Branch::write`.
    #[doc(hidden)]
    pub fn write<R>(&self, b: &BranchId, f: impl FnOnce(&LoroDoc) -> R) -> LoroResult<R> {
        let (_, doc) = self.resolve(b, Intent::Write)?;
        let r = f(&doc);
        doc.commit_then_renew();
        Ok(r)
    }

    /// Txn on-commit hook target, run after the head's locks drop. Defers when
    /// triggered inside a registry op, which re-keys `by_tip` itself.
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
            let unshared = head.unshared;
            let old_tip = std::mem::replace(
                &mut reg.heads.get_mut(&id).expect("head exists").tip,
                new_tip.clone(),
            );
            if old_tip != new_tip {
                if reg.by_tip.get(&old_tip) == Some(&id) {
                    reg.by_tip.remove(&old_tip);
                }
                if !unshared {
                    reg.by_tip.insert(new_tip.clone(), id);
                }
            }
            // Only a refs == 1 head commits, so at most one branch is bound.
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

    /// Run `f` with every head's transaction committed and stopped, then renew them.
    /// All `Txn` locks share one `LockKind`, so heads are stopped sequentially.
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

    /// History-only import: land ops into the shared `OpLog` without moving
    /// any head's state or binding. Heads observe the ops only when the
    /// registry advances them. Emits a history event with the imported bytes.
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
        // Resolve errors are non-fatal: unheld ids are picked up on a later import.
        let touched = self.inner.policy.after_import(self, &status)?;
        for b in touched {
            // `MultiHeadDoc::import` carries no origin, so it is empty.
            let _ = self.resolve_with(&b, Intent::Read, ResolveCause::Import { origin: "".into() });
        }
        if !self.history_subs.inner().is_empty() {
            self.history_subs.emit(&(), bytes.to_vec());
        }
        Ok(status)
    }

    /// Export the shared history. `Snapshot` also carries `snapshot_head`'s
    /// state; other modes are head-independent.
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

    /// Fires per new change in the shared history (head commit or import).
    pub fn subscribe_history(&self, callback: HistoryCallback) -> Subscription {
        let (sub, enable) = self.history_subs.inner().insert((), callback);
        enable();
        sub
    }

    /// Fires per local change on any branch, never on `import` (so a sync adaptor
    /// does not echo imported ops).
    pub fn subscribe_local_update(&self, callback: HistoryCallback) -> Subscription {
        let (sub, enable) = self.local_update_subs.inner().insert((), callback);
        enable();
        sub
    }

    /// The shared op log's version vector: what this peer holds across all branches.
    /// There is no single state version (each head has its own frontier).
    pub fn oplog_vv(&self) -> VersionVector {
        self.oplog.lock().vv().clone()
    }

    /// Fires per first commit from a peer, over all heads. Every head copy mints
    /// a fresh peer, so this fires once per write slot.
    pub fn subscribe_first_commit_from_peer(
        &self,
        callback: FirstCommitFromPeerCallback,
    ) -> Subscription {
        let (sub, enable) = self.first_commit_subs.inner().insert((), callback);
        enable();
        sub
    }

    pub fn head_count(&self) -> usize {
        self.reg.lock().heads.len()
    }

    /// Low-level raw head handle for tests / diagnostics.
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
