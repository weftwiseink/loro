use std::cmp::Ordering;

use crate::change::Timestamp;
use crate::container::IntoContainerId;
use crate::event::Index;
#[cfg(feature = "counter")]
use crate::handler::counter::CounterHandler;
use crate::handler::{
    ListHandler, MapHandler, MovableListHandler, TextHandler, TreeHandler, ValueOrHandler,
};
use crate::loro::CommitOptions;
use crate::pre_commit::PreCommitCallback;
use crate::subscription::Subscriber;
use crate::undo::DiffBatch;
use crate::utils::subscription::Subscription;
use crate::version::{Frontiers, VersionVector};
use crate::LoroDoc;
use loro_common::{ContainerID, LoroResult, LoroValue, PeerID};

use super::*;

/// The head-safe surface handed to a `Branch::read` / `write` closure: a
/// pass-through newtype over a resolved head's `LoroDoc` exposing only methods
/// that read or write this head's state or transaction.
///
/// There is no `Deref` or inner accessor, and history / attachment / identity
/// ops (`import*`, `export*`, `checkout`, `attach`, `checkout_to_latest`,
/// `detach`, `oplog_*`, `fork`, `set_peer_id`, config setters) are absent from
/// the type: they touch the shared op log or the head's attachment and are
/// incoherent on a registry head. `diff` is an intentional read passthrough
/// since `LoroDoc::diff` restores the head's attach-state and frontier on exit.
/// `subscribe*` and `checkout` are not branch-correct here (they must
/// re-install on rebind) and live on `Branch`.
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
/// History / attachment ops do not compile:
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
    /// The content delta from `a` to `b`, `apply_diff`-able onto `a`. The head's
    /// attach-state and frontier are restored on exit.
    pub fn diff(&self, a: &Frontiers, b: &Frontiers) -> LoroResult<DiffBatch> {
        self.0.diff(a, b)
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

    /// Commit this head's pending transaction and start the next. `commit_with`
    /// is omitted: its internal form leaks a lock guard.
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

    pub fn subscribe_pre_commit(&self, callback: PreCommitCallback) -> Subscription {
        self.0.subscribe_pre_commit(callback)
    }
    pub fn free_diff_calculator(&self) {
        self.0.free_diff_calculator()
    }
}

/// Handle for a branch subscription (`Branch::subscribe` / `subscribe_root`).
/// Unlike a raw `Subscription` on one head, it follows the branch across
/// rebinds. Dropping it unsubscribes.
#[allow(missing_debug_implementations)]
pub struct BranchSubscription {
    pub(super) remove: Option<Box<dyn FnOnce() + Send + Sync>>,
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

/// A branch handle: a name plus its `MultiHeadDoc`. Every operation
/// re-resolves the branch's head at call time. Only `fork` and `lens` name
/// `LoroDoc`; everything else stays on the head-safe `BranchingDocHead`.
#[allow(missing_debug_implementations)]
pub struct Branch<'a, P: HeadPolicy> {
    doc: &'a MultiHeadDoc<P>,
    name: BranchId,
}

impl<P: HeadPolicy> Branch<'_, P> {
    /// Resolve the branch's head for read and run `f` on it. A mutation on a
    /// Private head is committed on exit; on a shared head it errors at the sink.
    pub fn read<R>(&self, f: impl FnOnce(&BranchingDocHead) -> R) -> LoroResult<R> {
        self.doc
            .read(&self.name, |d| f(&BranchingDocHead::from_head(d.clone())))
    }

    /// Resolve the branch's head for write (copy-on-divergence), run `f`, commit.
    pub fn write<R>(&self, f: impl FnOnce(&BranchingDocHead) -> R) -> LoroResult<R> {
        self.doc
            .write(&self.name, |d| f(&BranchingDocHead::from_head(d.clone())))
    }

    // NOTE(claude-opus-4-8/branchingdocrepo-multiheaddoc): `checkout` /
    // `attach` / `checkout_to_latest` are deliberately not exposed: a branch has
    // no single "latest", and loro's `attach` snaps to the shared union, which
    // corrupts an owned head. The raw `LoroDoc` seam gates them.

    /// Branch-scoped container subscription, delivering the same events a plain
    /// `LoroDoc` subscription would: local commits, plus every advance / merge /
    /// import-driven move of the branch, tagged `Import` (never `Checkout`).
    pub fn subscribe(&self, cid: &ContainerID, cb: Subscriber) -> LoroResult<BranchSubscription> {
        self.doc.subscribe_branch(&self.name, Some(cid.clone()), cb)
    }

    /// Branch-scoped root subscription (see `subscribe`).
    pub fn subscribe_root(&self, cb: Subscriber) -> LoroResult<BranchSubscription> {
        self.doc.subscribe_branch(&self.name, None, cb)
    }

    /// Eject a standalone `LoroDoc` (independent op log, full surface) that
    /// aliases no registry head.
    ///
    /// Uses `fork_at(&head.state_frontiers())`, not `LoroDoc::fork()`: `fork()`
    /// assumes an attached doc's state equals `oplog.frontiers()`, but a
    /// registry head is behind the shared union, so the ejected doc would carry
    /// the union frontier label over this head's state and panic on first
    /// `checkout` (`richtext_state.rs`).
    pub fn fork(&self) -> LoroResult<LoroDoc> {
        let (_, head) = self.doc.resolve(&self.name, Intent::Read)?;
        head.fork_at(&head.state_frontiers())
    }

    /// The branch head's own aliased `LoroDoc`, which editor / undo / presence /
    /// canvas bind to. The head is resolved for write, then pinned as the
    /// branch's unique unshared head (`open_lens`), so no sibling can flip it
    /// shared or retire it. Unlike `fork`, an edit + `commit` advances this
    /// branch's head and fires its subscriptions, and a forward-extending import
    /// advances it in place (fast-forward-only).
    ///
    /// NOTE(claude-opus-4-8/unified-branching-content/P1): this exposes the full
    /// `LoroDoc` history surface; the `BranchingDocHead` containment is likely
    /// over-cautious since the shared oplog is append-only. By convention, not
    /// type, consumers only read/subscribe/edit/commit/undo/cursor. `checkout`
    /// is incoherent while the head is behind the shared union (see `fork`).
    pub fn lens(&self) -> LoroResult<LoroDoc> {
        self.doc.open_lens(&self.name)
    }
}

impl<P: HeadPolicy> MultiHeadDoc<P> {
    /// A handle to branch `name`, re-resolved per operation since its head can
    /// change underneath (a copy, a rebind). This is the consumer surface.
    pub fn branch(&self, name: impl Into<BranchId>) -> Branch<'_, P> {
        Branch {
            doc: self,
            name: name.into(),
        }
    }
}
