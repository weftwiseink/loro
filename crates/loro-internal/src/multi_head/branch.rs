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

    /// Branch-scoped container subscription. Parity contract: it delivers the
    /// same events for every update as a subscription on a plain `LoroDoc` would.
    /// It fires on the branch's current head (local commits) AND, because the
    /// registry re-installs it across a rebind and every head transition
    /// synthesizes and dispatches its diff, on every advance / merge / import-
    /// driven move of the branch. A move is tagged `Import` (live content), never
    /// `Checkout`.
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

impl<P: HeadPolicy> MultiHeadDoc<P> {
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
