//! `MultiHeadDoc<P>`: the shared base of `BranchingDocRepo`.
//!
//! One `OpLog` backs N materialized heads (each a `LoroDoc` over that op log,
//! with its own `DocState`, `DiffCalculator`, `Txn` lock, observer, and peer).
//! Branches are bound to heads, not owners of them: a head bound to more than
//! one branch is immutable, and the first divergent write copies it
//! (copy-on-divergence). The two variants (`SelfRooted` index, `Delegated`
//! content) differ only in their [`HeadPolicy`]; [`Manual`] is test-only.
//!
//! Shared-head immutability is enforced at the two `DocState` mutation sinks
//! (`apply_local_op`, `apply_diff`) via `head_mode` + `ensure_private`: a mutation
//! reaching a shared head errors with
//! [`LoroError::HeadShared`](loro_common::LoroError::HeadShared) instead of corrupting
//! a co-owner.
//!
//! `by_tip` re-keying and `HeadPolicy::after_commit` run through the injected txn
//! on-commit hook (see [`DocOwner`](crate::DocOwner) / `on_head_committed`), not from `write`.
//! `import` is history-only: ops enter the shared op log via
//! `LoroDoc::import_to_history` under the all-heads barrier, never materializing
//! into a live head.

use loro_common::{InternalString, ID};

mod base;
#[cfg(test)]
mod bench;
mod branch;
mod branching_doc;
mod head_registry;
mod index_doc;
mod policy;
mod repo;
#[cfg(test)]
mod tests;

pub use base::{MultiHeadDoc, MultiHeadInner};
pub use branch::{Branch, BranchSubscription, BranchingDocHead};
pub use branching_doc::{BranchingDoc, ForkDiff, MergeOutcome};
pub use index_doc::{IndexDoc, RecordScope};
pub use policy::{Attribution, Delegated, HeadPolicy, Manual, SelfRooted};
pub use repo::{BranchingDocRepo, IndexImportOutcome, GENESIS_BRANCH};

pub(crate) use head_registry::in_registry_op;

pub type BranchId = InternalString;
pub type DocId = InternalString;
pub type HeadId = u64;

/// How a branch came into being, as recovered from the index attribution runs.
///
/// Single-variant: `runs` stores only `(start, BranchId)`, so merge markers cannot be told
/// from creation markers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchSourceKind {
    /// The branch's creation marker (`branch.name = "<b>"`).
    Create,
}

/// Where a branch came from: its creation marker's op-id plus the branch it
/// was forked from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchSource {
    /// The creation-marker op's own id (its `(peer, counter)`), not its deps.
    pub op: ID,
    /// Always [`BranchSourceKind::Create`] under the current recovery.
    pub kind: BranchSourceKind,
    /// The parent branch NAME (the branch of the marker's deps), or `None` for
    /// genesis (whose creation marker is the root op, with no deps).
    pub parent: Option<BranchId>,
}

/// Pinned root head: born at `refs == 0`, never retired, because
/// `import` / `export` / `materialize` need a live head.
const ROOT_HEAD_ID: HeadId = 0;

/// Fires once per new change in the shared history (head commit or import),
/// with that change's update bytes.
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

/// Why a `resolve` moved a branch; determines the `by`/`origin` of the emitted
/// event. All causes move live state by ops that are not this head's local
/// commits, so all map to `EventTriggerKind::Import`.
#[derive(Debug, Clone)]
pub enum ResolveCause {
    /// A move driven by imported (remote) ops; `origin` is the import's origin.
    Import { origin: InternalString },
    /// A move driven by a local `advance` / `merge`.
    Advance,
    /// A lazy realization on access (`read` / `write` / `subscribe`).
    Access,
}

impl ResolveCause {
    /// The `(origin, by)` an emitted `DocDiff` carries for this cause.
    pub(super) fn to_event(&self) -> (InternalString, crate::event::EventTriggerKind) {
        use crate::event::EventTriggerKind;
        match self {
            ResolveCause::Import { origin } => (origin.clone(), EventTriggerKind::Import),
            ResolveCause::Advance => ("advance".into(), EventTriggerKind::Import),
            ResolveCause::Access => ("resolve".into(), EventTriggerKind::Import),
        }
    }
}
