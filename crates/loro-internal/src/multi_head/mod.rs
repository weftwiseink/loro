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

use loro_common::{InternalString, ID};

mod base;
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
pub use index_doc::IndexDoc;
pub use policy::{Attribution, Delegated, HeadPolicy, Manual, SelfRooted};
pub use repo::{BranchingDocRepo, GENESIS_BRANCH};

pub(crate) use head_registry::in_registry_op;

pub type BranchId = InternalString;
pub type DocId = InternalString;
pub type HeadId = u64;

/// How a branch came into being, as recovered from the index attribution runs.
///
/// A single-variant enum today: the `runs` structure discards a marker's kind
/// (it stores only `(start, BranchId)` per segment), so a run-based recovery
/// classifies every branch's source as its CREATION marker and cannot
/// distinguish merge markers. It is kept an enum so a future merge-source
/// surface can add `Merge` without changing `BranchSource`'s shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchSourceKind {
    /// The branch's creation marker (`branch.name = "<b>"`).
    Create,
}

/// Where a branch came from: the op-id of its creation marker plus the branch
/// it was forked from. The holistic sibling of `tips` (where each branch IS)
/// and `fork_point` (a branch's parent FRONTIER): this surfaces the creation
/// marker op ITSELF, which neither `tips` nor `fork_point` returns.
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

/// Why a `resolve` moved a branch, which determines the `by`/`origin` of the
/// event it delivers so a branch-surfaced doc tags a move exactly as a plain
/// doc would. An import- or advance-driven move, and a lazy realization on
/// access, all move the branch's LIVE state forward by ops that are not this
/// head's local commits, which is what `Import` means to every consumer;
/// `Checkout` (detached time-travel, filtered as non-content) is reserved for a
/// future explicit history view.
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
