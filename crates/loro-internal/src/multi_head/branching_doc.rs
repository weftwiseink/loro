use std::cmp::Ordering;

use crate::version::{shrink_frontiers, Frontiers};
use loro_common::{LoroError, LoroResult};

use super::*;

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
    ///
    /// Uses ADDITIVE `record_frontier` (not `set_frontier`): `advance` / `merge`
    /// move a LIVE branch FORWARD, accumulating per-peer positions -- a merge's
    /// join must keep every peer's op. This is the opposite of `create_branch_at`,
    /// which SETS an exact frontier and must exclude other peers (`set_frontier`).
    pub fn advance(&self, b: &BranchId, to: &Frontiers) -> LoroResult<()> {
        self.index().record_frontier(b, self.doc_id(), to)?;
        // An advance/merge moves the branch's live state forward: tag `Advance`.
        let _ = self.resolve_with(b, Intent::Read, ResolveCause::Advance)?;
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
    ///
    /// Errors if `name` already exists (no silent reposition). The fork's
    /// recorded frontier is set to EXACTLY `at` via `set_frontier` -- overriding,
    /// not additively merging into, the parent frontier the eager
    /// `create_index_branch` copy seeds. Additive merge would leave stale
    /// other-peer entries under a MULTI-PEER parent, silently pulling those peers'
    /// ops into the fork.
    pub fn create_branch_at(&self, name: &BranchId, at: &Frontiers) -> LoroResult<()> {
        if self.index().branches().contains(name) {
            return Err(LoroError::ArgErr(
                format!("cannot create_branch_at: branch '{name}' already exists").into_boxed_str(),
            ));
        }
        self.index()
            .create_index_branch(name, &GENESIS_BRANCH.into())?;
        self.index().set_frontier(name, self.doc_id(), at)?;
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
