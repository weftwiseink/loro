use std::borrow::Cow;
use std::cmp::Ordering;

use crate::dag::DagUtils;
use crate::encoding::ExportMode;
use crate::event::Diff;
use crate::handler::TextDelta;
use crate::undo::DiffBatch;
use crate::version::{shrink_frontiers, Frontiers, VersionVector};
use loro_common::{IdSpan, LoroError, LoroResult};

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

/// The branch-vs-fork-base content diff plus the base doc's text length, both
/// from ONE [`diff_from_fork`](MultiHeadDoc::diff_from_fork) call so a consumer
/// computing edited-weight (`content_delta / merge_base_length`) needs no second
/// round-trip.
#[derive(Debug, Clone)]
pub struct ForkDiff {
    /// The delta from the dynamic merge-base (GCA) to `branch`'s head.
    pub diff: DiffBatch,
    /// The base doc's shallow TEXT length (summed insert char count at the base;
    /// mark-only edits and non-text containers contribute 0).
    pub merge_base_length: usize,
}

/// The summed TEXT-container insert char count across a `DiffBatch`: the count
/// of inserted unicode scalar values, ignoring deletes, retains, mark-only
/// attributes, and non-text containers.
fn text_insert_len(batch: &DiffBatch) -> usize {
    batch
        .iter()
        .map(|(_, d)| match d {
            Diff::Text(t) => TextDelta::from_text_diff(t.iter())
                .iter()
                .map(|td| match td {
                    TextDelta::Insert { insert, .. } => insert.chars().count(),
                    _ => 0,
                })
                .sum(),
            _ => 0,
        })
        .sum()
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

    /// Resolve a content frontier to a version vector, or a LOUD error if the
    /// oplog cannot resolve it. NEVER a silent `None`: an unresolvable frontier
    /// must surface, not degrade to an empty vv (which reads downstream as
    /// "nothing there", the opposite of "unresolvable input").
    fn resolve_vv(&self, f: &Frontiers) -> LoroResult<VersionVector> {
        self.oplog.lock().dag.frontiers_to_vv(f).ok_or_else(|| {
            f.iter()
                .next()
                .map(LoroError::FrontiersNotFound)
                .unwrap_or_else(|| {
                    LoroError::Unknown("cannot resolve frontier to a version vector".into())
                })
        })
    }

    /// The union (merge) frontier of two content frontiers: `vv(a) ∪ vv(b)`
    /// mapped back to a frontier. A loro merge is an ADDITIVE op union, so this
    /// is the frontier the merged doc would sit at. A frontier the oplog cannot
    /// resolve is a loud error (see [`resolve_vv`](Self::resolve_vv)).
    pub(super) fn union_frontier(&self, a: &Frontiers, b: &Frontiers) -> LoroResult<Frontiers> {
        let mut u = self.resolve_vv(a)?;
        u.merge(&self.resolve_vv(b)?);
        Ok(self.oplog.lock().dag.vv_to_frontiers(&u))
    }

    /// What `branch` changed since it forked from `against`: the delta from the
    /// DYNAMIC merge-base (git-style GCA over the two CONTENT frontiers) to
    /// `branch`'s head, plus the base doc's text length.
    ///
    /// The baseline is `find_common_ancestor(frontier_of(branch),
    /// frontier_of(against))`, NOT the static `index.fork_point(branch)`: the
    /// GCA MOVES as branches cross-merge, so this stays the true "changed since
    /// they diverged" delta after a cross-merge, where the static fork frontier
    /// would drift wrong.
    ///
    /// The two diffs run on a throwaway scratch head (the shared oplog holds
    /// every branch's ops, and [`read_at`](MultiHeadDoc::read_at) retires the
    /// scratch after), so no live head is disturbed.
    pub fn diff_from_fork(
        &self,
        branch: &BranchId,
        against: &BranchId,
    ) -> LoroResult<ForkDiff> {
        let fb = self.frontier_of(branch)?;
        let fa = self.frontier_of(against)?;
        let base = self.oplog.lock().dag.find_common_ancestor(&fb, &fa).0;
        self.read_at(&fb, |d| -> LoroResult<ForkDiff> {
            let diff = d.diff(&base, &fb)?;
            let base_diff = d.diff(&Frontiers::default(), &base)?;
            Ok(ForkDiff {
                diff,
                merge_base_length: text_insert_len(&base_diff),
            })
        })?
    }

    /// What merging `source` into `target` would ADD to `target`: the delta from
    /// `target`'s head to the union frontier. EMPTY when `source ⊆ target`
    /// (already-contained), and free of spurious deletions of `target`'s own
    /// edits in the diverged case, because the union is a SUPERSET of `target`.
    ///
    /// This is the additive-union merge effect, NOT a head-vs-head diff (which
    /// would report `target`'s concurrent edits as deletions). An unresolvable
    /// frontier is a loud error (see [`union_frontier`](Self::union_frontier)).
    pub fn merge_preview(
        &self,
        target: &BranchId,
        source: &BranchId,
    ) -> LoroResult<DiffBatch> {
        let ft = self.frontier_of(target)?;
        let fs = self.frontier_of(source)?;
        let union = self.union_frontier(&ft, &fs)?;
        self.read_at(&ft, |d| d.diff(&ft, &union))?
    }

    /// A raw head-vs-head content diff between two branches: `diff(frontier_of(a),
    /// frontier_of(b))`. For a "compare two branches" inspection view ONLY --
    /// NEVER a merge-base or merge-preview backing, which it would mis-serve by
    /// double-counting each side's concurrent changes (use
    /// [`diff_from_fork`](Self::diff_from_fork) / [`merge_preview`](Self::merge_preview)).
    pub fn diff_branches(&self, a: &BranchId, b: &BranchId) -> LoroResult<DiffBatch> {
        let fa = self.frontier_of(a)?;
        let fb = self.frontier_of(b)?;
        self.read_at(&fa, |d| d.diff(&fa, &fb))?
    }

    /// The oplog history between two content frontiers, as update bytes for
    /// evidence / inspection. `to = None` exports everything since `from`
    /// (`ExportMode::Updates`); a bounded `to` exports the range `(from, to]`
    /// (`ExportMode::UpdatesInRange`). An unresolvable frontier is a loud error.
    pub fn export_oplog_slice(
        &self,
        from: &Frontiers,
        to: Option<&Frontiers>,
    ) -> LoroResult<Vec<u8>> {
        let vv_from = self.resolve_vv(from)?;
        let mode = match to {
            None => ExportMode::Updates {
                from: Cow::Owned(vv_from),
            },
            Some(to) => {
                let vv_to = self.resolve_vv(to)?;
                let spans: Vec<IdSpan> = vv_to.sub_iter(&vv_from).collect();
                ExportMode::UpdatesInRange {
                    spans: Cow::Owned(spans),
                }
            }
        };
        self.export(mode).map_err(LoroError::from)
    }
}
