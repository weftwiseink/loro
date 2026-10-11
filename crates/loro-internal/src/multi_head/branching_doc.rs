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

/// The branch-vs-fork-base content diff plus the base doc's text length, from
/// one [`diff_from_fork`](MultiHeadDoc::diff_from_fork) call.
#[derive(Debug, Clone)]
pub struct ForkDiff {
    /// The delta from the dynamic merge-base (GCA) to `branch`'s head.
    pub diff: DiffBatch,
    /// Summed text insert char count at the base; non-text containers and
    /// mark-only edits contribute 0.
    pub merge_base_length: usize,
}

/// Inserted unicode scalar values across the text containers of a `DiffBatch`.
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

    /// Whether branch `b` is behind its recorded content (head-lag signal): the
    /// fast-forward guard held it on its last resolve, or its recorded frontier names
    /// ops this oplog does not yet hold (as on a fresh open of a doc edited elsewhere).
    /// Deliberately avoids `frontier_of`, which collapses to the held frontier.
    pub fn lens_held(&self, b: &BranchId) -> bool {
        if self.guard_held_flag(b) {
            return true;
        }
        match self.index().recorded_ids(b, self.doc_id()) {
            Ok(raw) => {
                let ol = self.oplog.lock();
                raw.iter()
                    .any(|id| !ol.vv().get_last(id.peer).is_some_and(|c| c >= id.counter))
            }
            Err(_) => false,
        }
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

    /// Move branch `b` to frontier `to`: record it in the index, then re-resolve.
    /// Additive `record_frontier`, not `set_frontier`: a merge's join keeps every peer's op.
    pub fn advance(&self, b: &BranchId, to: &Frontiers) -> LoroResult<()> {
        self.index().record_frontier(b, self.doc_id(), to)?;
        let _ = self.resolve_with(b, Intent::Read, ResolveCause::Advance)?;
        Ok(())
    }

    /// Create branch `name` as a writable fork at historical frontier `at`, cold until
    /// first access. Errors if `name` exists. Uses `set_frontier` (exact), not an
    /// additive record: see `IndexDoc::set_frontier`.
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

    /// Merge `from` into `into` as frontier advancement: no new ops are created
    /// and none dropped. `into` advances to the join (shrink of the union of
    /// both branches' ids).
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

    /// Resolve a frontier to a version vector. Unresolvable is a loud error,
    /// never an empty vv (which would read as "nothing there").
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

    /// The frontier of `vv(a) ∪ vv(b)`; unresolvable input errors (see `resolve_vv`).
    pub(super) fn union_frontier(&self, a: &Frontiers, b: &Frontiers) -> LoroResult<Frontiers> {
        let mut u = self.resolve_vv(a)?;
        u.merge(&self.resolve_vv(b)?);
        Ok(self.oplog.lock().dag.vv_to_frontiers(&u))
    }

    /// What `branch` changed since it forked from `against`: the delta from their
    /// merge-base to `branch`'s head, plus the base text length. The base is the GCA,
    /// not the static `fork_point`, which goes stale after a cross-merge. Diffs run on
    /// a scratch head ([`read_at`](MultiHeadDoc::read_at)).
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

    /// What merging `source` into `target` would add: the delta from `target`'s head
    /// to the union frontier. Unlike a head-vs-head diff, `target`'s own concurrent
    /// edits are not reported as deletions.
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

    /// Raw head-vs-head diff, for inspection only. Not a merge-base or merge
    /// preview: it double-counts each side's concurrent changes (use
    /// [`diff_from_fork`](Self::diff_from_fork) / [`merge_preview`](Self::merge_preview)).
    pub fn diff_branches(&self, a: &BranchId, b: &BranchId) -> LoroResult<DiffBatch> {
        let fa = self.frontier_of(a)?;
        let fb = self.frontier_of(b)?;
        self.read_at(&fa, |d| d.diff(&fa, &fb))?
    }

    /// Update bytes for the history between two frontiers: everything since
    /// `from` if `to` is `None`, else the range `(from, to]`.
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
