use std::sync::{Arc, OnceLock};

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::container::map::MapSet;
use crate::encoding::ImportStatus;
use crate::lock::{LockKind, LoroMutex};
use crate::op::InnerContent;
use crate::oplog::OpLog;
use crate::version::{shrink_frontiers, Frontiers};
use loro_common::{
    ContainerID, ContainerType, Counter, HasIdSpan, IdSpan, InternalString, Lamport, LoroError,
    LoroResult, LoroValue, PeerID, ID,
};

use super::*;

/// The only axis on which the two `MultiHeadDoc` variants differ.
pub trait HeadPolicy: Send + Sync + 'static + Sized {
    /// Copy discipline for this variant.
    const COPY: CopyMode;

    /// The frontier branch `b` should currently resolve to. Called while the
    /// registry lock is held, so it MUST NOT lock the registry.
    fn target(&self, this: &MultiHeadDoc<Self>, b: &BranchId) -> LoroResult<Frontiers>;

    /// Publish the new tip after `b`'s head commits `committed` (the change's
    /// full id span; its last id is the head's new single tip). Runs outside
    /// the registry lock.
    fn after_commit(&self, this: &MultiHeadDoc<Self>, b: &BranchId, committed: IdSpan);

    /// The branches to re-resolve after `status`'s spans landed. Errs on a
    /// malformed import (an op the index policy cannot attribute).
    fn after_import(
        &self,
        this: &MultiHeadDoc<Self>,
        status: &ImportStatus,
    ) -> LoroResult<Vec<BranchId>>;
}

/// Root map carrying every branch's creation marker: `branch.name = "<b>"`.
///
/// The name is a plain string readable from the op log alone, so one root container
/// serves all branches and names are unconstrained by container-id charset.
pub(super) const BRANCH_ROOT: &str = "branch";
pub(super) const BRANCH_NAME_KEY: &str = "name";

/// Key on the `branch` root map for a merge marker: `branch.merge = "<seq>:<b>"`
/// where `<seq>` is a monotonic integer.
///
/// Cannot reuse the `name` key: `branch.name` may already equal `into`, and a
/// same-value map set emits no op, so the merge would be silently dropped on every
/// peer. The `<seq>` prefix guarantees the value differs from the last merge's. The
/// branch name is everything after the first `:`, so names containing `:` survive.
pub(super) const BRANCH_MERGE_KEY: &str = "merge";

/// The branch a marker op names, or `None` for a non-marker op. Reads the op
/// log only, no materialized state. See [`BRANCH_MERGE_KEY`] for the merge form.
fn marker_branch_of(ol: &OpLog, op: &crate::op::Op) -> Option<BranchId> {
    let InnerContent::Map(MapSet {
        key,
        value: Some(LoroValue::String(v)),
    }) = &op.content
    else {
        return None;
    };
    let Some(ContainerID::Root {
        name: root,
        container_type: ContainerType::Map,
    }) = ol.arena.idx_to_id(op.container)
    else {
        return None;
    };
    if root.as_str() != BRANCH_ROOT {
        return None;
    }
    match key.as_str() {
        BRANCH_NAME_KEY => Some(InternalString::from(v.as_ref())),
        BRANCH_MERGE_KEY => v
            .as_ref()
            .split_once(':')
            .map(|(_, name)| InternalString::from(name)),
        _ => None,
    }
}

/// Loud error for an imported op the fold cannot attribute to a branch. Unreachable
/// for well-formed input, so the import is rejected rather than silently attributed
/// to a default branch (data loss).
fn invalid_attribution(msg: impl std::fmt::Display) -> LoroError {
    tracing::error!("index attribution: invalid import: {msg}");
    LoroError::Unknown(format!("index attribution: invalid import: {msg}").into_boxed_str())
}

/// The index's derived branch projection, folded from the op log by causal
/// attribution: an op belongs to the branch named by the nearest marker in its
/// causal past.
#[derive(Debug, Default)]
pub struct Attribution {
    /// Branch -> its index frontier. `target(b)` is exactly `tips[b]`.
    pub tips: FxHashMap<BranchId, Frontiers>,
    /// Per-peer runs sorted by start counter: op `(peer, c)` belongs to the run
    /// with the greatest start `<= c`.
    pub runs: FxHashMap<PeerID, SmallVec<[(Counter, BranchId); 1]>>,
}

impl Attribution {
    /// The branch the op at `id` belongs to, if attributed.
    pub fn branch_at(&self, id: ID) -> Option<&BranchId> {
        self.runs
            .get(&id.peer)?
            .iter()
            .rev()
            .find(|(s, _)| *s <= id.counter)
            .map(|(_, b)| b)
    }

    /// Make the run covering `(peer, start)` name `b`. Idempotent; relabels a
    /// run starting exactly at `start` (re-fold of a stored, merged change).
    fn ensure_run(&mut self, peer: PeerID, start: Counter, b: &BranchId) {
        let runs = self.runs.entry(peer).or_default();
        match runs.iter().rposition(|(s, _)| *s <= start) {
            Some(i) if runs[i].1 == *b => {}
            Some(i) if runs[i].0 == start => runs[i].1 = b.clone(),
            Some(i) => runs.insert(i + 1, (start, b.clone())),
            None => runs.insert(0, (start, b.clone())),
        }
    }

    /// The single branch every dependency attributes to, or a loud error if the
    /// deps are unknown, disagree, or absent (all malformed-input cases).
    pub(super) fn branch_of_deps(&self, deps: &Frontiers) -> LoroResult<BranchId> {
        let mut found: Option<BranchId> = None;
        for dep in deps.iter() {
            let b = self.branch_at(dep).ok_or_else(|| {
                invalid_attribution(format!("dependency {dep} lies outside every attribution run"))
            })?;
            match &found {
                None => found = Some(b.clone()),
                Some(f) if f == b => {}
                Some(f) => {
                    return Err(invalid_attribution(format!(
                        "a non-marker change's dependencies span two branches ({f} and {b})"
                    )))
                }
            }
        }
        found.ok_or_else(|| invalid_attribution("a root change carries no marker and no deps"))
    }
}

/// The index's resolution policy: its own op log is the source of truth for
/// where each branch is. Self-rooting breaks the `BranchingDoc`-depends-on-index
/// circularity: the index consults no other index.
///
/// Attribution is causal, not by peer identity: a change belongs to the branch
/// of its own marker if it carries one, else to the branch of its dependencies
/// (which must agree, or the import is rejected). A head always sits at
/// `tips[b]` when it commits (`resolve` guarantees it), so a head minted by
/// `resolve` needs no marker: its first change's deps already attribute it.
///
/// `(tips, runs)` only mutate under the `Attribution` leaf lock, one whole fold
/// or `after_commit` per critical section, so readers see all of a fold or none.
/// `after_commit` publishes after the commit returns; a session is
/// single-writer, so its next `resolve` runs after the publish.
#[allow(missing_debug_implementations)]
pub struct SelfRooted {
    // Lazy: the lock group exists only once `MultiHeadDoc::new` has run.
    // `LockKind::Attribution` is the leaf, acquired alone or after `OpLog`.
    attr: OnceLock<LoroMutex<Attribution>>,
}

impl Default for SelfRooted {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfRooted {
    pub fn new() -> Self {
        SelfRooted {
            attr: OnceLock::new(),
        }
    }

    pub(super) fn attribution(&self, this: &MultiHeadDoc<SelfRooted>) -> &LoroMutex<Attribution> {
        self.attr.get_or_init(|| {
            this.lock_group
                .new_lock(Attribution::default(), LockKind::Attribution)
        })
    }
}

/// One imported change, as the fold sees it.
struct Imported {
    lamport: Lamport,
    peer: PeerID,
    start: Counter,
    last: Counter,
    deps: Frontiers,
    /// `(op counter, branch)` per marker op. A stored change may merge several
    /// commits, so a marker can sit past the first op; each starts a segment.
    markers: SmallVec<[(Counter, BranchId); 1]>,
}

impl HeadPolicy for SelfRooted {
    const COPY: CopyMode = CopyMode::Eager;

    /// `tips[b]`, or the empty (root) frontier for an unknown branch.
    fn target(&self, this: &MultiHeadDoc<Self>, b: &BranchId) -> LoroResult<Frontiers> {
        Ok(self
            .attribution(this)
            .lock()
            .tips
            .get(b)
            .cloned()
            .unwrap_or_default())
    }

    /// The committing head sat at `tips[b]`, so its new single tip is the
    /// branch's frontier.
    fn after_commit(&self, this: &MultiHeadDoc<Self>, b: &BranchId, committed: IdSpan) {
        let mut attr = self.attribution(this).lock();
        attr.ensure_run(committed.peer, committed.counter.start, b);
        attr.tips
            .insert(b.clone(), Frontiers::from_id(committed.id_last()));
    }

    /// The causal fold. Sorts imported changes by `(lamport, peer, counter)`
    /// (topological: a change's lamport exceeds its deps'), so every dependency
    /// is in `runs` before lookup (Loro holds changes in `pending` until deps
    /// arrive). Returns the branches whose `tips` moved, or an error if any op
    /// cannot be attributed.
    fn after_import(
        &self,
        this: &MultiHeadDoc<Self>,
        st: &ImportStatus,
    ) -> LoroResult<Vec<BranchId>> {
        let ol = this.oplog.lock(); // OpLog before Attribution
        let mut batch: Vec<Imported> = Vec::new();
        for (peer, (start, end)) in st.success.iter() {
            for ch in ol.iter_changes(IdSpan::new(*peer, *start, *end)) {
                let mut markers = SmallVec::new();
                for op in ch.ops().iter() {
                    if let Some(b) = marker_branch_of(&ol, op) {
                        markers.push((op.counter, b));
                    }
                }
                batch.push(Imported {
                    lamport: ch.lamport(),
                    peer: ch.peer(),
                    start: ch.id().counter,
                    last: ch.id_last().counter,
                    deps: ch.deps().clone(),
                    markers,
                });
            }
        }
        batch.sort_by_key(|c| (c.lamport, c.peer, c.start));

        let mut touched: FxHashSet<BranchId> = FxHashSet::default();
        let mut attr = self.attribution(this).lock();
        for c in batch {
            // Segments `(first, last, branch)`: the prefix before the first
            // marker is attributed by deps; each marker covers up to the next.
            let mut segments: SmallVec<[(Counter, Counter, BranchId); 2]> = SmallVec::new();
            let first_marker = c.markers.first().map(|(k, _)| *k);
            if first_marker != Some(c.start) {
                let seg_last = first_marker.map_or(c.last, |k| k - 1);
                segments.push((c.start, seg_last, attr.branch_of_deps(&c.deps)?));
            }
            for (i, (k, b)) in c.markers.iter().enumerate() {
                let seg_last = c.markers.get(i + 1).map_or(c.last, |(n, _)| *n - 1);
                segments.push((*k, seg_last, b.clone()));
            }
            for (s, l, b) in segments {
                attr.ensure_run(c.peer, s, &b);
                let mut f = attr.tips.get(&b).cloned().unwrap_or_default();
                f.push(ID::new(c.peer, l));
                let nf = shrink_frontiers(&f, &ol.dag).map_err(|id| {
                    invalid_attribution(format!(
                        "attributed id {id} of branch {b} is not in the DAG"
                    ))
                })?;
                if attr.tips.get(&b) != Some(&nf) {
                    attr.tips.insert(b.clone(), nf);
                    touched.insert(b);
                }
            }
        }
        drop(attr);
        drop(ol);
        Ok(touched.into_iter().collect())
    }
}

/// A content doc's resolution policy: delegates to the repo's index for where
/// branch `b` is for this doc, copying lazily on divergence.
#[allow(missing_debug_implementations)]
pub struct Delegated {
    pub(super) id: DocId,
    pub(super) index: Arc<IndexDoc>,
}

impl Delegated {
    pub fn new(id: DocId, index: Arc<IndexDoc>) -> Self {
        Delegated { id, index }
    }
}

impl HeadPolicy for Delegated {
    const COPY: CopyMode = CopyMode::OnDivergence;

    /// The index's recorded ids for `b`, minus ids this doc does not yet hold
    /// ("skipped until held"), shrunk. The index and content locks never nest.
    fn target(&self, this: &MultiHeadDoc<Self>, b: &BranchId) -> LoroResult<Frontiers> {
        let raw = self.index.recorded_ids(b, &self.id)?;
        let ol = this.oplog.lock();
        let held: Vec<ID> = raw
            .into_iter()
            .filter(|id| ol.vv().get_last(id.peer).is_some_and(|c| c >= id.counter))
            .collect();
        shrink_frontiers(&Frontiers::from(held), &ol.dag).map_err(LoroError::FrontiersNotFound)
    }

    /// Publish this commit's new tip into the index: `docs[doc].heads[peer] = c`.
    fn after_commit(&self, _this: &MultiHeadDoc<Self>, b: &BranchId, committed: IdSpan) {
        let last = committed.id_last();
        let _ = self.index.record_head(b, &self.id, last.peer, last.counter);
    }

    /// Re-resolve every branch the index knows: ids that just became held are
    /// picked up; still-unheld ids are dropped by `target` until next time.
    fn after_import(
        &self,
        _this: &MultiHeadDoc<Self>,
        _status: &ImportStatus,
    ) -> LoroResult<Vec<BranchId>> {
        Ok(self.index.branches())
    }
}

/// Test-only policy for exercising the base without the real policies.
/// `target` is whatever the test last recorded (default: empty frontier);
/// `after_commit` moves it to the head's new tip.
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

    fn after_commit(&self, _this: &MultiHeadDoc<Self>, b: &BranchId, committed: IdSpan) {
        self.targets
            .lock()
            .unwrap()
            .insert(b.clone(), Frontiers::from_id(committed.id_last()));
    }

    fn after_import(
        &self,
        _this: &MultiHeadDoc<Self>,
        _status: &ImportStatus,
    ) -> LoroResult<Vec<BranchId>> {
        Ok(Vec::new())
    }
}
