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
///
/// `target` says where a branch should be, `after_commit` publishes a new tip,
/// and `after_import` says which branches to rebind after history lands. The
/// base calls these; it never branches on which policy it holds.
pub trait HeadPolicy: Send + Sync + 'static + Sized {
    /// Copy discipline for this variant.
    const COPY: CopyMode;

    /// The frontier branch `b` should currently resolve to. Called while the
    /// registry lock is held, so it MUST NOT lock the registry.
    fn target(&self, this: &MultiHeadDoc<Self>, b: &BranchId) -> LoroResult<Frontiers>;

    /// Publish the new tip after `b`'s head commits the ops in `committed` (the
    /// change's full id span; its last id is the head's new single tip). May be
    /// a no-op. Runs OUTSIDE the registry lock.
    fn after_commit(&self, this: &MultiHeadDoc<Self>, b: &BranchId, committed: IdSpan);

    /// Given the spans that just landed on import, return the branches whose
    /// binding should be re-resolved. Errs if the import is malformed (an op the
    /// index policy cannot attribute to any branch); the caller propagates it.
    fn after_import(
        &self,
        this: &MultiHeadDoc<Self>,
        status: &ImportStatus,
    ) -> LoroResult<Vec<BranchId>>;
}

/// Root map that carries every branch's creation MARKER as a plain map value:
/// `branch.name = "<b>"` (`Root{name: "branch"}`, key `"name"`).
///
/// The first op of a fresh index peer sets this key to the branch it was created
/// for. Because the name travels as a `LoroValue::String` in the op log (a
/// `MapSet` op, op-log-resident and readable by `after_import` with no
/// materialized state), the whole index needs ONE marker root container rather
/// than one per branch, and a branch name is any string (no container-id charset
/// constraint).
pub(super) const BRANCH_ROOT: &str = "branch";
pub(super) const BRANCH_NAME_KEY: &str = "name";

/// Key on the `branch` root map for a MERGE marker: `branch.merge = "<seq>:<b>"`
/// where `<seq>` is a monotonic integer.
///
/// A repo-wide merge advances `into`'s index head to the join and needs ONE op
/// naming `into`. It cannot reuse the `name` key: at the join `branch.name` may
/// ALREADY equal `into` (the winner of the concurrent creation-marker LWW, or
/// `into`'s own set causally dominating the source's), and a same-value map set
/// emits NO op -- so `after_commit` never fires and the whole merge is silently
/// dropped, on this peer AND on every peer that would fold the op stream. The
/// `<seq>` prefix (read-then-increment of the current merge value) guarantees the
/// value DIFFERS from whatever the last merge into `into` wrote, so the op always
/// emits. The branch name is everything after the FIRST `:`, so a name that
/// itself contains `:` is recovered intact (the seq is a decimal integer, no
/// `:`). It stays a KEY on the existing `branch` map, so the root-container count
/// is unchanged (still 2).
pub(super) const BRANCH_MERGE_KEY: &str = "merge";

/// The branch a marker op on the `branch` root map names, or `None` for a
/// non-marker op. Pure op-log discovery: attribution never needs a materialized
/// state to identify the branch. Two marker kinds:
/// - `branch.name = "<b>"` (creation): the value IS the branch name;
/// - `branch.merge = "<seq>:<b>"` (merge): the name is everything after the
///   first `:` (the `<seq>` prefix keeps the map set from being a same-value
///   no-op; see [`BRANCH_MERGE_KEY`]).
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

/// Build the loud error for an imported op the fold cannot attribute to a
/// branch. This is UNREACHABLE in well-formed operation: every index op
/// descends from genesis, a normal record resolves to exactly one nearest
/// marker, and a merge is itself a marker. An unattributable op is therefore
/// malformed input, and the fold refuses it (advancing no branch) and returns
/// this error, the way Loro rejects an invalid import -- NEVER silently
/// attributing it to some default branch (that is the S0 silent-data-loss
/// class).
fn invalid_attribution(msg: impl std::fmt::Display) -> LoroError {
    tracing::error!("index attribution: invalid import: {msg}");
    LoroError::Unknown(format!("index attribution: invalid import: {msg}").into_boxed_str())
}

/// The index's derived branch projection (the maintainer's `active_branches`),
/// folded from the op log by causal attribution: an op belongs to the branch
/// named by the nearest marker in its causal past.
#[derive(Debug, Default)]
pub struct Attribution {
    /// Branch -> its index frontier. `target(b)` is exactly `tips[b]`.
    pub tips: FxHashMap<BranchId, Frontiers>,
    /// Per-peer attribution runs, sorted by start counter: the op `(peer, c)`
    /// belongs to the run with the greatest start `<= c`. A fresh index peer
    /// has one run (its creation marker at counter 0); a peer whose head is
    /// later rebound, or a peer that writes a marker mid-history, gains more.
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

    /// Make the run covering `(peer, start)` name `b`: a no-op if it already
    /// does, else a new run starting at `start` (or a re-labelling of a run
    /// that starts exactly there, for an idempotent re-fold of a stored change
    /// that merged with an earlier one).
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

/// The index's resolution policy: its OWN op log is the root of truth for where
/// each branch is, so it consults no other index -- this breaks the
/// `BranchingDoc`-depends-on-index circularity. Eager copy at branch creation
/// means every index head is born `refs == 1` (never shared), so the sink guard
/// never fires on an index head.
///
/// Attribution is CAUSAL, not by peer identity: a change belongs to the branch
/// named by its own marker if it carries one, else to the branch of its
/// dependencies (which must agree, or the import is rejected as malformed). A
/// head's state is always at `tips[b]` when it commits (`resolve` guarantees it),
/// so
/// every local change depends on `b`'s tips and inherits `b` with no
/// declaration -- which is why a head minted by the resolve materialize arm
/// needs no marker of its own: its first change's deps already attribute it.
///
/// Consistency of the derived `tips`: every mutation of `(tips, runs)` happens
/// under the `Attribution` leaf lock as one critical section (a whole fold, or
/// a whole `after_commit`), so a reader holding that lock sees all of a fold or
/// none of it. `after_commit` publishes AFTER the commit returns; a session is
/// single-writer, so its own next `resolve` runs after that publish.
#[allow(missing_debug_implementations)]
pub struct SelfRooted {
    // Created lazily in the doc's lock group (`LockKind::Attribution`, the leaf
    // acquired alone or after `OpLog`) on first use, since the group exists
    // only once `MultiHeadDoc::new` has run.
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

/// One imported change, as the fold sees it: its position in the DAG plus the
/// counters of its marker ops.
struct Imported {
    lamport: Lamport,
    peer: PeerID,
    start: Counter,
    last: Counter,
    deps: Frontiers,
    /// `(op counter, branch)` per marker op, in counter order. A stored change
    /// may merge several consecutive commits of one peer, so a marker can sit
    /// past the first op; each marker starts a new attribution segment.
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

    /// The head that committed sat at `tips[b]`, so its new single tip IS the
    /// branch's frontier; its peer's run covers the committed counters as `b`.
    fn after_commit(&self, this: &MultiHeadDoc<Self>, b: &BranchId, committed: IdSpan) {
        let mut attr = self.attribution(this).lock();
        attr.ensure_run(committed.peer, committed.counter.start, b);
        attr.tips
            .insert(b.clone(), Frontiers::from_id(committed.id_last()));
    }

    /// The causal fold. Collect the just-imported changes, sort them by
    /// `(lamport, peer, counter)` (a topological order: a change's lamport
    /// exceeds every dependency's), then attribute each in that order, so every
    /// dependency of a successful change is in `runs` before it is looked up
    /// (Loro holds a change back in `pending` until its deps are present).
    /// Returns the branches whose `tips` moved, or an error if any op cannot be
    /// attributed (malformed import; no branch advances for it).
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
            // Segments `(first counter, last counter, branch)`: the prefix
            // before the first marker is attributed by deps; each marker
            // attributes the ops from itself up to the next marker.
            let mut segments: SmallVec<[(Counter, Counter, BranchId); 2]> = SmallVec::new();
            let first_marker = c.markers.first().map(|(k, _)| *k);
            if first_marker != Some(c.start) {
                let seg_last = first_marker.map_or(c.last, |k| k - 1);
                // A non-marker prefix inherits its deps' branch; deps that are
                // unknown or disagree mean a malformed op -> reject the import.
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

/// A content doc's resolution policy: DELEGATE to the repo's index for "where is
/// branch `b` for this doc", copying lazily on divergence. The index is the
/// source of truth; a content head materializes at the recorded frontier.
///
/// NOTE(claude-opus-4-8/branchingdocrepo-multiheaddoc): the proposal's
/// `frontier_cache` (a per-branch mirror of the index frontier refreshed by the
/// index's history subscription) is a HOT-PATH OPTIMIZATION and is deferred:
/// `target` reads the index directly each call (the source of truth), which is
/// correct, just not cached. The cache is Phase-3 follow-up scope.
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

    /// Where branch `b` is for this doc: the index's recorded ids reduced against
    /// THIS doc's history -- drop ids the content history does not yet hold
    /// ("skipped until held"), then shrink. Reads the index first (its locks
    /// released), then the content DAG; the two docs' locks never nest.
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

    /// After a content import, re-resolve every branch the index knows: a branch
    /// whose recorded ids for this doc just became held advances to them (the
    /// ingest); ids still unheld are dropped by `target` and picked up next time.
    fn after_import(
        &self,
        _this: &MultiHeadDoc<Self>,
        _status: &ImportStatus,
    ) -> LoroResult<Vec<BranchId>> {
        Ok(self.index.branches())
    }
}

/// A test-only policy that exercises the base's registry/copy/guard mechanics
/// without the real index or content policies. `target` is whatever the test
/// last recorded (default: the empty root frontier); `after_commit` advances a
/// branch's recorded target to its head's new tip so resolution stays stable.
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
