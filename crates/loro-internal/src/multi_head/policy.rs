use std::sync::{Arc, OnceLock};

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::encoding::ImportStatus;
use crate::lock::{LockKind, LoroMutex};
use crate::oplog::OpLog;
use crate::version::{shrink_frontiers, Frontiers};
use loro_common::{ContainerID, IdSpan, InternalString, LoroError, LoroResult, PeerID, ID};

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

    /// Publish the new tip after `b`'s head commits `last`. May be a no-op (the
    /// index's tip IS its record). Runs OUTSIDE the registry lock.
    fn after_commit(&self, this: &MultiHeadDoc<Self>, b: &BranchId, last: ID);

    /// Given the spans that just landed on import, return the branches whose
    /// binding should be re-resolved.
    fn after_import(&self, this: &MultiHeadDoc<Self>, status: &ImportStatus) -> Vec<BranchId>;
}

/// Root-container name prefix for a branch's lineage list (`lineage:<b>`).
const LINEAGE_PREFIX: &str = "lineage:";

pub(super) fn lineage_name(b: &BranchId) -> String {
    format!("{LINEAGE_PREFIX}{b}")
}

/// If `idx` names a `lineage:<b>` root container, return `b`. Pure op-log
/// discovery: the branch is encoded in the (name-addressable) container id, so
/// `after_import` never needs a materialized state to identify it.
fn lineage_branch_of(ol: &OpLog, idx: crate::container::idx::ContainerIdx) -> Option<BranchId> {
    match ol.arena.idx_to_id(idx)? {
        ContainerID::Root { name, .. } => name
            .as_str()
            .strip_prefix(LINEAGE_PREFIX)
            .map(InternalString::from),
        _ => None,
    }
}

/// The index's resolution policy: its OWN per-branch lineage IS the root of
/// truth for where each branch is, so it consults no other index -- this breaks
/// the `BranchingDoc`-depends-on-index circularity. Eager copy at branch
/// creation means every index head is born `refs == 1` (never shared), so the
/// sink guard never fires on an index head.
///
/// `lineage` maps a branch to the peers of its index heads. It is rebuilt from
/// the index's own op log: each index head writes under exactly one branch with
/// a unique peer, so "the index ops of branch `b`" is exactly "the ops of the
/// peers in `lineage:<b>`", and `b`'s index frontier is the join of those peers'
/// latest ids.
#[allow(missing_debug_implementations)]
pub struct SelfRooted {
    // Created lazily in the doc's lock group (`LockKind::Lineage`, the leaf
    // acquired only after `OpLog`) on first use, since the group exists only
    // once `MultiHeadDoc::new` has run.
    lineage: OnceLock<LoroMutex<FxHashMap<BranchId, SmallVec<[PeerID; 2]>>>>,
}

impl Default for SelfRooted {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfRooted {
    pub fn new() -> Self {
        SelfRooted {
            lineage: OnceLock::new(),
        }
    }

    pub(super) fn lineage(
        &self,
        this: &MultiHeadDoc<SelfRooted>,
    ) -> &LoroMutex<FxHashMap<BranchId, SmallVec<[PeerID; 2]>>> {
        self.lineage.get_or_init(|| {
            this.lock_group
                .new_lock(FxHashMap::default(), LockKind::Lineage)
        })
    }
}

impl HeadPolicy for SelfRooted {
    const COPY: CopyMode = CopyMode::Eager;

    /// Branch `b`'s index frontier = the join of the latest ids of the peers in
    /// `lineage:<b>`, shrunk against the op-log DAG. The causal past does the
    /// rest: a peer's ops depend on the tip its head forked from, so this brings
    /// the inherited parent record and `b`'s own writes, never a sibling's later
    /// writes (which no `b` peer depends on).
    fn target(&self, this: &MultiHeadDoc<Self>, b: &BranchId) -> LoroResult<Frontiers> {
        let ol = this.oplog.lock(); // OpLog before Lineage
        let peers: SmallVec<[PeerID; 2]> = self
            .lineage(this)
            .lock()
            .get(b)
            .cloned()
            .unwrap_or_default();
        let ids: Vec<ID> = peers
            .iter()
            .filter_map(|p| ol.vv().get_last(*p).map(|c| ID::new(*p, c)))
            .collect();
        shrink_frontiers(&Frontiers::from(ids), &ol.dag).map_err(LoroError::FrontiersNotFound)
    }

    /// The index tip IS the record; nothing extra to publish.
    fn after_commit(&self, _this: &MultiHeadDoc<Self>, _b: &BranchId, _last: ID) {}

    /// Walk the just-imported spans: an op in a `lineage:<b>` container names a
    /// (possibly remote) peer of `b`; any op by a known lineage peer marks its
    /// branch touched. Returns the branches whose index head should be rebound.
    fn after_import(&self, this: &MultiHeadDoc<Self>, st: &ImportStatus) -> Vec<BranchId> {
        // Collect under the OpLog lock, then fold into the lineage map (leaf
        // lock, acquired alone) -- never nesting the two here.
        let mut lineage_ops: Vec<(BranchId, PeerID)> = Vec::new();
        let mut change_peers: Vec<PeerID> = Vec::new();
        {
            let ol = this.oplog.lock();
            for (peer, (start, end)) in st.success.iter() {
                for ch in ol.iter_changes(IdSpan::new(*peer, *start, *end)) {
                    let cp = ch.peer();
                    change_peers.push(cp);
                    for op in ch.ops().iter() {
                        if let Some(b) = lineage_branch_of(&ol, op.container) {
                            lineage_ops.push((b, cp));
                        }
                    }
                }
            }
        }
        let mut touched: FxHashSet<BranchId> = FxHashSet::default();
        let mut lineage = self.lineage(this).lock();
        for (b, p) in lineage_ops {
            let entry = lineage.entry(b.clone()).or_default();
            if !entry.contains(&p) {
                entry.push(p);
            }
            touched.insert(b);
        }
        for cp in change_peers {
            for (b, peers) in lineage.iter() {
                if peers.contains(&cp) {
                    touched.insert(b.clone());
                }
            }
        }
        drop(lineage);
        touched.into_iter().collect()
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
    fn after_commit(&self, _this: &MultiHeadDoc<Self>, b: &BranchId, last: ID) {
        let _ = self.index.record_head(b, &self.id, last.peer, last.counter);
    }

    /// After a content import, re-resolve every branch the index knows: a branch
    /// whose recorded ids for this doc just became held advances to them (the
    /// ingest); ids still unheld are dropped by `target` and picked up next time.
    fn after_import(&self, _this: &MultiHeadDoc<Self>, _status: &ImportStatus) -> Vec<BranchId> {
        self.index.branches()
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

    fn after_commit(&self, _this: &MultiHeadDoc<Self>, b: &BranchId, last: ID) {
        self.targets
            .lock()
            .unwrap()
            .insert(b.clone(), Frontiers::from_id(last));
    }

    fn after_import(&self, _this: &MultiHeadDoc<Self>, _status: &ImportStatus) -> Vec<BranchId> {
        Vec::new()
    }
}
