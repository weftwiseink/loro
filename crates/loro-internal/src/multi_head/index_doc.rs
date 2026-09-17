use rustc_hash::FxHashMap;

use std::cmp::Ordering;

use crate::version::{shrink_frontiers, Frontiers};
use loro_common::{Counter, InternalString, Lamport, LoroError, LoroResult, LoroValue, PeerID, ID};

use super::policy::{BRANCH_MERGE_KEY, BRANCH_NAME_KEY, BRANCH_ROOT};
use super::ROOT_HEAD_ID;
use super::*;

/// The repo's index doc. Its heads hold only frontiers: TWO root maps total, a
/// `docs: LoroMap<DocId, LoroMap<"heads", LoroMap<PeerID, Counter>>>` and a
/// `branch` map whose `name` key set (`branch.name = "<b>"`) is a branch's
/// creation MARKER (see `SelfRooted`). No weft filesystem metadata (that is a
/// TS-side content doc). The only types that appear are `ID`/`Frontiers`/
/// `DocId`/`BranchId`.
pub type IndexDoc = MultiHeadDoc<SelfRooted>;

impl MultiHeadDoc<SelfRooted> {
    /// Establish the first (genesis) branch, bound to the pinned root head, by
    /// writing its creation marker as the root's first op: `branch.name =
    /// "<genesis>"`. The commit hook's `after_commit` sets `tips[genesis]` to
    /// that op. Must be called once before other branches are created.
    pub fn init_genesis(&self, genesis: &BranchId) -> LoroResult<()> {
        let root = self.head_doc(ROOT_HEAD_ID).expect("root head exists");
        self.with_reg(|this, reg| this.rebind(reg, genesis, ROOT_HEAD_ID));
        root.get_map(BRANCH_ROOT)
            .insert(BRANCH_NAME_KEY, genesis.as_str())?;
        root.commit_then_renew();
        Ok(())
    }

    /// Create `new` from `from` by EAGER copy: fork `from`'s index head (a state
    /// of a few map entries), then write the copy's first op -- the creation
    /// marker, `branch.name = "<new>"` (overwriting the copied parent's name) --
    /// directly on the copy. That op depends on `tips[from]` and names `new`, so
    /// the commit hook's `after_commit` sets `tips[new]` to it. Every index head
    /// is thus born `refs == 1`.
    ///
    /// The name travels as a plain map value (a `LoroValue::String` in the op
    /// log), so `new` may be any string -- no container-id charset constraint.
    ///
    /// Errors if `new` already exists (mirroring `create_branch_at`): re-forking
    /// a live name would otherwise mint a second creation marker for it, and
    /// `tips[new]` would become the join of two unrelated lineages.
    pub fn create_index_branch(&self, new: &BranchId, from: &BranchId) -> LoroResult<()> {
        if self.branches().contains(new) {
            return Err(LoroError::ArgErr(
                format!("cannot create_index_branch: branch '{new}' already exists")
                    .into_boxed_str(),
            ));
        }
        let (from_head, _) = self.resolve(from, Intent::Read)?;
        let copy_doc = self.with_reg(|this, reg| {
            let c = this.copy_head(reg, from_head);
            this.rebind(reg, new, c);
            reg.heads[&c].doc.clone()
        });
        copy_doc
            .get_map(BRANCH_ROOT)
            .insert(BRANCH_NAME_KEY, new.as_str())?;
        copy_doc.commit_then_renew();
        Ok(())
    }

    /// Merge branch `from` into `into` REPO-WIDE with a SINGLE index marker op.
    ///
    /// Advance `into`'s index head to `join(tips[into], tips[from])`, whose CRDT
    /// state IS the per-key union of every doc's `docs[D].heads` map (a mergeable
    /// per-peer VV, so a concurrent same-branch writer on either side survives --
    /// Spike S5), then commit ONE `branch.name = "<into>"` marker on it. The
    /// per-doc `record_frontier` loop the content-level `BranchingDoc::merge`
    /// walks is not needed here: the union of both branches' recorded frontiers
    /// falls out of the checkout for free.
    ///
    /// The marker's dependencies ARE the join (a frontier spanning both
    /// branches), so on a remote import the causal fold attributes it to `into`
    /// (a marker needs no dep agreement) and `tips[into]` becomes `[marker]`.
    /// `from` is untouched: a merge never moves the source's head or tips, so a
    /// source merged into two targets is simply two marker ops on two heads.
    ///
    /// Returns `AlreadyContained` (nothing written) when `into` already contains
    /// `from`, else `FastForward` / `Merged` (one marker written either way).
    pub fn merge(&self, into: &BranchId, from: &BranchId) -> LoroResult<MergeOutcome> {
        let ti = self.policy().target(self, into)?;
        let tf = self.policy().target(self, from)?;
        let (outcome, join) = {
            let ol = self.oplog.lock();
            let outcome = match ol.dag.cmp_frontiers(&ti, &tf).map_err(LoroError::from)? {
                Some(Ordering::Equal) | Some(Ordering::Greater) => {
                    return Ok(MergeOutcome::AlreadyContained)
                }
                Some(Ordering::Less) => MergeOutcome::FastForward,
                None => MergeOutcome::Merged,
            };
            // The join: shrink(union of both tips' ids). Every id of `from` is in
            // the union, so no op is dropped.
            let mut u = ti.clone();
            for id in tf.iter() {
                u.push(id);
            }
            (
                outcome,
                shrink_frontiers(&u, &ol.dag).map_err(LoroError::FrontiersNotFound)?,
            )
        };
        // Bind `into` (at tips[into], refs == 1 eager), advance its head to the
        // join, and commit the single merge marker. The commit hook's
        // `after_commit` sets `tips[into]` to the marker.
        //
        // The marker goes on the `branch.merge` key as "<seq>:<into>", NOT on
        // `branch.name`: at the join `branch.name` may already equal `into`, and
        // a same-value map set emits NO op -- which would leave the merge with no
        // op to carry it (dropping it locally and on every peer). The `<seq>`
        // prefix, read-then-incremented from the current merge value, guarantees
        // the set writes a fresh value and always emits. See `BRANCH_MERGE_KEY`.
        self.resolve(into, Intent::Read)?;
        let doc = self.advance_bound_writable(into, &join)?;
        let branch_map = doc.get_map(BRANCH_ROOT);
        let next_seq = branch_map
            .get(BRANCH_MERGE_KEY)
            .and_then(|v| v.into_string().ok())
            .and_then(|s| s.split_once(':').and_then(|(seq, _)| seq.parse::<i64>().ok()))
            .map_or(0, |n| n + 1);
        branch_map.insert(BRANCH_MERGE_KEY, format!("{next_seq}:{into}"))?;
        doc.commit_then_renew();
        Ok(outcome)
    }

    /// Record `docs[doc].heads[peer] = counter` on `b`'s index head: the
    /// per-writer head record the frontier reduction reads back. One local op.
    pub fn record_head(
        &self,
        b: &BranchId,
        doc: &DocId,
        peer: PeerID,
        counter: Counter,
    ) -> LoroResult<()> {
        self.write(b, |d| {
            // Mergeable (map-key) child containers so two sessions writing the
            // same doc/heads path converge to one container across import.
            let docs = d.get_map("docs");
            let per_doc = docs.ensure_mergeable_map(doc.as_str())?;
            let heads = per_doc.ensure_mergeable_map("heads")?;
            heads.insert(&peer.to_string(), counter as i64)
        })?
    }

    /// ADDITIVELY record every id of frontier `f` into `docs[doc].heads` on `b`'s
    /// index head: `heads[peer] = counter` per id, MERGING into existing entries
    /// (a per-writer max in effect). This is correct for forward accumulation --
    /// `advance` / `merge` / ingest move a LIVE branch forward, and each keeps the
    /// other peers' recorded positions. It is NOT correct for setting a branch to
    /// an exact frontier that must EXCLUDE other peers -- see `set_frontier`.
    pub fn record_frontier(&self, b: &BranchId, doc: &DocId, f: &Frontiers) -> LoroResult<()> {
        self.write(b, |d| {
            let docs = d.get_map("docs");
            let per_doc = docs.ensure_mergeable_map(doc.as_str())?;
            let heads = per_doc.ensure_mergeable_map("heads")?;
            for id in f.iter() {
                heads.insert(&id.peer.to_string(), id.counter as i64)?;
            }
            Ok(())
        })?
    }

    /// Set `docs[doc].heads` on `b`'s index head to EXACTLY frontier `f`: clear
    /// every existing entry first, then record `f`'s ids. Unlike the additive
    /// `record_frontier`, this OVERRIDES -- the recorded frontier becomes `f` and
    /// nothing else. `create_branch_at` needs this: eager `create_index_branch`
    /// seeds the fork with the parent's (possibly multi-peer) recorded frontier,
    /// and forking at a single-peer historical `at` must not leave stale
    /// other-peer entries (which would silently pull those peers' ops into the
    /// fork).
    pub fn set_frontier(&self, b: &BranchId, doc: &DocId, f: &Frontiers) -> LoroResult<()> {
        self.write(b, |d| {
            let docs = d.get_map("docs");
            let per_doc = docs.ensure_mergeable_map(doc.as_str())?;
            let heads = per_doc.ensure_mergeable_map("heads")?;
            let existing: Vec<InternalString> = heads.keys().collect();
            for k in existing {
                heads.delete(k.as_str())?;
            }
            for id in f.iter() {
                heads.insert(&id.peer.to_string(), id.counter as i64)?;
            }
            Ok(())
        })?
    }

    /// The raw `(peer, counter)` ids recorded in `docs[doc].heads` on `b`'s index
    /// head (before any reduction against a content doc's history). Empty if the
    /// branch or doc is unknown. `Delegated::target` reduces these against the
    /// content doc's DAG.
    pub fn recorded_ids(&self, b: &BranchId, doc: &DocId) -> LoroResult<Vec<ID>> {
        self.read(b, |d| {
            let mut ids = Vec::new();
            let heads = d
                .get_deep_value()
                .as_map()
                .and_then(|root| root.get("docs").cloned())
                .and_then(|v| v.into_map().ok())
                .and_then(|docs| docs.get(doc.as_str()).cloned())
                .and_then(|v| v.into_map().ok())
                .and_then(|per| per.get("heads").cloned())
                .and_then(|v| v.into_map().ok());
            if let Some(heads) = heads {
                for (peer_str, counter) in heads.iter() {
                    if let (Ok(peer), Some(c)) = (peer_str.parse::<PeerID>(), counter.as_i64()) {
                        ids.push(ID::new(peer, *c as Counter));
                    }
                }
            }
            ids
        })
    }

    /// The branches this session knows: the keys of the derived `tips`.
    pub fn branches(&self) -> Vec<BranchId> {
        self.policy()
            .attribution(self)
            .lock()
            .tips
            .keys()
            .cloned()
            .collect()
    }

    /// Every branch's index frontier (`{b: target(b)}`), the derived read-only
    /// `active_branches` projection.
    pub fn tips(&self) -> FxHashMap<BranchId, Frontiers> {
        self.policy().attribution(self).lock().tips.clone()
    }

    /// The index frontier branch `b` was forked from: the DEPENDENCIES of `b`'s
    /// creation marker (the causally-earliest op attributed to `b`).
    ///
    /// Empty for the genesis branch (its creation marker is the root op, which
    /// has no deps) and for an unknown branch. For a branch created from a
    /// parent, this is the parent's index tips at the moment `b` was forked --
    /// the cross-doc fork point, which lives ONLY in the index (a content doc's
    /// own log cannot answer where `b` forked for a doc nobody has touched on `b`
    /// yet).
    pub fn fork_point(&self, b: &BranchId) -> Frontiers {
        let starts: Vec<ID> = {
            let attr = self.policy().attribution(self).lock();
            attr.runs
                .iter()
                .flat_map(|(peer, runs)| {
                    runs.iter()
                        .filter(|(_, br)| br == b)
                        .map(move |(start, _)| ID::new(*peer, *start))
                })
                .collect()
        };
        let ol = self.oplog.lock();
        starts
            .into_iter()
            .min_by_key(|id| {
                ol.get_change_at(*id)
                    .map(|c| c.lamport())
                    .unwrap_or(Lamport::MAX)
            })
            .and_then(|id| ol.get_deps_of(id))
            .unwrap_or_default()
    }

    /// The `docs` map (each `DocId` -> its `heads` value) as it stood at frontier
    /// `f`, materialized on a throwaway scratch head. The comparison substrate
    /// for the fork-point diffs below.
    fn docs_map_at(&self, f: &Frontiers) -> LoroResult<FxHashMap<DocId, LoroValue>> {
        self.read_at(f, |d| {
            d.get_deep_value()
                .as_map()
                .and_then(|root| root.get("docs").cloned())
                .and_then(|v| v.into_map().ok())
                .map(|docs| {
                    docs.iter()
                        .map(|(k, v)| (DocId::from(k.as_str()), v.clone()))
                        .collect()
                })
                .unwrap_or_default()
        })
    }

    /// The docs branch `b` ITSELF modified: those whose recorded frontier at
    /// `tips[b]` DIFFERS from their state at `fork_point(b)` (a fork-point diff).
    /// Docs inherited unchanged from `b`'s parent are excluded (their value is
    /// identical at both frontiers), unlike the raw `docs` key set at `tips[b]`
    /// (which is nearly every doc, since a fork eager-copies the parent's record).
    ///
    /// This is computed ON DEMAND, not from an incremental per-branch stack: the
    /// incremental "copy the parent's set at fork" step is not correct under a
    /// concurrent fork (a parent op concurrent with the fork, folded first by
    /// lamport order, would leak into the child's inherited set), so it
    /// degenerates to this same fork-point recompute. The recompute is also
    /// exactly what a MERGE needs (a merged branch's own set is recomputed from
    /// its branchpoint), so one fork-point diff is uniformly correct for forks,
    /// merges, and concurrent forks. Empty for an unknown branch.
    pub fn docs_modified_on_branch(&self, b: &BranchId) -> LoroResult<Vec<DocId>> {
        let tip = self.policy().target(self, b)?;
        let fork = self.fork_point(b);
        let at_tip = self.docs_map_at(&tip)?;
        let at_fork = self.docs_map_at(&fork)?;
        let mut out: Vec<DocId> = at_tip
            .into_iter()
            .filter(|(dc, v)| at_fork.get(dc) != Some(v))
            .map(|(dc, _)| dc)
            .collect();
        out.sort();
        Ok(out)
    }

    /// The docs modified anywhere on `b`'s lineage since it diverged from the
    /// genesis branch (the union of the per-branch stack "changed since main").
    /// Rust-only accessor for future logic; no wasm surface yet. Empty for the
    /// genesis branch.
    ///
    /// Computed as a fork-point diff from the frontier where `b`'s lineage left
    /// genesis (found by walking parent fork-points up to genesis) to `tips[b]`.
    pub fn docs_changed_since_main(&self, b: &BranchId) -> LoroResult<Vec<DocId>> {
        let departure = self.departure_from_main(b);
        if departure == Frontiers::default() {
            // `b` is the genesis branch (never diverged from main): the baseline
            // has nothing "changed since main".
            return Ok(Vec::new());
        }
        let tip = self.policy().target(self, b)?;
        let at_tip = self.docs_map_at(&tip)?;
        let at_dep = self.docs_map_at(&departure)?;
        let mut out: Vec<DocId> = at_tip
            .into_iter()
            .filter(|(dc, v)| at_dep.get(dc) != Some(v))
            .map(|(dc, _)| dc)
            .collect();
        out.sort();
        Ok(out)
    }

    /// The frontier at which `b`'s lineage left the genesis branch: walk parent
    /// fork-points (`fork_point` -> the branch it attributes to -> its
    /// `fork_point`) until the parent is genesis (an empty fork point). The
    /// returned frontier is the fork point of the topmost non-genesis ancestor.
    /// Empty for the genesis branch itself.
    fn departure_from_main(&self, b: &BranchId) -> Frontiers {
        let mut fork = self.fork_point(b);
        loop {
            if fork == Frontiers::default() {
                return Frontiers::default(); // `b` (or the current ancestor) is genesis
            }
            let parent = {
                let attr = self.policy().attribution(self).lock();
                fork.iter().next().and_then(|id| attr.branch_at(id).cloned())
            };
            let Some(parent) = parent else {
                return fork;
            };
            let parent_fork = self.fork_point(&parent);
            if parent_fork == Frontiers::default() {
                return fork; // parent is genesis; `b`'s lineage left main here
            }
            fork = parent_fork;
        }
    }

    /// Remove a branch from the index: unbind its index head (retiring it) and
    /// drop its `tips` entry, so `branches()` no longer lists it and no
    /// `bound`/`by_tip` entry dangles. Its attribution runs are kept, so a
    /// later-imported change that depends on its history still attributes (and
    /// re-lists the branch, exactly as a re-imported marker did before). The
    /// durable cross-peer "discard" is the wrapper's lifecycle log; this is the
    /// local registry cleanup.
    pub fn delete_index_branch(&self, name: &BranchId) {
        self.unbind(name);
        self.policy().attribution(self).lock().tips.remove(name);
    }

    /// The registry HeadId of branch `b`'s (self-rooted) index head, resolving
    /// it from lineage if not yet bound. (`Head` itself is registry-internal, so
    /// this hands back the id rather than the proposal's `Arc<Head>`.)
    ///
    /// CAVEATS: this calls `resolve` (a Read), so it has the same side effects
    /// (it may bind/advance/materialize `b`'s head). The returned `HeadId` is
    /// SESSION-LOCAL and rebind-mutable: a later copy-on-divergence / merge
    /// rebinds `b` to a DIFFERENT head, so a caller MUST NOT cache the id across
    /// any operation that could rebind `b` -- re-resolve instead.
    pub fn head_of(&self, b: &BranchId) -> LoroResult<HeadId> {
        Ok(self.resolve(b, Intent::Read)?.0)
    }
}
