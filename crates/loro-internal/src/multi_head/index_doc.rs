use rustc_hash::{FxHashMap, FxHashSet};

use std::cmp::Ordering;

use crate::container::map::MapSet;
use crate::handler::{Handler, HandlerTrait, ValueOrHandler};
use crate::op::{InnerContent, Op};
use crate::oplog::OpLog;
use crate::version::{shrink_frontiers, Frontiers};
use loro_common::{
    ContainerID, ContainerType, Counter, InternalString, Lamport, LoroError, LoroResult, LoroValue,
    PeerID, ID,
};
use rle::HasLength;

use super::policy::{BRANCH_MERGE_KEY, BRANCH_NAME_KEY, BRANCH_ROOT};
use super::ROOT_HEAD_ID;
use super::*;

/// The repo's index doc. Two root maps: `docs: LoroMap<DocId, LoroMap<"heads",
/// LoroMap<PeerID, Counter>>>`, and `branch`, whose `name` key (`"<b>"`) is a
/// branch's creation marker and `merge` key (`"<seq>:<b>"`) a merge marker (see
/// `SelfRooted`).
pub type IndexDoc = MultiHeadDoc<SelfRooted>;

impl MultiHeadDoc<SelfRooted> {
    /// Bind the genesis branch to the pinned root head by writing its creation
    /// marker as the root's first op. Call once, before creating other branches.
    pub fn init_genesis(&self, genesis: &BranchId) -> LoroResult<()> {
        let root = self.head_doc(ROOT_HEAD_ID).expect("root head exists");
        self.with_reg(|this, reg| this.rebind(reg, genesis, ROOT_HEAD_ID));
        root.get_map(BRANCH_ROOT)
            .insert(BRANCH_NAME_KEY, genesis.as_str())?;
        root.commit_then_renew();
        Ok(())
    }

    /// Create `new` from `from` by eager copy, then write the creation marker
    /// `branch.name = "<new>"` on the copy; `after_commit` sets `tips[new]` to it.
    /// Errors if `new` exists: re-forking would mint a second marker and make
    /// `tips[new]` the join of two unrelated lineages.
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

    /// Merge `from` into `into` repo-wide with a single index marker op: advance
    /// `into`'s head to `join(tips[into], tips[from])` (per-key union of every doc's
    /// `docs[D].heads` VV, so concurrent same-branch writers survive), then commit one
    /// marker. Its deps are the join, so the causal fold attributes it to `into` (a
    /// marker needs no dep agreement). `from` is untouched.
    /// Returns `AlreadyContained` (nothing written), else `FastForward`/`Merged` (one marker).
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
            let mut u = ti.clone();
            for id in tf.iter() {
                u.push(id);
            }
            (
                outcome,
                shrink_frontiers(&u, &ol.dag).map_err(LoroError::FrontiersNotFound)?,
            )
        };
        // The marker goes on `branch.merge` as "<seq>:<into>", not
        // `branch.name`, so the set always emits an op. See `BRANCH_MERGE_KEY`.
        self.resolve_with(into, Intent::Read, ResolveCause::Advance)?;
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

    /// Record `docs[doc].heads[peer] = counter` on `b`'s index head.
    pub fn record_head(
        &self,
        b: &BranchId,
        doc: &DocId,
        peer: PeerID,
        counter: Counter,
    ) -> LoroResult<()> {
        self.write(b, |d| {
            // Mergeable containers: concurrent sessions converge to one.
            let docs = d.get_map("docs");
            let per_doc = docs.ensure_mergeable_map(doc.as_str())?;
            let heads = per_doc.ensure_mergeable_map("heads")?;
            heads.insert(&peer.to_string(), counter as i64)
        })?
    }

    /// Additively record every id of `f` into `docs[doc].heads`, keeping other
    /// peers' entries. For forward moves (`advance` / `merge` / ingest); to set
    /// an exact frontier excluding other peers, use `set_frontier`.
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

    /// Set `docs[doc].heads` to exactly `f`, clearing existing entries first.
    /// `create_branch_at` needs this: the seeded parent record may be multi-peer, and
    /// stale other-peer entries would pull those peers' ops into the fork.
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

    /// The raw ids recorded in `docs[doc].heads` on `b`'s index head, before
    /// `Delegated::target` reduces them. Empty if the branch or doc is unknown.
    /// Reads one record through container handles, not a deep value, so it is O(1)
    /// in the mount.
    pub fn recorded_ids(&self, b: &BranchId, doc: &DocId) -> LoroResult<Vec<ID>> {
        self.read(b, |d| {
            let mut ids = Vec::new();
            let Some(ValueOrHandler::Handler(Handler::Map(per_doc))) =
                d.get_map("docs").get_(doc.as_str())
            else {
                return ids;
            };
            let Some(ValueOrHandler::Handler(Handler::Map(heads))) = per_doc.get_("heads") else {
                return ids;
            };
            if let LoroValue::Map(m) = heads.get_value() {
                for (peer_str, counter) in m.iter() {
                    if let (Ok(peer), Some(c)) = (peer_str.parse::<PeerID>(), counter.as_i64()) {
                        ids.push(ID::new(peer, *c as Counter));
                    }
                }
            }
            ids
        })
    }

    /// The docs whose `docs[D].heads` record an op in `vv(new) - vv(old)` touches,
    /// read from the op log alone (a mergeable child's cid names its key path).
    /// An empty `old` covers the whole causal past of `new`. O(ops in the range).
    ///
    /// Fails SAFE. A frontier the DAG cannot resolve, or a `new` that does not
    /// include `old` (its range would read empty), yields [`RecordScope::Every`],
    /// never an empty set, which would silently leave an open editor on a stale
    /// head. Assumes the index op log is never shallow-trimmed below a branch
    /// root (as `branch_sources` and causal attribution do).
    pub fn docs_changed_between(&self, old: &Frontiers, new: &Frontiers) -> RecordScope {
        let ol = self.oplog.lock();
        let (Some(vv_new), Some(vv_old)) =
            (ol.dag.frontiers_to_vv(new), ol.dag.frontiers_to_vv(old))
        else {
            tracing::warn!("index: cannot resolve a branch tip range; re-resolving every open doc");
            return RecordScope::Every;
        };
        if !vv_new.includes_vv(&vv_old) {
            tracing::warn!("index: a branch tip moved backward; re-resolving every open doc");
            return RecordScope::Every;
        }
        let mut out = FxHashSet::default();
        for span in vv_new.sub_iter(&vv_old) {
            // `iter_changes` yields whole changes overlapping the span: clip to its ops.
            let in_span = |op: &Op| {
                op.counter < span.counter.end
                    && op.counter + op.atom_len() as Counter > span.counter.start
            };
            for ch in ol.iter_changes(span) {
                for op in ch.ops().iter().filter(|op| in_span(op)) {
                    if let Some(d) = doc_of_op(&ol, op) {
                        out.insert(d);
                    }
                }
            }
        }
        RecordScope::Docs(out)
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

    /// Every branch's index frontier.
    pub fn tips(&self) -> FxHashMap<BranchId, Frontiers> {
        self.policy().attribution(self).lock().tips.clone()
    }

    /// The index frontier `b` was forked from: the deps of its creation marker.
    /// Empty for genesis and unknown branches. Only the index can answer this
    /// cross-doc, even for a doc nobody has touched on `b`.
    pub fn fork_point(&self, b: &BranchId) -> Frontiers {
        self.creation_marker_id(b)
            .and_then(|id| self.oplog.lock().get_deps_of(id))
            .unwrap_or_default()
    }

    /// The op-id of `b`'s creation marker: the min-lamport run-start labelled
    /// `b` across peers (the marker is `b`'s causal root). `None` for an unknown
    /// branch. The `Attribution` guard is released before `OpLog` is taken
    /// (`Attribution` is the leaf).
    fn creation_marker_id(&self, b: &BranchId) -> Option<ID> {
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
        starts.into_iter().min_by_key(|id| {
            ol.get_change_at(*id)
                .map(|c| c.lamport())
                .unwrap_or(Lamport::MAX)
        })
    }

    /// Every known branch's source: creation marker op-id, kind, and parent branch
    /// (the branch of the marker's deps; `None` for genesis). Assumes the index oplog
    /// is not shallow-trimmed below a branch root.
    pub fn branch_sources(&self) -> LoroResult<FxHashMap<BranchId, BranchSource>> {
        let mut out = FxHashMap::default();
        for b in self.branches() {
            let Some(op) = self.creation_marker_id(&b) else {
                continue;
            };
            let deps = self.oplog.lock().get_deps_of(op).unwrap_or_default();
            let parent = if deps.is_empty() {
                // Genesis: `branch_of_deps(empty)` is an Err.
                None
            } else {
                Some(
                    self.policy()
                        .attribution(self)
                        .lock()
                        .branch_of_deps(&deps)?,
                )
            };
            out.insert(
                b,
                BranchSource {
                    op,
                    kind: BranchSourceKind::Create,
                    parent,
                },
            );
        }
        Ok(out)
    }

    /// The `docs` map (`DocId` -> `heads`) at frontier `f`, read on a scratch head.
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

    /// The docs `b` itself modified: those whose recorded frontier at `tips[b]`
    /// differs from `fork_point(b)`. Inherited docs are excluded (a fork eager-copies
    /// the parent's record). Empty for an unknown branch.
    ///
    /// Computed on demand: an incremental set copied at fork is wrong under a
    /// concurrent fork (a concurrent parent op would leak into the child's set).
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

    /// The docs whose recorded head differs between `a` and `b`, including docs on one
    /// side only. Symmetric, unlike `docs_modified_on_branch`.
    pub fn diff_docs(&self, a: &BranchId, b: &BranchId) -> LoroResult<Vec<DocId>> {
        let ta = self.policy().target(self, a)?;
        let tb = self.policy().target(self, b)?;
        let at_a = self.docs_map_at(&ta)?;
        let at_b = self.docs_map_at(&tb)?;
        let keys: FxHashSet<&DocId> = at_a.keys().chain(at_b.keys()).collect();
        let mut out: Vec<DocId> = keys
            .into_iter()
            .filter(|dc| at_a.get(*dc) != at_b.get(*dc))
            .cloned()
            .collect();
        out.sort();
        Ok(out)
    }

    /// The docs modified anywhere on `b`'s lineage since it diverged from
    /// genesis: a diff from where the lineage left genesis to `tips[b]`. Empty
    /// for genesis.
    pub fn docs_changed_since_main(&self, b: &BranchId) -> LoroResult<Vec<DocId>> {
        let departure = self.departure_from_main(b);
        if departure == Frontiers::default() {
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

    /// The fork point of `b`'s topmost non-genesis ancestor, found by walking
    /// parent fork-points up to genesis. Empty for genesis itself.
    fn departure_from_main(&self, b: &BranchId) -> Frontiers {
        let mut fork = self.fork_point(b);
        loop {
            if fork == Frontiers::default() {
                return Frontiers::default();
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
                return fork;
            }
            fork = parent_fork;
        }
    }

    /// Unbind `name`'s index head and drop its `tips` entry. Attribution runs
    /// are kept, so a later-imported change depending on its history still
    /// attributes (and re-lists the branch). Local cleanup only.
    pub fn delete_index_branch(&self, name: &BranchId) {
        self.unbind(name);
        self.policy().attribution(self).lock().tips.remove(name);
    }

    /// The `HeadId` of `b`'s index head, resolving it if unbound (with `resolve`'s side
    /// effects). Session-local: do not cache across a rebind.
    pub fn head_of(&self, b: &BranchId) -> LoroResult<HeadId> {
        Ok(self.resolve(b, Intent::Read)?.0)
    }
}

/// Which open docs a moved branch must re-resolve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordScope {
    /// Exactly these docs' records moved.
    Docs(FxHashSet<DocId>),
    /// Unknown (a branch new to this session, or an unresolvable tip): every open doc.
    Every,
}

/// The doc an index op records a head for: an op on the mergeable `docs[D].heads`
/// map, or the `docs` root's marker set keyed `D`. `None` for any other op: branch
/// markers, and the `docs[D]` map's own `"heads"` marker set (its cid parses
/// mergeable with key `D`). The same commit always carries the `heads` insert, so
/// that `None` is not a miss.
fn doc_of_op(ol: &OpLog, op: &Op) -> Option<DocId> {
    let cid = ol.arena.idx_to_id(op.container)?;
    if let Some((parent, key, ContainerType::Map)) = cid.parse_mergeable() {
        if key == "heads" {
            if let Some((root, doc, ContainerType::Map)) = parent.parse_mergeable() {
                if matches!(&root, ContainerID::Root { name, .. } if name.as_str() == "docs") {
                    return Some(doc.into());
                }
            }
        }
        return None;
    }
    if let ContainerID::Root {
        name,
        container_type: ContainerType::Map,
    } = &cid
    {
        if name.as_str() == "docs" {
            if let InnerContent::Map(MapSet { key, .. }) = &op.content {
                return Some(key.clone());
            }
        }
    }
    None
}
