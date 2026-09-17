use rustc_hash::FxHashMap;

use crate::version::Frontiers;
use loro_common::{Counter, InternalString, LoroError, LoroResult, PeerID, ID};

use super::policy::{lineage_name, QuarantineReason};
use super::ROOT_HEAD_ID;
use super::*;

/// The repo's index doc. Its heads hold only frontiers: a root map
/// `docs: LoroMap<DocId, LoroMap<"heads", LoroMap<PeerID, Counter>>>` plus one
/// root list per branch, `lineage:<b>`, whose push is the branch's creation
/// MARKER (see `SelfRooted`). No weft filesystem metadata (that is a TS-side
/// content doc). The only types that appear are `ID`/`Frontiers`/`DocId`/
/// `BranchId`.
pub type IndexDoc = MultiHeadDoc<SelfRooted>;

impl MultiHeadDoc<SelfRooted> {
    /// Establish the first (genesis) branch, bound to the pinned root head, by
    /// writing its creation marker as the root's first op. The commit hook's
    /// `after_commit` sets `tips[genesis]` to that op. Must be called once
    /// before other branches are created.
    pub fn init_genesis(&self, genesis: &BranchId) -> LoroResult<()> {
        let root = self.head_doc(ROOT_HEAD_ID).expect("root head exists");
        let peer = root.peer_id();
        self.with_reg(|this, reg| this.rebind(reg, genesis, ROOT_HEAD_ID));
        root.get_list(lineage_name(genesis).as_str())
            .push(peer as i64)?;
        root.commit_then_renew();
        Ok(())
    }

    /// Create `new` from `from` by EAGER copy: fork `from`'s index head (a state
    /// of a few map entries), then write the copy's first op -- the creation
    /// marker, a push of the copy's fresh peer into `lineage:<new>` -- directly
    /// on the copy. That op depends on `tips[from]` and names `new`, so the
    /// commit hook's `after_commit` sets `tips[new]` to it. Every index head is
    /// thus born `refs == 1`.
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
        let peer = copy_doc.peer_id();
        copy_doc
            .get_list(lineage_name(new).as_str())
            .push(peer as i64)?;
        copy_doc.commit_then_renew();
        Ok(())
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

    /// The imported spans the causal fold refused to attribute (see
    /// `QuarantineReason`), keyed by first op id. Empty on a well-formed
    /// history; a non-empty report is the loud, local signal of a model
    /// violation whose ops stayed in the op log without moving any `tips`.
    pub fn quarantine_report(&self) -> Vec<(ID, QuarantineReason)> {
        self.policy().attribution(self).lock().quarantined.clone()
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
