use std::sync::Arc;

use rustc_hash::FxHashMap;

use loro_common::LoroResult;

use crate::encoding::ImportStatus;

use super::*;

/// The repo: one loro index doc plus its content docs. Branch existence and
/// "where each branch is for each doc" live in the index; content docs delegate.
#[allow(missing_debug_implementations)]
pub struct BranchingDocRepo {
    index: Arc<IndexDoc>,
    docs: std::sync::Mutex<FxHashMap<DocId, Arc<BranchingDoc>>>,
}

/// The genesis branch every repo opens with.
pub const GENESIS_BRANCH: &str = "main";

impl BranchingDocRepo {
    /// Open a fresh repo with the genesis branch.
    pub fn open() -> LoroResult<Self> {
        let index = Arc::new(IndexDoc::new(SelfRooted::new()));
        index.init_genesis(&GENESIS_BRANCH.into())?;
        Ok(BranchingDocRepo {
            index,
            docs: std::sync::Mutex::new(FxHashMap::default()),
        })
    }

    /// The index doc (pure ids / frontiers).
    pub fn index(&self) -> &IndexDoc {
        &self.index
    }

    /// The branches the repo knows (from the index lineage).
    pub fn branches(&self) -> Vec<BranchId> {
        self.index.branches()
    }

    /// Open (or get) a content doc. Idempotent per id.
    pub fn open_doc(&self, id: DocId) -> Arc<BranchingDoc> {
        let mut docs = self.docs.lock().unwrap();
        if let Some(d) = docs.get(&id) {
            return d.clone();
        }
        let bd = Arc::new(MultiHeadDoc::new(Delegated::new(
            id.clone(),
            self.index.clone(),
        )));
        docs.insert(id, bd.clone());
        bd
    }

    /// Drop the in-memory content doc (its recorded frontiers stay in the index;
    /// reopening re-materializes lazily).
    pub fn close_doc(&self, id: &DocId) {
        self.docs.lock().unwrap().remove(id);
    }

    /// Repo-wide branch creation: eager-copy the index head for `new` from
    /// `from`; content docs bind `new` lazily on first access (free creation).
    pub fn create_branch(&self, new: &BranchId, from: &BranchId) -> LoroResult<()> {
        self.index.create_index_branch(new, from)
    }

    /// Repo-wide merge of `from` into `into`: ONE index marker op (the union of
    /// both branches' recorded doc frontiers falls out of the index checkout, no
    /// per-doc record), then re-resolve every OPEN content doc so its branch
    /// subscribers receive their catch-up event. A content doc not currently open
    /// picks up the merged frontier lazily on reopen (the index carries it).
    ///
    /// Returns the index-level `MergeOutcome` (`AlreadyContained` when `into`
    /// already contains `from`).
    pub fn merge_branch(&self, into: &BranchId, from: &BranchId) -> LoroResult<MergeOutcome> {
        let outcome = self.index.merge(into, from)?;
        // Catch every open content doc up to the merged index frontier. A resolve
        // that cannot yet advance (ids not held) is non-fatal, exactly as the
        // import ingest path treats it.
        for doc in self.docs.lock().unwrap().values() {
            let _ = doc.resolve_with(into, Intent::Read, ResolveCause::Advance);
        }
        Ok(outcome)
    }

    /// Import LINEAGE/index bytes (Trigger 2 of the bidirectional lens), then
    /// re-resolve every index-known branch of every OPEN content doc so a lens
    /// bound on a branch whose recorded frontier just moved advances FORWARD to it
    /// (guarded fast-forward-only). This closes the content-first ordering gap: a
    /// content op that arrived before its lineage record becomes a held descendant
    /// the moment the record lands, and the re-resolve advances the lens then. A
    /// resolve that cannot yet advance (ids not held) is a non-fatal no-op, exactly
    /// as the content-import ingest treats it; nothing touches an unopened doc (the
    /// loop is over `self.docs`, the docs already hydrated). Mirrors `merge_branch`,
    /// with the imported-frontier `Import` cause rather than a local `Advance`.
    ///
    /// The index import itself is history-only + all-heads-barriered (it lands
    /// index ops without moving a head), so the content re-resolves are the only
    /// state moves. See `cdocs/proposals/2026-09-20-branch-bidirectional-lens.md`.
    pub fn import_index(&self, bytes: &[u8]) -> LoroResult<ImportStatus> {
        let status = self.index.import(bytes)?;
        let branches = self.index.branches();
        for doc in self.docs.lock().unwrap().values() {
            for b in &branches {
                let _ = doc.resolve_with(b, Intent::Read, ResolveCause::Import { origin: "".into() });
            }
        }
        Ok(status)
    }

    /// Repo-wide branch deletion: unbind the branch from every open content doc
    /// AND the index, and drop its index lineage. `branches()` then excludes it
    /// with no dangling `bound`/`by_tip` entry anywhere; the pinned root is never
    /// removed. Its committed ops stay in each doc's history (unreferenced).
    ///
    /// NOTE(claude-opus-4-8/branchingdocrepo-multiheaddoc): this is LOCAL-ONLY
    /// registry cleanup. The branch's marker op (`branch.name = "<name>"`)
    /// remains in the shared index history, so a later sync that re-imports it
    /// re-discovers the branch (`SelfRooted::after_import`). Durable cross-peer
    /// deletion /
    /// tombstoning is the wrapper's lifecycle-log job and a Phase-5 follow-up;
    /// this method does not attempt it.
    pub fn delete_branch(&self, name: &BranchId) -> LoroResult<()> {
        for doc in self.docs.lock().unwrap().values() {
            doc.unbind(name);
        }
        self.index.delete_index_branch(name);
        Ok(())
    }
}
