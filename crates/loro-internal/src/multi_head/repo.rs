use std::sync::Arc;

use rustc_hash::FxHashMap;

use loro_common::LoroResult;

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
