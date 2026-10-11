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

    /// Repo-wide merge of `from` into `into`: one index marker op, then re-resolve
    /// every open content doc so its subscribers get their catch-up event.
    pub fn merge_branch(&self, into: &BranchId, from: &BranchId) -> LoroResult<MergeOutcome> {
        let outcome = self.index.merge(into, from)?;
        // A resolve that cannot yet advance (ids not held) is non-fatal.
        for doc in self.docs.lock().unwrap().values() {
            let _ = doc.resolve_with(into, Intent::Read, ResolveCause::Advance);
        }
        Ok(outcome)
    }

    /// Import lineage bytes, then re-resolve every open content doc's index-known
    /// branches (fast-forward-only). Closes the content-before-lineage gap;
    /// unresolvable ids are a no-op.
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

    /// Unbind the branch from every open content doc and the index, and drop its
    /// index lineage. Local-only: a later sync re-discovers the branch from its marker
    /// (`SelfRooted::after_import`).
    pub fn delete_branch(&self, name: &BranchId) -> LoroResult<()> {
        for doc in self.docs.lock().unwrap().values() {
            doc.unbind(name);
        }
        self.index.delete_index_branch(name);
        Ok(())
    }
}
