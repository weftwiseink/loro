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

/// What an `import_index` did: the op import's status, plus the branches whose
/// tip it moved. `Some(docs)`: the docs whose record moved in the branch's tip
/// range, open here or not (the open ones were re-resolved); `Some([])`: the tip
/// moved but no record did, as when a peer's genesis root joins `main`; `None`:
/// the range was not computed (a branch new to this session, or an unresolvable
/// tip) and every open doc was re-resolved. Sorted by branch, then doc. Empty
/// for an already-contained frame.
#[derive(Debug, Clone)]
pub struct IndexImportOutcome {
    pub status: ImportStatus,
    pub moved: Vec<(BranchId, Option<Vec<DocId>>)>,
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

    /// Import lineage bytes, then re-resolve each open content doc whose record
    /// moved on a branch the import moved (fast-forward-only). Closes the
    /// content-before-lineage gap; unresolvable ids are a no-op.
    ///
    /// The docs to re-resolve come from an op-log scan of each moved branch's
    /// old-to-new tip range (`IndexDoc::docs_changed_between`), so the cost is the
    /// bare import plus one resolve per changed open doc, at any mount size. Every
    /// ambiguity resolves more, never less: a branch new to this session (no old
    /// tip) or one whose range cannot be computed re-resolves every open doc. A
    /// frame that moves no tip (already contained) re-resolves nothing and reports
    /// `moved: []`.
    ///
    /// `before` is read outside the import's barrier: a concurrent local commit can
    /// only widen a range. The `docs` lock is held for the whole loop, so a
    /// concurrent `open_doc` lands before or after it, never mid-loop.
    pub fn import_index(&self, bytes: &[u8]) -> LoroResult<IndexImportOutcome> {
        let before = self.index.tips();
        let status = self.index.import(bytes)?;
        let after = self.index.tips();
        let docs = self.docs.lock().unwrap();
        let cause = || ResolveCause::Import { origin: "".into() };
        let mut moved: Vec<(BranchId, Option<Vec<DocId>>)> = Vec::new();
        for (b, new) in &after {
            let scope = match before.get(b) {
                Some(old) if old == new => continue,
                Some(old) => self.index.docs_changed_between(old, new),
                None => RecordScope::Every,
            };
            match scope {
                RecordScope::Docs(set) => {
                    let mut names: Vec<DocId> = set.into_iter().collect();
                    names.sort();
                    for doc in names.iter().filter_map(|d| docs.get(d)) {
                        let _ = doc.resolve_with(b, Intent::Read, cause());
                    }
                    moved.push((b.clone(), Some(names)));
                }
                RecordScope::Every => {
                    for doc in docs.values() {
                        let _ = doc.resolve_with(b, Intent::Read, cause());
                    }
                    moved.push((b.clone(), None));
                }
            }
        }
        moved.sort_by(|x, y| x.0.cmp(&y.0));
        Ok(IndexImportOutcome { status, moved })
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
