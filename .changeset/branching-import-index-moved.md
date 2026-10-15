---
"loro-crdt": patch
---

Make `BranchingDocRepo.importIndex` cost what the frame changed: it re-resolves
only the open docs whose record moved on the branches the import moved, and
returns `IndexImportOutcome`, the existing `ImportStatus` plus an additive,
sorted `moved: Array<{ branch, docs: string[] | null }>` naming them (`[]` for
an already-contained frame).
