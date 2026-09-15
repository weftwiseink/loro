import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BranchingDocRepo, type LoroEventBatch } from "../bundler/index";

// The discriminator: proves branch data crosses the wasm boundary correctly — a write is
// readable back on its branch, a divergent branch stays ISOLATED (copy-on-divergence), and
// `merge` CONVERGES. The multi-peer `create_branch_at` bug class lives on exactly this seam.
describe("BranchingDocRepo wasm surface", () => {
  it("round-trips a write and reads it back on the same branch", () => {
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    doc.branch("main").write((head) => head.getText("t").insert(0, "main"));
    const text = doc.branch("main").read((head) => head.getText("t").toString());
    expect(text).toBe("main");
  });

  it("keeps a divergent branch isolated (copy-on-divergence)", () => {
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    doc.branch("main").write((head) => head.getText("t").insert(0, "main"));

    repo.createBranch("draft", "main");
    expect(repo.branches().sort()).toEqual(["draft", "main"]);

    // draft shares main's head until it writes.
    expect(doc.branch("draft").read((h) => h.getText("t").toString())).toBe(
      "main",
    );

    // draft diverges (copy-on-divergence); main is UNAFFECTED.
    doc.branch("draft").write((h) => h.getText("t").insert(4, "-draft"));
    expect(doc.branch("draft").read((h) => h.getText("t").toString())).toBe(
      "main-draft",
    );
    expect(doc.branch("main").read((h) => h.getText("t").toString())).toBe(
      "main",
    );
  });

  it("merge converges: the merged branch sees both branches' ops", () => {
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    doc.branch("main").write((h) => h.getText("t").insert(0, "base"));
    repo.createBranch("feature", "main");

    // Both branches diverge with a concurrent edit at the same position.
    doc.branch("main").write((h) => h.getText("t").insert(4, "M"));
    doc.branch("feature").write((h) => h.getText("t").insert(4, "F"));
    expect(doc.contains("main", "feature")).toBe(false);

    expect(doc.merge("main", "feature")).toBe("merged");

    // main now contains BOTH edits — no dropped op.
    const text = doc.branch("main").read((h) => h.getText("t").toString());
    expect([...text].length).toBe(6);
    expect(text.startsWith("base")).toBe(true);
    expect(text.includes("M")).toBe(true);
    expect(text.includes("F")).toBe(true);
    // feature is now contained in main.
    expect(doc.contains("main", "feature")).toBe(true);
  });

  it("merge reports fast-forward and already-contained", () => {
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    doc.branch("main").write((h) => h.getText("t").insert(0, "x"));
    repo.createBranch("feature", "main");
    // Only feature advances; main is behind -> fast-forward.
    doc.branch("feature").write((h) => h.getText("t").insert(1, "y"));
    expect(doc.merge("main", "feature")).toBe("fast-forward");
    expect(doc.branch("main").read((h) => h.getText("t").toString())).toBe("xy");
    // Merging again: already contained.
    expect(doc.merge("main", "feature")).toBe("already-contained");
  });

  it("createBranchAt forks at a historical frontier and errors on a used name", () => {
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    doc.branch("main").write((h) => h.getText("t").insert(0, "v1"));
    const at = doc.frontierOf("main");
    doc.branch("main").write((h) => h.getText("t").insert(2, "-v2"));

    doc.createBranchAt("snapshot", at);
    expect(doc.branch("snapshot").read((h) => h.getText("t").toString())).toBe(
      "v1",
    );
    expect(doc.branch("main").read((h) => h.getText("t").toString())).toBe(
      "v1-v2",
    );
    expect(() => doc.createBranchAt("snapshot", at)).toThrow();
  });
});

// Cross-peer byte-sync convergence: the highest data-loss surface. Two independent
// `BranchingDocRepo` instances (peer A, peer B) sync via DOC-LEVEL export/import — index
// bytes (branch existence + per-doc frontier records) then content bytes (the ops). The
// `after_import` eager re-resolve (SelfRooted for the index, Delegated for content) must
// drive BOTH peers to IDENTICAL branch state with no dropped and no duplicated op. The head
// stays out of it: sync never touches a `BranchingDocHead`.
describe("BranchingDocRepo cross-peer sync", () => {
  const UPDATE = { mode: "update" } as const;

  // Snapshot every peer's index + content export BEFORE any import, then cross-apply: index
  // both directions (a peer learns the other's branches) then content both directions (the
  // ops land and every known branch is eagerly re-resolved). One symmetric round converges
  // even with concurrent branch creation, because each direction imports index before content.
  function converge(
    a: BranchingDocRepo,
    b: BranchingDocRepo,
    docIds: string[],
  ) {
    const aIdx = a.index().export(UPDATE);
    const bIdx = b.index().export(UPDATE);
    const aDocs = docIds.map((id) => a.openDoc(id).export(UPDATE));
    const bDocs = docIds.map((id) => b.openDoc(id).export(UPDATE));

    a.index().import(bIdx);
    b.index().import(aIdx);
    docIds.forEach((id, i) => {
      a.openDoc(id).import(bDocs[i]);
      b.openDoc(id).import(aDocs[i]);
    });
  }

  const readBranch = (repo: BranchingDocRepo, doc: string, branch: string) =>
    repo.openDoc(doc).branch(branch).read((h) => h.getText("t").toString());

  it("converges concurrent edits + per-peer branches to identical state (no drop, no dup)", () => {
    const a = new BranchingDocRepo();
    const b = new BranchingDocRepo();

    // Concurrent edits on the SAME branch (main) on doc G — the classic conflicting-write case.
    a.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "A"));
    b.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "B"));

    // Each peer forks a DIFFERENT branch off its own main and edits it.
    a.createBranch("featA", "main");
    a.openDoc("G").branch("featA").write((h) => h.getText("t").insert(1, "-fa"));
    b.createBranch("featB", "main");
    b.openDoc("G").branch("featB").write((h) => h.getText("t").insert(1, "-fb"));

    converge(a, b, ["G"]);

    // Both peers know the same branch set.
    expect(a.branches().sort()).toEqual(["featA", "featB", "main"]);
    expect(b.branches().sort()).toEqual(["featA", "featB", "main"]);

    // main converged: BOTH concurrent ops present, exactly once each, identical across peers.
    const mainA = readBranch(a, "G", "main");
    const mainB = readBranch(b, "G", "main");
    expect(mainA).toBe(mainB); // deterministic CRDT merge -> byte-identical on both peers
    expect([...mainA].sort().join("")).toBe("AB"); // no dropped op, no duplicate (len 2)

    // Every branch resolves identically on both peers — the convergence invariant.
    for (const br of a.branches()) {
      expect(readBranch(a, "G", br)).toBe(readBranch(b, "G", br));
    }
    // featA carries main("A") + its own edit; featB carries main("B") + its own edit.
    expect(readBranch(a, "G", "featA")).toBe("A-fa");
    expect(readBranch(a, "G", "featB")).toBe("B-fb");
  });

  it("createBranchAt at a single-peer frontier does NOT leak other-peer ops across the boundary", () => {
    // The A1-not-A1B1 bug class (Phase-3 round 2), lifted to the wasm boundary.
    const a = new BranchingDocRepo();
    a.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "A1"));
    const atA = a.openDoc("G").frontierOf("main"); // A's single-peer frontier

    const b = new BranchingDocRepo();
    b.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "B1"));

    // Sync A into B so B's main is MULTI-PEER.
    b.index().import(a.index().export(UPDATE));
    b.openDoc("G").import(a.openDoc("G").export(UPDATE));
    const mainB = readBranch(b, "G", "main");
    expect(mainB.includes("A1") && mainB.includes("B1")).toBe(true);

    // Fork at A's single-peer historical frontier: it must be A1 ALONE, not the multi-peer parent.
    b.openDoc("G").createBranchAt("hist", atA);
    expect(readBranch(b, "G", "hist")).toBe("A1");
  });

  it("a cross-peer merge converges: the merge advancement propagates back", () => {
    const a = new BranchingDocRepo();
    a.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "base"));
    a.createBranch("feat", "main");
    a.openDoc("G").branch("feat").write((h) => h.getText("t").insert(4, "-feat"));

    const b = new BranchingDocRepo();
    b.openDoc("G"); // so B has a content doc to sync into
    converge(a, b, ["G"]);

    // B sees feat before the merge; main does not yet contain it.
    expect(readBranch(b, "G", "feat")).toBe("base-feat");
    expect(b.openDoc("G").contains("main", "feat")).toBe(false);

    // B merges feat into main (frontier advancement, recorded in B's index).
    expect(b.openDoc("G").merge("main", "feat")).toBe("fast-forward");
    expect(b.openDoc("G").contains("main", "feat")).toBe(true);

    // Sync back: A's main must advance to contain feat too (the merge propagates).
    converge(a, b, ["G"]);
    expect(a.openDoc("G").contains("main", "feat")).toBe(true);
    expect(readBranch(a, "G", "main")).toBe(readBranch(b, "G", "main"));
    expect(readBranch(a, "G", "main")).toBe("base-feat");
  });
});

// Branch-scoped subscriptions: proves the `subscribe`/`subscribeRoot` bindings actually DELIVER a
// change event to a JS callback after a mutation lands on the branch (not merely that the call
// does not throw), that the container-scoped variant carries the right target, and that the
// unsubscribe closure stops delivery.
//
// Delivery needs NO manual flush: `Branch.write` is decorated in `index.ts` to auto-flush
// `callPendingEvents()` after it commits (mirroring `LoroDoc`/`BranchingDocHead.commit`), so a
// consumer doing `subscribeRoot(cb); write(...)` observes the change directly. The callback arg is
// the exported `LoroEventBatch` type — `{ by, origin, events: [{ target, path, diff }], from, to }`.
describe("Branch subscriptions deliver change events", () => {
  it("subscribeRoot fires on a branch write and stops after unsubscribe", () => {
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    const branch = doc.branch("main");

    const events: LoroEventBatch[] = [];
    const unsubscribe = branch.subscribeRoot((e) => events.push(e));

    // A mutation on the branch's content doc reaches the root subscriber with no manual flush.
    branch.write((h) => h.getText("t").insert(0, "hi"));

    expect(events.length).toBe(1);
    // The delivered payload is a real change batch: a LOCAL edit carrying the text container's diff.
    expect(events[0].by).toBe("local");
    expect(events[0].events.length).toBeGreaterThan(0);
    expect(events[0].events.some((ev) => ev.target === "cid:root-t:Text")).toBe(
      true,
    );

    // After unsubscribe, a further write delivers NOTHING (the callback is detached).
    unsubscribe();
    branch.write((h) => h.getText("t").insert(2, "!"));
    expect(events.length).toBe(1);
  });

  it("subscribe(containerId) fires for the scoped container on a write", () => {
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    const branch = doc.branch("main");

    // Materialize the container first, then subscribe to it by id.
    branch.write((h) => h.getText("t").insert(0, "x"));
    const cid = branch.read((h) => h.getText("t").id);

    const events: LoroEventBatch[] = [];
    const unsubscribe = branch.subscribe(cid, (e) => events.push(e));

    branch.write((h) => h.getText("t").insert(1, "y"));

    expect(events.length).toBe(1);
    expect(events[0].by).toBe("local");
    // The container-scoped subscription's event targets exactly the subscribed container.
    expect(events[0].events[0].target).toBe("cid:root-t:Text");

    unsubscribe();
    branch.write((h) => h.getText("t").insert(2, "z"));
    expect(events.length).toBe(1);
  });

  it("a TRUE 3-way merge delivers a catch-up event to the target subscriber", () => {
    // The discriminator the fast-forward case misses: two DIVERGENT edits at the same position
    // force a real merge (outcome "merged"), which catches the target head up via
    // `advance_in_place` -> `head.checkout`, enqueuing a `by:"checkout"` event. `merge` is
    // decorated to auto-flush it, so the `main` subscriber must observe it with NO manual flush.
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    doc.branch("main").write((h) => h.getText("t").insert(0, "base"));
    repo.createBranch("feature", "main");
    doc.branch("main").write((h) => h.getText("t").insert(4, "M"));
    doc.branch("feature").write((h) => h.getText("t").insert(4, "F"));
    // Precondition: this is a genuine divergence, not a fast-forward.
    expect(doc.contains("main", "feature")).toBe(false);

    const events: LoroEventBatch[] = [];
    const unsubscribe = doc.branch("main").subscribeRoot((e) => events.push(e));

    expect(doc.merge("main", "feature")).toBe("merged"); // TRUE 3-way merge, not "fast-forward"

    // The catch-up event is delivered with no manual `callPendingEvents()`.
    expect(events.length).toBe(1);
    expect(events[0].by).toBe("checkout"); // the copy/catch-up arm, not a local edit
    expect(events[0].events.some((ev) => ev.target === "cid:root-t:Text")).toBe(
      true,
    );
    // main now carries BOTH divergent edits (6 chars) — the subscriber's view is not stale.
    expect(
      doc.branch("main").read((h) => [...h.getText("t").toString()].length),
    ).toBe(6);

    unsubscribe();
  });
});

// The wire-surface discriminator: `subscribeLocalUpdates` + persistent `version()` /
// `oplogVersion()` on the branching doc/index, the surface weft's `EnvelopedAdaptor` drives the
// wire off. Proven on a NON-MAIN branch (the multi-head case), because the ops land on a
// copy-on-divergence head — a version-tracking loop keyed off only main's state would MISS them.
//
// Two independent failures this must catch:
//   - `subscribeLocalUpdates` firing only for the root/main head (if forwarders were not installed
//     on the copy-on-divergence head): the feat edit would deliver NO callback.
//   - `version()` reflecting a single head's DocState VV instead of the shared oplog vv: a feat
//     edit would NOT advance it, so `after.compare(before)` would be 0, not 1.
describe("BranchingDoc/Index wire surface: subscribeLocalUpdates + version delta round-trip", () => {
  const UPDATE = { mode: "update" } as const;
  const readBranch = (repo: BranchingDocRepo, doc: string, branch: string) =>
    repo.openDoc(doc).branch(branch).read((h) => h.getText("t").toString());

  it("delta-syncs a NON-MAIN branch edit to a peer via the callback bytes and via export(from)", () => {
    const a = new BranchingDocRepo();
    const b = new BranchingDocRepo(); // converges via export({from}) deltas
    const c = new BranchingDocRepo(); // converges via the subscribeLocalUpdates callback bytes

    // Base on main, fork a NON-MAIN branch, and bring both peers to this shared pre-edit state.
    a.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "base"));
    a.createBranch("feat", "main");
    for (const peer of [b, c]) {
      peer.openDoc("G");
      peer.index().import(a.index().export(UPDATE));
      peer.openDoc("G").import(a.openDoc("G").export(UPDATE));
    }
    expect(b.branches().sort()).toEqual(["feat", "main"]);
    expect(readBranch(b, "G", "feat")).toBe("base");

    // Capture the shared-oplog versions BEFORE the non-main edit, on BOTH the content doc and the
    // index, and wire both local-update streams (what an EnvelopedAdaptor per doc would send).
    const beforeDoc = a.openDoc("G").version();
    const beforeIdx = a.index().version();
    const docUpdates: Uint8Array[] = [];
    const idxUpdates: Uint8Array[] = [];
    const unsubDoc = a.openDoc("G").subscribeLocalUpdates((bytes) => docUpdates.push(bytes));
    const unsubIdx = a.index().subscribeLocalUpdates((bytes) => idxUpdates.push(bytes));

    // The edit lands on a NON-MAIN branch (a copy-on-divergence head).
    a.openDoc("G").branch("feat").write((h) => h.getText("t").insert(4, "-feat"));

    // (1) subscribeLocalUpdates FIRED for the non-main edit — on the content doc (the feat ops)
    //     and on the index (feat's advanced frontier record, committed by the Delegated policy).
    expect(docUpdates.length).toBeGreaterThan(0);
    expect(idxUpdates.length).toBeGreaterThan(0);

    // (2) version()/oplogVersion() ADVANCED on the non-main edit. A head-state VV keyed off main
    //     would NOT move here — this is the version-tracking-loop-misses-branch-edits discriminator.
    const afterDoc = a.openDoc("G").version();
    expect(afterDoc.compare(beforeDoc)).toBe(1);
    expect(a.openDoc("G").oplogVersion().compare(beforeDoc)).toBe(1);
    expect(a.index().version().compare(beforeIdx)).toBe(1);

    // (3) export({mode:"update", from: before}) yields exactly the delta for that edit: a peer
    //     already at `before` converges by importing ONLY the delta (no full re-export).
    const docDelta = a.openDoc("G").export({ mode: "update", from: beforeDoc });
    const idxDelta = a.index().export({ mode: "update", from: beforeIdx });
    expect(docDelta.length).toBeGreaterThan(0);
    expect(idxDelta.length).toBeGreaterThan(0);

    b.index().import(idxDelta);
    b.openDoc("G").import(docDelta);
    expect(readBranch(b, "G", "feat")).toBe("base-feat"); // converged on the NON-MAIN branch
    expect(readBranch(b, "G", "main")).toBe("base"); // isolated: main untouched

    // (4) The wire bytes the adaptor would SEND (the callback payloads) are themselves valid
    //     deltas: a peer at `before` fed only the callback bytes converges identically.
    unsubDoc();
    unsubIdx();
    for (const bytes of idxUpdates) c.index().import(bytes);
    for (const bytes of docUpdates) c.openDoc("G").import(bytes);
    expect(readBranch(c, "G", "feat")).toBe("base-feat");
    expect(readBranch(c, "G", "main")).toBe("base");
  });

  // The NON-ECHO invariant, permanently guarded: `subscribeLocalUpdates` fires on LOCAL edits ONLY,
  // NEVER on `import`. A regression that let an import echo onto the local-update stream would be a
  // PRODUCTION SYNC LOOP (a peer re-broadcasts every op it receives). Covers BOTH the content
  // `BranchingDoc` and the `BranchingIndex` — including the index's `after_import` policy-resolve
  // path (SelfRooted discovers the remote branch and advances index heads), the sharpest residual
  // echo risk. Proven live too: B's OWN edits DO fire, so a passing "0 on import" is not a dead callback.
  it("import does NOT echo onto subscribeLocalUpdates (content doc + index); a local edit DOES", () => {
    const a = new BranchingDocRepo();
    const b = new BranchingDocRepo();

    // A builds a NON-MAIN branch with an edit (the multi-head case).
    a.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "base"));
    a.createBranch("feat", "main");
    a.openDoc("G").branch("feat").write((h) => h.getText("t").insert(4, "-feat"));

    b.openDoc("G"); // B has a content doc to import into

    // Wire B's local-update streams BEFORE importing, so ANY echo from `import` (incl. the index
    // after_import resolve path and the content after_import re-resolve) is caught.
    const bDocUpdates: Uint8Array[] = [];
    const bIdxUpdates: Uint8Array[] = [];
    const unsubDoc = b.openDoc("G").subscribeLocalUpdates((bytes) => bDocUpdates.push(bytes));
    const unsubIdx = b.index().subscribeLocalUpdates((bytes) => bIdxUpdates.push(bytes));

    // Import A's REMOTE updates: index first (learn feat + frontier -> SelfRooted after_import
    // advances index heads), then content (feat ops -> Delegated after_import re-resolves).
    b.index().import(a.index().export(UPDATE));
    b.openDoc("G").import(a.openDoc("G").export(UPDATE));

    // NON-ECHO: importing remote ops fires the LOCAL-update stream ZERO times on BOTH surfaces.
    expect(bDocUpdates.length).toBe(0);
    expect(bIdxUpdates.length).toBe(0);

    // Not vacuous: the import genuinely landed and B converged (so the streams had a real chance to fire).
    expect(b.branches().sort()).toEqual(["feat", "main"]);
    expect(readBranch(b, "G", "feat")).toBe("base-feat");

    // ALIVE (content): B's OWN local edit DOES fire the content stream.
    b.openDoc("G").branch("feat").write((h) => h.getText("t").insert(0, "x"));
    expect(bDocUpdates.length).toBeGreaterThan(0);

    // ALIVE (index): a LOCAL frontier advance (merge feat -> main, a fast-forward) records a new
    // frontier into B's index -> a local index commit -> fires the index stream. `BranchingDoc.merge`
    // is auto-flushed (decorated in index.ts), so delivery is synchronous and deterministic here.
    expect(b.openDoc("G").merge("main", "feat")).toBe("fast-forward");
    expect(bIdxUpdates.length).toBeGreaterThan(0);

    unsubDoc();
    unsubIdx();
  });
});

// Repo-level branch-lifecycle streaming: `BranchingDocRepo.createBranch` commits a lineage op
// onto the shared INDEX op log, which must reach the `BranchingIndex.subscribeLocalUpdates` wire
// stream so a peer learns of a locally-created branch INCREMENTALLY (not only via a full
// re-export). The op is enqueued onto the global pending-event queue at commit time, but the
// queue flushes only when a DECORATED method runs `callPendingEvents()`; `createBranch`/
// `deleteBranch` are decorated in `index.ts` for exactly this reason.
//
// The regression this guards (observed before the decoration): `createBranch` streamed ZERO
// frames AND logged `[LORO_INTERNAL_ERROR] Event not called` (the microtask check finding the
// enqueued frame never flushed), even though a full `export({mode:"update"})` afterward DID carry
// the branch. That is the "branch never reaches the live wire" bug: peer B never learns peer A's
// freshly-created branch incrementally.
describe("BranchingDocRepo branch-lifecycle local-update streaming", () => {
  const UPDATE = { mode: "update" } as const;

  // Capture `console.error` so the microtask-scheduled `[LORO_INTERNAL_ERROR] Event not called`
  // (logged when an enqueued pending event was never flushed) is assertable.
  const consoleErrors: string[] = [];
  // Only `mockRestore` is used; a structural type sidesteps vitest's version-specific
  // `MockInstance` generic without an `any`.
  let errSpy: { mockRestore: () => void };
  beforeEach(() => {
    consoleErrors.length = 0;
    errSpy = vi.spyOn(console, "error").mockImplementation((...args) => {
      consoleErrors.push(args.map((a) => String(a)).join(" "));
    });
  });
  afterEach(() => {
    errSpy.mockRestore();
  });
  const noInternalError = () =>
    consoleErrors.some((e) => e.includes("[LORO_INTERNAL_ERROR]"));

  // Drain the microtask/macrotask turn so `schedule_pending_event_check`'s promise callback runs.
  // That callback is where `[LORO_INTERNAL_ERROR] Event not called` would be logged on a missed flush.
  const drain = () => new Promise((r) => setTimeout(r, 0));

  it("createBranch streams its lineage op on the index stream with NO manual flush (no LORO_INTERNAL_ERROR)", async () => {
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    doc.branch("main").write((h) => h.getText("t").insert(0, "base"));

    const idxUpdates: Uint8Array[] = [];
    const unsub = repo.index().subscribeLocalUpdates((b) => idxUpdates.push(b));

    repo.createBranch("draft", "main");

    // Delivered synchronously (decorated auto-flush) — before any awaited turn.
    expect(idxUpdates.length).toBeGreaterThanOrEqual(1);
    expect(repo.branches().sort()).toEqual(["draft", "main"]);

    // The pending-event queue was flushed, so the scheduled microtask check finds no orphan.
    await drain();
    expect(noInternalError()).toBe(false);

    unsub();
  });

  it("a peer at the shared prefix learns the branch from the streamed createBranch delta alone", async () => {
    const a = new BranchingDocRepo();
    a.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "base"));

    // B is synced to A's pre-createBranch state (the causal prefix) — an incremental delta only
    // applies onto the prefix it descends from, exactly as the wire-surface delta test above.
    const b = new BranchingDocRepo();
    b.openDoc("G");
    b.index().import(a.index().export(UPDATE));
    b.openDoc("G").import(a.openDoc("G").export(UPDATE));
    expect(b.branches().sort()).toEqual(["main"]);

    // Capture ONLY the incremental createBranch index frame off the wire stream.
    const idxUpdates: Uint8Array[] = [];
    const unsub = a.index().subscribeLocalUpdates((by) => idxUpdates.push(by));
    a.createBranch("draft", "main");
    await drain();
    unsub();
    expect(idxUpdates.length).toBeGreaterThanOrEqual(1);
    expect(noInternalError()).toBe(false);

    // B converges on the new branch from the streamed delta — no full re-export needed.
    for (const bytes of idxUpdates) b.index().import(bytes);
    expect(b.branches().sort()).toEqual(["draft", "main"]);
  });

  // `deleteBranch` is CURRENTLY local-only registry cleanup: it drops the branch's lineage entry
  // WITHOUT committing a durable "discard" op (durable cross-peer deletion is a wrapper
  // lifecycle-log follow-up). So it emits NO local-update frame and — because nothing is enqueued
  // — trips no `[LORO_INTERNAL_ERROR]`. Its `index.ts` decoration is therefore harmless (a flush
  // of an empty queue) and future-proofs the method for when a durable delete op is added. This
  // test pins that contract so a later durable-delete change is a deliberate, visible break here.
  it("deleteBranch removes the branch locally, streams no frame, and trips no LORO_INTERNAL_ERROR", async () => {
    const repo = new BranchingDocRepo();
    repo.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "x"));
    repo.createBranch("draft", "main");

    const idxUpdates: Uint8Array[] = [];
    const unsub = repo.index().subscribeLocalUpdates((b) => idxUpdates.push(b));

    repo.deleteBranch("draft");
    await drain();

    expect(repo.branches().sort()).toEqual(["main"]);
    expect(idxUpdates.length).toBe(0); // local-only: no durable delete op on the shared log
    expect(noInternalError()).toBe(false);

    unsub();
  });
});
