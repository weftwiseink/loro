import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BranchingDocRepo, type LoroEventBatch } from "../bundler/index";

// Proves branch data crosses the wasm boundary: a write reads back on its branch, a divergent
// branch stays isolated (copy-on-divergence), and `merge` converges.
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

    expect(doc.branch("draft").read((h) => h.getText("t").toString())).toBe(
      "main",
    );

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

    const text = doc.branch("main").read((h) => h.getText("t").toString());
    expect([...text].length).toBe(6);
    expect(text.startsWith("base")).toBe(true);
    expect(text.includes("M")).toBe(true);
    expect(text.includes("F")).toBe(true);
    expect(doc.contains("main", "feature")).toBe(true);
  });

  it("merge reports fast-forward and already-contained", () => {
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    doc.branch("main").write((h) => h.getText("t").insert(0, "x"));
    repo.createBranch("feature", "main");
    doc.branch("feature").write((h) => h.getText("t").insert(1, "y"));
    expect(doc.merge("main", "feature")).toBe("fast-forward");
    expect(doc.branch("main").read((h) => h.getText("t").toString())).toBe("xy");
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

// Cross-peer convergence: two repos sync via export/import (index bytes, then content bytes) and
// must end with identical branch state, no dropped or duplicated op.
describe("BranchingDocRepo cross-peer sync", () => {
  const UPDATE = { mode: "update" } as const;

  // Snapshot every export before any import, then cross-apply: index both ways, then content both
  // ways. One round converges even with concurrent branch creation, since index lands before
  // content.
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

    a.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "A"));
    b.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "B"));

    a.createBranch("featA", "main");
    a.openDoc("G").branch("featA").write((h) => h.getText("t").insert(1, "-fa"));
    b.createBranch("featB", "main");
    b.openDoc("G").branch("featB").write((h) => h.getText("t").insert(1, "-fb"));

    converge(a, b, ["G"]);

    expect(a.branches().sort()).toEqual(["featA", "featB", "main"]);
    expect(b.branches().sort()).toEqual(["featA", "featB", "main"]);

    const mainA = readBranch(a, "G", "main");
    const mainB = readBranch(b, "G", "main");
    expect(mainA).toBe(mainB);
    expect([...mainA].sort().join("")).toBe("AB"); // both ops, none duplicated

    for (const br of a.branches()) {
      expect(readBranch(a, "G", br)).toBe(readBranch(b, "G", br));
    }
    expect(readBranch(a, "G", "featA")).toBe("A-fa");
    expect(readBranch(a, "G", "featB")).toBe("B-fb");
  });

  it("createBranchAt at a single-peer frontier does NOT leak other-peer ops across the boundary", () => {
    const a = new BranchingDocRepo();
    a.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "A1"));
    const atA = a.openDoc("G").frontierOf("main");

    const b = new BranchingDocRepo();
    b.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "B1"));

    b.index().import(a.index().export(UPDATE));
    b.openDoc("G").import(a.openDoc("G").export(UPDATE));
    const mainB = readBranch(b, "G", "main");
    expect(mainB.includes("A1") && mainB.includes("B1")).toBe(true);

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

    expect(readBranch(b, "G", "feat")).toBe("base-feat");
    expect(b.openDoc("G").contains("main", "feat")).toBe(false);

    expect(b.openDoc("G").merge("main", "feat")).toBe("fast-forward");
    expect(b.openDoc("G").contains("main", "feat")).toBe(true);

    converge(a, b, ["G"]);
    expect(a.openDoc("G").contains("main", "feat")).toBe(true);
    expect(readBranch(a, "G", "main")).toBe(readBranch(b, "G", "main"));
    expect(readBranch(a, "G", "main")).toBe("base-feat");
  });
});

// Branch-scoped `subscribe`/`subscribeRoot` deliver change events to JS callbacks with no manual
// flush (the decoration is in `index.ts`).
describe("Branch subscriptions deliver change events", () => {
  it("subscribeRoot fires on a branch write and stops after unsubscribe", () => {
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    const branch = doc.branch("main");

    const events: LoroEventBatch[] = [];
    const unsubscribe = branch.subscribeRoot((e) => events.push(e));

    branch.write((h) => h.getText("t").insert(0, "hi"));

    expect(events.length).toBe(1);
    expect(events[0].by).toBe("local");
    expect(events[0].events.length).toBeGreaterThan(0);
    expect(events[0].events.some((ev) => ev.target === "cid:root-t:Text")).toBe(
      true,
    );

    unsubscribe();
    branch.write((h) => h.getText("t").insert(2, "!"));
    expect(events.length).toBe(1);
  });

  it("subscribe(containerId) fires for the scoped container on a write", () => {
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    const branch = doc.branch("main");

    branch.write((h) => h.getText("t").insert(0, "x"));
    const cid = branch.read((h) => h.getText("t").id);

    const events: LoroEventBatch[] = [];
    const unsubscribe = branch.subscribe(cid, (e) => events.push(e));

    branch.write((h) => h.getText("t").insert(1, "y"));

    expect(events.length).toBe(1);
    expect(events[0].by).toBe("local");
    expect(events[0].events[0].target).toBe("cid:root-t:Text");

    unsubscribe();
    branch.write((h) => h.getText("t").insert(2, "z"));
    expect(events.length).toBe(1);
  });

  it("a TRUE 3-way merge delivers a catch-up event to the target subscriber", () => {
    // Same-position edits force a real merge, dispatched as `by:"import"` (not a `checkout`).
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    doc.branch("main").write((h) => h.getText("t").insert(0, "base"));
    repo.createBranch("feature", "main");
    doc.branch("main").write((h) => h.getText("t").insert(4, "M"));
    doc.branch("feature").write((h) => h.getText("t").insert(4, "F"));
    expect(doc.contains("main", "feature")).toBe(false);

    const events: LoroEventBatch[] = [];
    const unsubscribe = doc.branch("main").subscribeRoot((e) => events.push(e));

    expect(doc.merge("main", "feature")).toBe("merged");

    expect(events.length).toBe(1);
    expect(events[0].by).toBe("import");
    expect(events[0].origin).toBe("advance");
    expect(events[0].events.some((ev) => ev.target === "cid:root-t:Text")).toBe(
      true,
    );
    expect(
      doc.branch("main").read((h) => [...h.getText("t").toString()].length),
    ).toBe(6);

    unsubscribe();
  });

  it("a FAST-FORWARD merge delivers a synthesized event to the target branch subscriber", () => {
    // The rebind-to-existing arm must still dispatch diff(X -> Y) to the `main` subscriber.
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    doc.branch("main").write((h) => h.getText("t").insert(0, "base"));
    repo.createBranch("feature", "main");
    doc.branch("feature").write((h) => h.getText("t").insert(4, "-ff"));
    expect(doc.contains("main", "feature")).toBe(false);

    const events: LoroEventBatch[] = [];
    const unsubscribe = doc.branch("main").subscribeRoot((e) => events.push(e));

    expect(doc.merge("main", "feature")).toBe("fast-forward");

    expect(events.length).toBe(1);
    expect(events[0].by).toBe("import");
    expect(events[0].events.some((ev) => ev.target === "cid:root-t:Text")).toBe(
      true,
    );
    expect(doc.branch("main").read((h) => h.getText("t").toString())).toBe(
      "base-ff",
    );

    unsubscribe();
  });

  it("an event realized inside `read` is delivered before `read` returns to JS", () => {
    // A move realized lazily on ACCESS: `BranchingIndex.import` advances a branch's frontier but
    // does not re-resolve open content docs, so catch-up happens at the next `read`. `Branch.read`
    // is decorated, so the event must flush at `read`'s exit rather than leak into a later
    // unrelated call.
    const UPDATE = { mode: "update" } as const;

    const a = new BranchingDocRepo();
    const da = a.openDoc("G");
    da.branch("main").write((h) => h.getText("t").insert(0, "base"));
    a.createBranch("feat", "main");
    da.branch("feat").read((h) => h.getText("t").toString()); // bind feat at "base"
    const lineageV1 = a.index().export(UPDATE);
    const contentV1 = da.export(UPDATE);
    da.branch("feat").write((h) => h.getText("t").insert(4, "-feat"));
    const lineageV2 = a.index().export(UPDATE);
    const contentV2 = da.export(UPDATE);

    // Peer B builds feat bound at "base", subscribes, then receives stage 2 content-then-lineage,
    // so the move stays pending until the read.
    const b = new BranchingDocRepo();
    const db = b.openDoc("G");
    b.index().import(lineageV1);
    db.import(contentV1);

    const events: LoroEventBatch[] = [];
    const unsubscribe = db.branch("feat").subscribeRoot((e) => events.push(e));
    db.branch("feat").read((h) => h.getText("t").toString());

    db.import(contentV2); // feat's index frontier is still v1, so no move yet
    b.index().import(lineageV2);
    const beforeRead = events.length;

    const text = db.branch("feat").read((h) => h.getText("t").toString());
    expect(text).toBe("base-feat");
    expect(events.length).toBeGreaterThan(beforeRead);
    expect(events.slice(beforeRead).every((e) => e.by === "import")).toBe(true);

    unsubscribe();
  });
});

// Wire surface (what weft's `EnvelopedAdaptor` drives sync off), exercised on a NON-MAIN branch,
// whose copy-on-divergence head a main-keyed tracking loop would miss: `subscribeLocalUpdates`
// must fire for it, and `version()` must reflect the shared oplog vv, not one head's.
describe("BranchingDoc/Index wire surface: subscribeLocalUpdates + version delta round-trip", () => {
  const UPDATE = { mode: "update" } as const;
  const readBranch = (repo: BranchingDocRepo, doc: string, branch: string) =>
    repo.openDoc(doc).branch(branch).read((h) => h.getText("t").toString());

  it("delta-syncs a NON-MAIN branch edit to a peer via the callback bytes and via export(from)", () => {
    const a = new BranchingDocRepo();
    const b = new BranchingDocRepo(); // converges via export({from}) deltas
    const c = new BranchingDocRepo(); // converges via the subscribeLocalUpdates callback bytes

    a.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "base"));
    a.createBranch("feat", "main");
    for (const peer of [b, c]) {
      peer.openDoc("G");
      peer.index().import(a.index().export(UPDATE));
      peer.openDoc("G").import(a.openDoc("G").export(UPDATE));
    }
    expect(b.branches().sort()).toEqual(["feat", "main"]);
    expect(readBranch(b, "G", "feat")).toBe("base");

    const beforeDoc = a.openDoc("G").version();
    const beforeIdx = a.index().version();
    const docUpdates: Uint8Array[] = [];
    const idxUpdates: Uint8Array[] = [];
    const unsubDoc = a.openDoc("G").subscribeLocalUpdates((bytes) => docUpdates.push(bytes));
    const unsubIdx = a.index().subscribeLocalUpdates((bytes) => idxUpdates.push(bytes));

    a.openDoc("G").branch("feat").write((h) => h.getText("t").insert(4, "-feat"));

    expect(docUpdates.length).toBeGreaterThan(0);
    expect(idxUpdates.length).toBeGreaterThan(0);

    const afterDoc = a.openDoc("G").version();
    expect(afterDoc.compare(beforeDoc)).toBe(1);
    expect(a.openDoc("G").oplogVersion().compare(beforeDoc)).toBe(1);
    expect(a.index().version().compare(beforeIdx)).toBe(1);

    const docDelta = a.openDoc("G").export({ mode: "update", from: beforeDoc });
    const idxDelta = a.index().export({ mode: "update", from: beforeIdx });
    expect(docDelta.length).toBeGreaterThan(0);
    expect(idxDelta.length).toBeGreaterThan(0);

    b.index().import(idxDelta);
    b.openDoc("G").import(docDelta);
    expect(readBranch(b, "G", "feat")).toBe("base-feat");
    expect(readBranch(b, "G", "main")).toBe("base"); // isolated

    // The callback bytes alone are valid deltas for a peer at `before`.
    unsubDoc();
    unsubIdx();
    for (const bytes of idxUpdates) c.index().import(bytes);
    for (const bytes of docUpdates) c.openDoc("G").import(bytes);
    expect(readBranch(c, "G", "feat")).toBe("base-feat");
    expect(readBranch(c, "G", "main")).toBe("base");
  });

  // NON-ECHO invariant: `subscribeLocalUpdates` fires on local edits only, never on `import`;
  // an echo would make every peer re-broadcast what it receives. Covers the content doc and the
  // index, including the index's `after_import` policy-resolve path. B's own edits still fire, so
  // a zero count on import is not a dead callback.
  it("import does NOT echo onto subscribeLocalUpdates (content doc + index); a local edit DOES", () => {
    const a = new BranchingDocRepo();
    const b = new BranchingDocRepo();

    a.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "base"));
    a.createBranch("feat", "main");
    a.openDoc("G").branch("feat").write((h) => h.getText("t").insert(4, "-feat"));

    b.openDoc("G");

    // Wire B's streams before importing so any echo from `import` is caught.
    const bDocUpdates: Uint8Array[] = [];
    const bIdxUpdates: Uint8Array[] = [];
    const unsubDoc = b.openDoc("G").subscribeLocalUpdates((bytes) => bDocUpdates.push(bytes));
    const unsubIdx = b.index().subscribeLocalUpdates((bytes) => bIdxUpdates.push(bytes));

    b.index().import(a.index().export(UPDATE));
    b.openDoc("G").import(a.openDoc("G").export(UPDATE));

    expect(bDocUpdates.length).toBe(0);
    expect(bIdxUpdates.length).toBe(0);

    expect(b.branches().sort()).toEqual(["feat", "main"]);
    expect(readBranch(b, "G", "feat")).toBe("base-feat");

    b.openDoc("G").branch("feat").write((h) => h.getText("t").insert(0, "x"));
    expect(bDocUpdates.length).toBeGreaterThan(0);

    expect(b.openDoc("G").merge("main", "feat")).toBe("fast-forward");
    expect(bIdxUpdates.length).toBeGreaterThan(0);

    unsubDoc();
    unsubIdx();
  });
});

// `createBranch` commits a lineage op that must reach `BranchingIndex.subscribeLocalUpdates` so a
// peer learns of a new branch incrementally. The op waits in the global pending-event queue until
// a decorated method flushes it (`index.ts`); undecorated, it streamed zero frames and logged
// `[LORO_INTERNAL_ERROR] Event not called`.
describe("BranchingDocRepo branch-lifecycle local-update streaming", () => {
  const UPDATE = { mode: "update" } as const;

  // Capture `console.error` so the `[LORO_INTERNAL_ERROR] Event not called` log is assertable.
  const consoleErrors: string[] = [];
  // Structural type sidesteps vitest's version-specific `MockInstance` generic without `any`.
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

  // Drain a turn so `schedule_pending_event_check`'s promise callback runs.
  const drain = () => new Promise((r) => setTimeout(r, 0));

  it("createBranch streams its lineage op on the index stream with NO manual flush (no LORO_INTERNAL_ERROR)", async () => {
    const repo = new BranchingDocRepo();
    const doc = repo.openDoc("G");
    doc.branch("main").write((h) => h.getText("t").insert(0, "base"));

    const idxUpdates: Uint8Array[] = [];
    const unsub = repo.index().subscribeLocalUpdates((b) => idxUpdates.push(b));

    repo.createBranch("draft", "main");

    expect(idxUpdates.length).toBeGreaterThanOrEqual(1);
    expect(repo.branches().sort()).toEqual(["draft", "main"]);

    await drain();
    expect(noInternalError()).toBe(false);

    unsub();
  });

  it("a peer at the shared prefix learns the branch from the streamed createBranch delta alone", async () => {
    const a = new BranchingDocRepo();
    a.openDoc("G").branch("main").write((h) => h.getText("t").insert(0, "base"));

    // B is at A's pre-createBranch state; an incremental delta only applies onto its causal prefix.
    const b = new BranchingDocRepo();
    b.openDoc("G");
    b.index().import(a.index().export(UPDATE));
    b.openDoc("G").import(a.openDoc("G").export(UPDATE));
    expect(b.branches().sort()).toEqual(["main"]);

    const idxUpdates: Uint8Array[] = [];
    const unsub = a.index().subscribeLocalUpdates((by) => idxUpdates.push(by));
    a.createBranch("draft", "main");
    await drain();
    unsub();
    expect(idxUpdates.length).toBeGreaterThanOrEqual(1);
    expect(noInternalError()).toBe(false);

    for (const bytes of idxUpdates) b.index().import(bytes);
    expect(b.branches().sort()).toEqual(["draft", "main"]);
  });

  // `deleteBranch` is local-only registry cleanup: it commits no durable "discard" op, so it emits
  // no local-update frame and trips no `[LORO_INTERNAL_ERROR]`. Pins that contract so adding a
  // durable delete op is a deliberate, visible break here.
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

// Derived read-only index projections: tips, forkPoint, docsModifiedOnBranch.
describe("BranchingIndex derived reads", () => {
  it("tips / forkPoint / docsModifiedOnBranch cross the wasm boundary", () => {
    const repo = new BranchingDocRepo();
    const g = repo.openDoc("G");
    const h = repo.openDoc("H");
    g.branch("main").write((doc) => doc.getText("t").insert(0, "base")); // main touches G
    repo.createBranch("feat", "main");
    h.branch("feat").write((doc) => doc.getText("t").insert(0, "hi")); // feat touches H (NOT G)

    const idx = repo.index();

    const tips = idx.tips();
    expect(Object.keys(tips).sort()).toEqual(["feat", "main"]);
    for (const b of ["main", "feat"]) {
      expect(Array.isArray(tips[b])).toBe(true);
      expect(tips[b].length).toBeGreaterThanOrEqual(1);
      expect(typeof tips[b][0].peer).toBe("string");
      expect(typeof tips[b][0].counter).toBe("number");
    }

    // docsModifiedOnBranch = docs the branch itself changed since forking (feat inherits G
    // unchanged).
    expect(idx.docsModifiedOnBranch("main")).toEqual(["G"]);
    expect(idx.docsModifiedOnBranch("feat")).toEqual(["H"]);
    expect(idx.docsModifiedOnBranch("nope")).toEqual([]);

    expect(idx.forkPoint("main")).toEqual([]);
    expect(idx.forkPoint("feat").length).toBeGreaterThanOrEqual(1);
    expect(typeof idx.forkPoint("feat")[0].peer).toBe("string");
  });
});
