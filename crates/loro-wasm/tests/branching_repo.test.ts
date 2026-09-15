import { describe, expect, it } from "vitest";
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
});
