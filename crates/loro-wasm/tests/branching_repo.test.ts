import { describe, expect, it } from "vitest";
import { BranchingDocRepo } from "../bundler/index";

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
