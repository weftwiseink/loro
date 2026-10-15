//! Cost harness for the index import and per-record read paths. Every test is
//! `#[ignore]`: run by hand, in release, one thread, and read the printed rows.
//!
//! `cargo test --release -p loro-internal --lib multi_head::bench -- --ignored --nocapture --test-threads=1`
//!
//! Shapes follow `cdocs/proposals/2026-10-11-loro-branch-doc-lookup-cost.md`
//! (weftwise): N notes recorded on `main`, R resident (hydrated) docs, B
//! branches, one agent commit on `tidy`.
use super::*;
use crate::encoding::ExportMode;
use crate::version::Frontiers;
use loro_common::{Counter, LoroResult, PeerID, ID};
use std::sync::Arc;
use std::time::Instant;

fn guid(i: usize) -> DocId {
    format!("n{i:05}").into()
}

fn body(i: usize) -> String {
    let mut s = format!("# Note {i}\n\n");
    while s.len() < 2000 {
        s.push_str("the of and to in is that for it as with was on be by this are or from at\n");
    }
    s
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Resident set size in MiB from `/proc/self/status`, or `None` off Linux.
fn rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: f64 = status
        .lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    Some(kb / 1024.0)
}

/// Seed: N docs recorded on `main`, branches `tidy` + `x0..x{extra-1}` forked from
/// `main`. Returns the repo, its index update bytes, and per-doc content bytes.
fn seed(n: usize, extra: usize) -> (BranchingDocRepo, Vec<u8>, Vec<Vec<u8>>) {
    let repo = BranchingDocRepo::open().unwrap();
    let mut contents = Vec::with_capacity(n);
    for i in 0..n {
        let d = repo.open_doc(guid(i));
        let s = body(i);
        d.branch(GENESIS_BRANCH)
            .write(|h| h.get_text("content").insert_unicode(0, &s).unwrap())
            .unwrap();
        contents.push(d.export(ExportMode::all_updates()).unwrap());
    }
    repo.create_branch(&"tidy".into(), &GENESIS_BRANCH.into())
        .unwrap();
    for b in 0..extra {
        repo.create_branch(&format!("x{b}").into(), &GENESIS_BRANCH.into())
            .unwrap();
    }
    let idx = repo.index().export(ExportMode::all_updates()).unwrap();
    (repo, idx, contents)
}

/// A renderer: a fresh repo holding the index and `resident` hydrated docs.
fn renderer(
    idx: &[u8],
    contents: &[Vec<u8>],
    resident: usize,
) -> (BranchingDocRepo, Vec<Arc<BranchingDoc>>) {
    let r = BranchingDocRepo::open().unwrap();
    r.index().import(idx).unwrap();
    let mut docs = Vec::new();
    for (i, c) in contents.iter().enumerate().take(resident) {
        let d = r.open_doc(guid(i));
        d.import(c).unwrap();
        docs.push(d);
    }
    (r, docs)
}

/// One agent commit on `tidy` to the last doc; returns the lineage update only.
/// Exported from the SEED index version, not the agent's own post-open version, so
/// the agent's genesis marker rides along (as a real `export({ mode: "update",
/// from: receiverVersion })` does); exported from the agent's version the frame
/// can only land `pending`.
fn agent_update(idx: &[u8], contents: &[Vec<u8>], n: usize) -> Vec<u8> {
    let a = BranchingDocRepo::open().unwrap();
    a.index().import(idx).unwrap();
    let d = a.open_doc(guid(n - 1));
    d.import(&contents[n - 1]).unwrap();
    let base = {
        let t = IndexDoc::new(SelfRooted::new());
        t.import(idx).unwrap();
        t.oplog_vv()
    };
    d.branch("tidy")
        .write(|h| {
            h.get_text("content")
                .insert_unicode(0, "agent line\n")
                .unwrap()
        })
        .unwrap();
    a.index().export(ExportMode::updates_owned(base)).unwrap()
}

/// The whole-index deep-value read of one record: the reference `recorded_ids`
/// is measured and checked against. Test-only.
pub(super) fn recorded_ids_deep(idx: &IndexDoc, b: &BranchId, doc: &DocId) -> LoroResult<Vec<ID>> {
    idx.read(b, |d| {
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

/// Index-only seeding: N heads recorded on `main`, then `tidy`, `x0`, `x1`.
fn index_only(n: usize) -> IndexDoc {
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&GENESIS_BRANCH.into()).unwrap();
    for i in 0..n {
        idx.record_head(&GENESIS_BRANCH.into(), &guid(i), 7, i as Counter)
            .unwrap();
    }
    idx.create_index_branch(&"tidy".into(), &GENESIS_BRANCH.into())
        .unwrap();
    idx.create_index_branch(&"x0".into(), &GENESIS_BRANCH.into())
        .unwrap();
    idx.create_index_branch(&"x1".into(), &GENESIS_BRANCH.into())
        .unwrap();
    idx
}

/// S1: per-record read, `recorded_ids` against the deep-value reference, on the
/// root head (`main`) and a copied head (`tidy`).
#[test]
#[ignore]
fn bench_s1_lookup_cost() {
    for &n in &[400usize, 1000, 2125] {
        let idx = index_only(n);
        let reps = 20;
        let d0 = guid(0);
        let main = BranchId::from(GENESIS_BRANCH);
        let tidy = BranchId::from("tidy");
        let t = Instant::now();
        for _ in 0..reps {
            idx.recorded_ids(&main, &d0).unwrap();
        }
        let cur_main = ms(t) / reps as f64;
        let t = Instant::now();
        for _ in 0..reps {
            idx.recorded_ids(&tidy, &d0).unwrap();
        }
        let cur_tidy = ms(t) / reps as f64;
        let t = Instant::now();
        for _ in 0..reps {
            recorded_ids_deep(&idx, &main, &d0).unwrap();
        }
        let deep_main = ms(t) / reps as f64;
        let t = Instant::now();
        for _ in 0..reps {
            recorded_ids_deep(&idx, &tidy, &d0).unwrap();
        }
        let deep_tidy = ms(t) / reps as f64;
        assert_eq!(
            idx.recorded_ids(&tidy, &d0).unwrap(),
            recorded_ids_deep(&idx, &tidy, &d0).unwrap()
        );
        println!(
            "S1 n={n} recorded_ids(main)={cur_main:.4}ms recorded_ids(tidy)={cur_tidy:.4}ms deep(main)={deep_main:.3}ms deep(tidy)={deep_tidy:.3}ms"
        );
    }
}

/// S3: one lineage import, `import_index` against the bare `index().import`, at
/// the report's shapes. Ceiling: 5x the bare import at 400/50/32 and 1000/1000.
#[test]
#[ignore]
fn bench_s3_import_cost() {
    let mut over: Vec<String> = Vec::new();
    for &(n, r, extra) in &[
        (400usize, 50usize, 0usize),
        (400, 50, 30),
        (1000, 50, 0),
        (1000, 1000, 0),
        (2125, 50, 0),
    ] {
        let (_seed, idx, contents) = seed(n, extra);
        let upd = agent_update(&idx, &contents, n);
        let (r1, _d1) = renderer(&idx, &contents, r);
        let (r2, _d2) = renderer(&idx, &contents, r);
        let branches = r1.branches().len();
        let t = Instant::now();
        r1.import_index(&upd).unwrap();
        let full = ms(t);
        let t = Instant::now();
        r2.index().import(&upd).unwrap();
        let bare = ms(t);
        println!(
            "S3 n={n} resident={r} branches={branches} updBytes={} import_index={full:.2}ms index.import={bare:.3}ms",
            upd.len()
        );
        let ceiling = (n, r, extra) == (400, 50, 30) || (n, r, extra) == (1000, 1000, 0);
        if ceiling && full > 5.0 * bare.max(0.1) {
            over.push(format!(
                "n={n} r={r} b={branches}: import_index {full:.2}ms > 5x bare {bare:.3}ms"
            ));
        }
    }
    assert!(over.is_empty(), "over the 5x ceiling: {over:?}");
}

/// S3h: a long index history (50 commits/doc, ~20k index ops) must not change the
/// import cost: the tip-range scan is O(ops in the range), not O(history).
#[test]
#[ignore]
fn bench_s3h_long_history() {
    let n = 400;
    let repo = BranchingDocRepo::open().unwrap();
    let mut contents = Vec::new();
    for i in 0..n {
        let d = repo.open_doc(guid(i));
        for c in 0..50 {
            d.branch(GENESIS_BRANCH)
                .write(|h| {
                    h.get_text("content")
                        .insert_unicode(0, &format!("c{c}\n"))
                        .unwrap()
                })
                .unwrap();
        }
        contents.push(d.export(ExportMode::all_updates()).unwrap());
    }
    repo.create_branch(&"tidy".into(), &GENESIS_BRANCH.into())
        .unwrap();
    let idx = repo.index().export(ExportMode::all_updates()).unwrap();
    let upd = agent_update(&idx, &contents, n);
    let (r1, _d1) = renderer(&idx, &contents, 50);
    let (r2, _d2) = renderer(&idx, &contents, 50);
    let t = Instant::now();
    r1.import_index(&upd).unwrap();
    let full = ms(t);
    let t = Instant::now();
    r2.index().import(&upd).unwrap();
    let bare = ms(t);
    println!(
        "S3h n={n} resident=50 branches=2 index_ops~{} import_index={full:.2}ms index.import={bare:.3}ms",
        n * 50
    );
}

/// S4: the other mount-scaled paths: hydrate (a content import, B resolves), the
/// eager index branch copy, `docs_modified_on_branch`; then history length.
#[test]
#[ignore]
fn bench_s4_other_paths() {
    for &(n, extra) in &[(400usize, 0usize), (400, 30), (2125, 0)] {
        let (seed_repo, idx, contents) = seed(n, extra);
        let r = BranchingDocRepo::open().unwrap();
        r.index().import(&idx).unwrap();
        let b = r.branches().len();
        let k = 20;
        let t = Instant::now();
        for (i, c) in contents.iter().enumerate().take(k) {
            r.open_doc(guid(i)).import(c).unwrap();
        }
        let hyd = ms(t) / k as f64;
        let t = Instant::now();
        seed_repo
            .create_branch(&"y".into(), &GENESIS_BRANCH.into())
            .unwrap();
        let cb = ms(t);
        let t = Instant::now();
        let m = seed_repo
            .index()
            .docs_modified_on_branch(&"tidy".into())
            .unwrap();
        let dm = ms(t);
        println!(
            "S4 n={n} branches={b} hydrate/doc={hyd:.3}ms create_index_branch={cb:.3}ms docs_modified_on_branch(tidy)={dm:.3}ms ({} docs)",
            m.len()
        );
    }
    for &commits in &[1usize, 10] {
        let repo = BranchingDocRepo::open().unwrap();
        for i in 0..400 {
            let d = repo.open_doc(guid(i));
            for c in 0..commits {
                d.branch(GENESIS_BRANCH)
                    .write(|h| {
                        h.get_text("content")
                            .insert_unicode(0, &format!("c{c}\n"))
                            .unwrap()
                    })
                    .unwrap();
            }
        }
        repo.create_branch(&"tidy".into(), &GENESIS_BRANCH.into())
            .unwrap();
        repo.open_doc(guid(0))
            .branch("tidy")
            .write(|h| h.get_text("content").insert_unicode(0, "t\n").unwrap())
            .unwrap();
        let t = Instant::now();
        let m = repo
            .index()
            .docs_modified_on_branch(&"tidy".into())
            .unwrap();
        let dm = ms(t);
        let t = Instant::now();
        let _ = repo.index().recorded_ids(&"tidy".into(), &guid(0)).unwrap();
        let rl = ms(t);
        println!(
            "S4h n=400 commits/doc={commits} index_ops~{} docs_modified_on_branch={dm:.3}ms ({} docs) recorded_ids={rl:.3}ms",
            400 * commits,
            m.len()
        );
    }
}

/// RSS: the memory of the eager per-branch index heads, 32 branches at 2125
/// notes (index-only seeding), as a resident-set delta. No threshold: recorded.
#[test]
#[ignore]
fn bench_rss_eager_index_branches() {
    let n = 2125;
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&GENESIS_BRANCH.into()).unwrap();
    for i in 0..n {
        idx.record_head(&GENESIS_BRANCH.into(), &guid(i), 7, i as Counter)
            .unwrap();
    }
    let before = rss_mib();
    let t = Instant::now();
    for b in 0..32 {
        idx.create_index_branch(&format!("x{b}").into(), &GENESIS_BRANCH.into())
            .unwrap();
    }
    let dt = ms(t);
    // Touch every branch head so lazily built state is counted.
    for b in 0..32 {
        let _ = idx.recorded_ids(&format!("x{b}").into(), &guid(0)).unwrap();
    }
    let after = rss_mib();
    match (before, after) {
        (Some(b), Some(a)) => println!(
            "RSS n={n} branches=32 before={b:.1}MiB after={a:.1}MiB delta={:.1}MiB per_branch={:.2}MiB create_x32={dt:.1}ms",
            a - b,
            (a - b) / 32.0
        ),
        _ => println!("RSS n={n}: /proc/self/status unavailable on this platform"),
    }
}

/// S5: fail-safe and already-contained semantics at a small shape (asserted, not
/// timed): an unresolvable tip is `Every`; a repeated frame moves nothing.
#[test]
#[ignore]
fn bench_s5_failsafe_and_contained() {
    let n = 5;
    let (_seed, idx, contents) = seed(n, 0);
    let (r, docs) = renderer(&idx, &contents, n);
    let main = BranchId::from(GENESIS_BRANCH);
    let tidy = BranchId::from("tidy");
    let bogus = Frontiers::from_id(ID::new(999_999, 7));
    let tips = r.index().tips();
    assert_eq!(
        r.index().docs_changed_between(&tips[&main], &bogus),
        RecordScope::Every
    );
    assert_eq!(
        r.index().docs_changed_between(&bogus, &tips[&main]),
        RecordScope::Every
    );
    let upd = agent_update(&idx, &contents, n);
    let o1 = r.import_index(&upd).unwrap();
    assert!(!o1.moved.is_empty());
    let heads: Vec<_> = docs.iter().map(|d| d.bound_head(&tidy)).collect();
    let o2 = r.import_index(&upd).unwrap();
    assert!(o2.moved.is_empty(), "already contained: moved is empty");
    assert_eq!(o2.status.success.iter().count(), 0);
    let heads2: Vec<_> = docs.iter().map(|d| d.bound_head(&tidy)).collect();
    assert_eq!(heads, heads2);
    println!(
        "S5 moved(first)={:?} moved(repeat)={:?}",
        o1.moved, o2.moved
    );
}
