//! Behavior-parity harness: a branch-surfaced doc emits the same subscription
//! events for every update as a plain `LoroDoc` would.
//!
//! The governing principle (see
//! `cdocs/proposals/2026-09-18-loro-fork-multihead-subscription-parity.md`):
//! a `LoroDoc` surfaced over a `MultiHeadDoc` branch MUST deliver the same
//! `subscribe`/`subscribe_root` events for every update as a plain `LoroDoc`.
//!
//! Each case drives a `Plain` reference (a `LoroDoc`) and a `Branched` surface
//! (one branch of a `MultiHeadDoc<Manual>`) through matching operations and
//! asserts the recorded event rows match. Every recorded event is normalized to
//! an [`EventRow`] `{ by, origin, touched, mirror_after }` where `mirror_after`
//! is a `LoroValue` mirror rebuilt by applying each `ContainerDiff` along its
//! path exactly as `tests/test.rs::test_checkout` does. Frontiers are NOT
//! compared directly (heads mint their own peers, so `from`/`to` differ by
//! construction); the mirror is the property the frontiers exist to guarantee.
//!
//! `by`/`origin` are gated to the dedicated M9 cases (cause threading is a later
//! phase; every earlier phase must stay independently green, so its cases assert
//! only the delivery discriminator: count, order, `touched`, `mirror_after`).
//! The RED cases are `#[ignore]`d and reproduced on demand with
//! `cargo test -p loro-internal --test multi_head_parity -- --ignored`.

use std::sync::{Arc, Mutex};

use loro_internal::cursor::PosType;
use loro_internal::encoding::ExportMode;
use loro_internal::event::{DiffEvent, EventTriggerKind, Index, Path};
use loro_internal::multi_head::{
    BranchId, BranchSubscription, Intent, Manual, MultiHeadDoc, ResolveCause,
};
use loro_internal::version::Frontiers;
use loro_internal::{ApplyDiff, LoroDoc, LoroValue, TextHandler};

fn b(s: &str) -> BranchId {
    s.into()
}

// ---------------------------------------------------------------------------
// Recorder: normalizes every delivered event into an EventRow.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct EventRow {
    by: EventTriggerKind,
    origin: String,
    /// Sorted container paths touched by this event (paths are content-derived,
    /// so they match across surfaces even though peer ids / frontiers differ).
    touched: Vec<Vec<Index>>,
    /// The `LoroValue` mirror after applying this event's diffs.
    mirror_after: LoroValue,
}

#[derive(Clone)]
struct Recorder {
    rows: Arc<Mutex<Vec<EventRow>>>,
    mirror: Arc<Mutex<LoroValue>>,
}

impl Recorder {
    fn new() -> Self {
        Recorder {
            rows: Arc::new(Mutex::new(Vec::new())),
            mirror: Arc::new(Mutex::new(LoroValue::Map(Default::default()))),
        }
    }

    /// A root subscriber that records normalized rows.
    fn root_subscriber(&self) -> Arc<dyn for<'a> Fn(DiffEvent<'a>) + Send + Sync> {
        let rows = self.rows.clone();
        let mirror = self.mirror.clone();
        Arc::new(move |ev: DiffEvent| {
            let mut m = mirror.lock().unwrap();
            let mut touched: Vec<Vec<Index>> = Vec::new();
            for cd in ev.events {
                let path: Vec<Index> = cd.path.iter().map(|x| x.1.clone()).collect();
                let p: Path = path.iter().cloned().collect();
                m.apply(&p, std::slice::from_ref(&cd.diff));
                touched.push(path);
            }
            touched.sort_by_key(|p| format!("{p:?}"));
            rows.lock().unwrap().push(EventRow {
                by: ev.event_meta.by,
                origin: ev.event_meta.origin.to_string(),
                touched,
                mirror_after: m.clone(),
            });
        })
    }

    fn rows(&self) -> Vec<EventRow> {
        self.rows.lock().unwrap().clone()
    }

    fn len(&self) -> usize {
        self.rows.lock().unwrap().len()
    }
}

// ---------------------------------------------------------------------------
// Comparison.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct CompareOpts {
    /// Compare `by` and `origin` too (the M9 / cause-threading cases).
    compare_by_origin: bool,
}

fn assert_parity(plain: &Recorder, branched: &Recorder, opts: CompareOpts) {
    let pr = plain.rows();
    let br = branched.rows();
    assert_eq!(
        pr.len(),
        br.len(),
        "event count mismatch: plain={} branched={}\nplain={pr:#?}\nbranched={br:#?}",
        pr.len(),
        br.len()
    );
    for (i, (p, b)) in pr.iter().zip(br.iter()).enumerate() {
        assert_eq!(
            p.touched, b.touched,
            "row {i}: touched-path mismatch\nplain={p:#?}\nbranched={b:#?}"
        );
        assert_eq!(
            p.mirror_after, b.mirror_after,
            "row {i}: mirror-after mismatch\nplain={p:#?}\nbranched={b:#?}"
        );
        if opts.compare_by_origin {
            assert_eq!(
                p.by, b.by,
                "row {i}: `by` mismatch (plain={:?} branched={:?})",
                p.by, b.by
            );
            assert_eq!(
                p.origin, b.origin,
                "row {i}: `origin` mismatch (plain={:?} branched={:?})",
                p.origin, b.origin
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Surface construction helpers.
// ---------------------------------------------------------------------------

/// A plain reference doc with a root recorder installed.
fn plain() -> (LoroDoc, Recorder, loro_internal::Subscription) {
    let doc = LoroDoc::new_auto_commit();
    let rec = Recorder::new();
    let sub = doc.subscribe_root(rec.root_subscriber());
    (doc, rec, sub)
}

/// A branched surface: `main` bound to the root head of a `MultiHeadDoc<Manual>`,
/// with a rebind-surviving recorder on `main`.
struct Branched {
    md: MultiHeadDoc<Manual>,
    rec: Recorder,
    _sub: BranchSubscription,
}

impl Branched {
    fn open() -> Self {
        let md = MultiHeadDoc::new(Manual::new());
        md.bind(&b("main"), md.root_head_id());
        let rec = Recorder::new();
        let sub = md
            .subscribe_branch(&b("main"), None, rec.root_subscriber())
            .unwrap();
        Branched {
            md,
            rec,
            _sub: sub,
        }
    }

    /// Resolve `main` to `target` via the access path (drives the transition +
    /// its dispatch); an access-realized move tags `("resolve", Import)`.
    fn resolve_main_to(&self, target: Frontiers) {
        self.md.policy().set_target(&b("main"), target);
        self.md.resolve(&b("main"), Intent::Read).unwrap();
    }

    /// Resolve `main` to `target` under an explicit cause (the import path tags
    /// `(origin, Import)`, mirroring `after_import`'s realization of a move).
    fn resolve_main_with(&self, target: Frontiers, cause: ResolveCause) {
        self.md.policy().set_target(&b("main"), target);
        self.md.resolve_with(&b("main"), Intent::Read, cause).unwrap();
    }
}

/// Union of two frontiers' ids; `checkout` shrinks it back to a valid frontier.
fn union(a: &Frontiers, b: &Frontiers) -> Frontiers {
    let mut ids: Vec<_> = a.iter().collect();
    ids.extend(b.iter());
    Frontiers::from(ids)
}

/// A remote peer: a plain `LoroDoc` whose exports both surfaces import.
fn remote() -> LoroDoc {
    LoroDoc::new_auto_commit()
}

fn all_updates(doc: &LoroDoc) -> Vec<u8> {
    doc.export(ExportMode::all_updates()).unwrap()
}

fn all_updates_md(md: &MultiHeadDoc<Manual>) -> Vec<u8> {
    md.export(ExportMode::all_updates()).unwrap()
}

fn write_map_str(d: &LoroDoc, map: &str, key: &str, val: &str) {
    d.get_map(map).insert(key, val).unwrap();
}

// ===========================================================================
// Cases
// ===========================================================================

// M1 local write: a local commit emits `Local` on both surfaces (no transition).
#[test]
fn m1_local_write() {
    let (doc, prec, _s) = plain();
    write_map_str(&doc, "map", "a", "1");
    doc.commit_then_renew();

    let br = Branched::open();
    br.md
        .write(&b("main"), |d| d.get_map("map").insert("a", "1").unwrap())
        .unwrap();

    assert_parity(&prec, &br.rec, CompareOpts::default());
    assert_eq!(prec.len(), 1, "one local event");
}

// M2 (i) import fast-forward, fresh doc, `main` alone: catch-up on the pinned
// ROOT head at refs == 1. GREEN on baseline.
#[test]
fn m2i_import_fast_forward_main_alone() {
    let r = remote();
    write_map_str(&r, "map", "a", "1");
    r.commit_then_renew();
    let bytes = all_updates(&r);

    let (doc, prec, _s) = plain();
    doc.import(&bytes).unwrap();

    let br = Branched::open();
    br.md.import(&bytes).unwrap();
    let target = r.state_frontiers();
    br.resolve_main_to(target);

    assert_parity(&prec, &br.rec, CompareOpts::default());
    assert_eq!(prec.len(), 1, "one import event");
}

// M2 (ii) import fast-forward with a bound sibling: the root is shared at
// refs == 2, so the move takes the copy+advance arm. RED on baseline (the jump
// diff is discarded on the non-recording fresh copy).
#[test]
fn m2ii_import_fast_forward_shared_root() {
    let r = remote();
    write_map_str(&r, "map", "a", "1");
    r.commit_then_renew();
    let bytes = all_updates(&r);

    let (doc, prec, _s) = plain();
    doc.import(&bytes).unwrap();

    let br = Branched::open();
    // Bind a second branch to the root so main's head is shared (refs == 2).
    br.md.create_branch(&b("draft"), &b("main")).unwrap();
    br.md.import(&bytes).unwrap();
    let target = r.state_frontiers();
    br.resolve_main_to(target);

    assert_parity(&prec, &br.rec, CompareOpts::default());
    assert_eq!(prec.len(), 1);
}

// M3 import with (disjoint) divergence, uniquely-owned head: catch-up arm.
// GREEN on baseline.
#[test]
fn m3_import_divergence_catchup() {
    let r = remote();
    write_map_str(&r, "map", "remote", "R");
    r.commit_then_renew();
    let bytes = all_updates(&r);

    let (doc, prec, _s) = plain();
    write_map_str(&doc, "map", "local", "L");
    doc.commit_then_renew();
    doc.import(&bytes).unwrap();

    let br = Branched::open();
    br.md
        .write(&b("main"), |d| d.get_map("map").insert("local", "L").unwrap())
        .unwrap();
    let local_tip = br.md.head_tip(br.md.bound_head(&b("main")).unwrap()).unwrap();
    br.md.import(&bytes).unwrap();
    br.resolve_main_to(union(&local_tip, &r.state_frontiers()));

    assert_parity(&prec, &br.rec, CompareOpts::default());
}

// M4 new container created remotely, shared root (copy+advance). RED on baseline.
#[test]
fn m4_new_container_remote_shared() {
    let r = remote();
    {
        let child = r
            .get_map("map")
            .insert_container("child", TextHandler::new_detached())
            .unwrap();
        child.insert(0, "x", PosType::Unicode).unwrap();
    }
    r.commit_then_renew();
    let bytes = all_updates(&r);

    let (doc, prec, _s) = plain();
    doc.import(&bytes).unwrap();

    let br = Branched::open();
    br.md.create_branch(&b("draft"), &b("main")).unwrap();
    br.md.import(&bytes).unwrap();
    br.resolve_main_to(r.state_frontiers());

    assert_parity(&prec, &br.rec, CompareOpts::default());
}

// M5 delete, catch-up arm. GREEN on baseline.
#[test]
fn m5_delete_catchup() {
    // Remote sets then deletes a key; we import the whole history and fast-forward.
    let r = remote();
    write_map_str(&r, "map", "a", "1");
    r.commit_then_renew();
    r.get_map("map").delete("a").unwrap();
    r.commit_then_renew();
    let bytes = all_updates(&r);

    let (doc, prec, _s) = plain();
    doc.import(&bytes).unwrap();

    let br = Branched::open();
    br.md.import(&bytes).unwrap();
    br.resolve_main_to(r.state_frontiers());

    assert_parity(&prec, &br.rec, CompareOpts::default());
}

// M6 sub-container (nested map.child text) edit: the dispatch must reproduce the
// nested-container diff and its container path. GREEN on baseline (catch-up). The
// root recorder sees the child edit via ancestor match, so path parity here
// exercises the same `emit_inner` ancestor/path machinery `subscribe(cid)` uses.
#[test]
fn m6_subcontainer_filter_catchup() {
    // Remote creates map.child (text) and edits it.
    let r = remote();
    {
        let child = r
            .get_map("map")
            .insert_container("child", TextHandler::new_detached())
            .unwrap();
        child.insert(0, "x", PosType::Unicode).unwrap();
    }
    r.commit_then_renew();
    let bytes = all_updates(&r);

    let (doc, prec, _s) = plain();
    doc.import(&bytes).unwrap();

    let br = Branched::open();
    br.md.import(&bytes).unwrap();
    br.resolve_main_to(r.state_frontiers());

    assert_parity(&prec, &br.rec, CompareOpts::default());
}

// M7 sibling shares the head, remote writes on main: copy+advance (refs == 2).
// RED on baseline (0 events while state advances).
#[test]
fn m7_sibling_shares_head_copy_advance() {
    let r = remote();
    write_map_str(&r, "map", "a", "1");
    r.commit_then_renew();
    let bytes = all_updates(&r);

    let (doc, prec, _s) = plain();
    doc.import(&bytes).unwrap();

    let br = Branched::open();
    // draft shares main's head (free share) -> refs == 2 on the root.
    br.md.create_branch(&b("draft"), &b("main")).unwrap();
    assert_eq!(br.md.refs_of(&b("main")), Some(2), "root shared");
    br.md.import(&bytes).unwrap();
    br.resolve_main_to(r.state_frontiers());

    assert_parity(&prec, &br.rec, CompareOpts::default());
    assert_eq!(prec.len(), 1);
}

// Build a branched surface where `draft` has diverged (its own head at tip Td)
// and `main` still sits on the empty root; returns (Branched, Td, draft_op_bytes).
fn setup_rebind_to_existing() -> (Branched, Frontiers, Vec<u8>) {
    let br = Branched::open();
    br.md.create_branch(&b("draft"), &b("main")).unwrap();
    // draft diverges (copy-on-divergence -> draft's own head).
    br.md
        .write(&b("draft"), |d| d.get_map("map").insert("d", "1").unwrap())
        .unwrap();
    let draft_head = br.md.bound_head(&b("draft")).unwrap();
    let td = br.md.head_tip(draft_head).unwrap();
    let bytes = all_updates_md(&br.md);
    (br, td, bytes)
}

// M8 rebind-to-existing: `main` moves onto `draft`'s existing head (by_tip hit).
// RED on baseline (rebind is a pointer swap; no diff is ever computed).
#[test]
fn m8_rebind_to_existing() {
    let (br, td, bytes) = setup_rebind_to_existing();

    // Plain analog: import draft's op -> one event.
    let (doc, prec, _s) = plain();
    doc.import(&bytes).unwrap();

    // main rebinds onto draft's head at Td (by_tip[Td] hit).
    br.resolve_main_to(td);

    assert_parity(&prec, &br.rec, CompareOpts::default());
    assert_eq!(prec.len(), 1, "one event for the fast-forward");
}

// M9 cause-threading: `by`/`origin` compared. Three sub-cases per the matrix.
// RED on baseline (`Checkout`/`"checkout"` where a plain doc emits `Import`/"").

// M9a: import case -> Import + "" (strict, compared against the plain analog).
#[test]
fn m9a_by_origin_import() {
    let r = remote();
    write_map_str(&r, "map", "a", "1");
    r.commit_then_renew();
    let bytes = all_updates(&r);

    let (doc, prec, _s) = plain();
    doc.import(&bytes).unwrap();

    let br = Branched::open();
    br.md.import(&bytes).unwrap();
    // The import path realizes the move with the Import cause (origin ""),
    // exactly as `MultiHeadDoc::import`'s after_import loop does.
    br.resolve_main_with(r.state_frontiers(), ResolveCause::Import { origin: "".into() });

    assert_parity(
        &prec,
        &br.rec,
        CompareOpts {
            compare_by_origin: true,
        },
    );
    // The branched event must be tagged Import, not Checkout.
    assert_eq!(br.rec.rows()[0].by, EventTriggerKind::Import);
    assert_eq!(br.rec.rows()[0].origin, "");
}

// M9b: rebind-to-existing via the access (`resolve`) path -> Import + pinned
// "resolve" origin (per the M4 cause mapping: Access -> ("resolve", Import)).
// The "advance" origin (the merge/advance cause) is covered by a unit test in
// `multi_head/tests.rs` (`resolve_with_advance_tags_import_advance`) since the
// Manual harness drives moves through `resolve` (Access), not `merge`.
#[test]
fn m9b_by_origin_resolve() {
    let (br, td, _bytes) = setup_rebind_to_existing();
    br.resolve_main_to(td);
    let rows = br.rec.rows();
    assert_eq!(rows.len(), 1, "one event");
    assert_eq!(rows[0].by, EventTriggerKind::Import, "access move tagged Import");
    assert_eq!(rows[0].origin, "resolve");
}

// M10 exactly-once: the rebind-to-existing move delivers exactly one event.
// RED on baseline (0 delivered).
#[test]
fn m10_exactly_once() {
    let (br, td, _bytes) = setup_rebind_to_existing();
    br.resolve_main_to(td);
    assert_eq!(br.rec.len(), 1, "exactly one event, no double delivery");
}

// M11 sibling isolation: the sibling on the shared destination head records
// ZERO rows when `main` rebinds onto it. GREEN on baseline (nothing delivered)
// and must STAY green after Phase 3 (dispatch is branch-scoped).
#[test]
fn m11_sibling_isolation() {
    let br = Branched::open();
    br.md.create_branch(&b("draft"), &b("main")).unwrap();
    br.md
        .write(&b("draft"), |d| d.get_map("map").insert("d", "1").unwrap())
        .unwrap();
    let draft_head = br.md.bound_head(&b("draft")).unwrap();
    let td = br.md.head_tip(draft_head).unwrap();

    // A recorder on draft (installed on draft's head Td).
    let draft_rec = Recorder::new();
    let _dsub = br
        .md
        .subscribe_branch(&b("draft"), None, draft_rec.root_subscriber())
        .unwrap();

    // main rebinds onto draft's head Td.
    br.resolve_main_to(td);

    assert_eq!(
        draft_rec.len(),
        0,
        "the sibling on the shared destination head must not see main's jump"
    );
}

// M12 re-entrancy: a `main` subscriber that re-enters `read` from its callback.
// RED (lock-order panic) on baseline in debug; GREEN after Phase 1 (dispatch
// runs after the registry lock drops).
#[test]
fn m12_reentrancy() {
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), md.root_head_id());
    let md2 = md.clone();
    let reentered = Arc::new(Mutex::new(false));
    let re2 = reentered.clone();
    let _sub = md
        .subscribe_branch(
            &b("main"),
            None,
            Arc::new(move |_ev: DiffEvent| {
                // Re-enter a registry op from inside the callback.
                let _ = md2.read(&b("main"), |d| d.get_deep_value());
                *re2.lock().unwrap() = true;
            }),
        )
        .unwrap();

    let r = remote();
    write_map_str(&r, "map", "a", "1");
    r.commit_then_renew();
    md.import(&all_updates(&r)).unwrap();
    md.policy().set_target(&b("main"), r.state_frontiers());
    md.resolve(&b("main"), Intent::Read).unwrap();

    assert!(*reentered.lock().unwrap(), "callback re-entered without deadlock/panic");
}

// M13 access-realized backwards rebind then catch-up (the G5 lineage-first
// path, simulated): `main` at a content tip is reduced to the ROOT head at `[]`
// (by_tip[[]] == ROOT hit -> rebind-to-existing, "everything removed"), then a
// read re-realizes it (catch-up, "everything re-added"). RED on baseline (the
// backwards rebind is silent).
#[test]
fn m13_access_realized_backwards_rebind() {
    let br = Branched::open();
    // A sibling on the root keeps the pinned ROOT resting at [] while main
    // diverges onto its own head H (so the backwards move hits by_tip[[]] ==
    // ROOT: the rebind-to-existing arm, not catch-up).
    br.md.create_branch(&b("draft"), &b("main")).unwrap();
    // main writes content -> copy-on-divergence -> main's own head advances to Y.
    br.md
        .write(&b("main"), |d| d.get_map("map").insert("a", "1").unwrap())
        .unwrap();
    let y = br.md.head_tip(br.md.bound_head(&b("main")).unwrap()).unwrap();
    let n_after_write = br.rec.len();

    // Plain analog for the "removed then re-added" pair: a doc that checks out to
    // empty and back. We instead assert delivery COUNT parity against a plain
    // reference driven through the same two content transitions.
    let (doc, prec, _s) = plain();
    doc.get_map("map").insert("a", "1").unwrap();
    doc.commit_then_renew();
    // Plain: checkout back to genesis (empty), then forward again.
    let genesis = Frontiers::default();
    let content = doc.state_frontiers();
    doc.checkout(&genesis).unwrap();
    doc.checkout(&content).unwrap();

    // Branched: backwards rebind to ROOT (empty), then forward to Y.
    br.resolve_main_to(Frontiers::default());
    br.resolve_main_to(y);

    // Compare only the transition rows (drop the initial local-write rows).
    let p_rows: Vec<_> = prec.rows().into_iter().skip(1).collect();
    let b_rows: Vec<_> = br.rec.rows().into_iter().skip(n_after_write).collect();
    assert_eq!(
        p_rows.len(),
        b_rows.len(),
        "backwards-then-forward transition count parity\nplain={p_rows:#?}\nbranched={b_rows:#?}"
    );
    for (p, q) in p_rows.iter().zip(b_rows.iter()) {
        assert_eq!(p.mirror_after, q.mirror_after, "mirror parity on the transition");
    }
}
