//! Behavior parity: a branch-surfaced doc (one branch of a `MultiHeadDoc<Manual>`) must emit the
//! same `subscribe`/`subscribe_root` events as a plain `LoroDoc` for every update.
//!
//! Events are normalized to `EventRow { by, origin, touched, mirror_after }`; `mirror_after` is a
//! `LoroValue` rebuilt by applying each `ContainerDiff` along its path (as
//! `tests/test.rs::test_checkout`). Frontiers are not compared: heads mint their own peers, so
//! `from`/`to` differ by construction.
//!
//! `by`/`origin` are only compared in the M9 cases.

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

#[derive(Clone, Copy, Default)]
struct CompareOpts {
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

    /// Resolve `main` to `target` via the access path (tags `("resolve", Import)`).
    fn resolve_main_to(&self, target: Frontiers) {
        self.md.policy().set_target(&b("main"), target);
        self.md.resolve(&b("main"), Intent::Read).unwrap();
    }

    /// Resolve `main` to `target` under an explicit cause (mirrors `after_import`).
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

// M1: a local commit emits `Local` on both surfaces (no transition).
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

// M2 (i): import fast-forward, fresh doc, `main` alone (catch-up on the root head, refs == 1).
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

// M2 (ii): import fast-forward with a bound sibling; the root is shared (refs == 2), so the
// move takes the copy+advance arm.
#[test]
fn m2ii_import_fast_forward_shared_root() {
    let r = remote();
    write_map_str(&r, "map", "a", "1");
    r.commit_then_renew();
    let bytes = all_updates(&r);

    let (doc, prec, _s) = plain();
    doc.import(&bytes).unwrap();

    let br = Branched::open();
    br.md.create_branch(&b("draft"), &b("main")).unwrap();
    br.md.import(&bytes).unwrap();
    let target = r.state_frontiers();
    br.resolve_main_to(target);

    assert_parity(&prec, &br.rec, CompareOpts::default());
    assert_eq!(prec.len(), 1);
}

// M3: import with disjoint divergence on a uniquely-owned head (catch-up arm).
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

// M4: container created remotely, shared root (copy+advance).
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

// M5: delete, catch-up arm.
#[test]
fn m5_delete_catchup() {
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

// M6: nested container (map.child text) edit; the dispatch must reproduce the nested diff and path.
#[test]
fn m6_subcontainer_filter_catchup() {
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

// M7: sibling shares the head, remote writes on main (copy+advance, refs == 2).
#[test]
fn m7_sibling_shares_head_copy_advance() {
    let r = remote();
    write_map_str(&r, "map", "a", "1");
    r.commit_then_renew();
    let bytes = all_updates(&r);

    let (doc, prec, _s) = plain();
    doc.import(&bytes).unwrap();

    let br = Branched::open();
    br.md.create_branch(&b("draft"), &b("main")).unwrap();
    assert_eq!(br.md.refs_of(&b("main")), Some(2), "root shared");
    br.md.import(&bytes).unwrap();
    br.resolve_main_to(r.state_frontiers());

    assert_parity(&prec, &br.rec, CompareOpts::default());
    assert_eq!(prec.len(), 1);
}

// `draft` has diverged to its own head (tip Td); `main` still sits on the empty root.
fn setup_rebind_to_existing() -> (Branched, Frontiers, Vec<u8>) {
    let br = Branched::open();
    br.md.create_branch(&b("draft"), &b("main")).unwrap();
    br.md
        .write(&b("draft"), |d| d.get_map("map").insert("d", "1").unwrap())
        .unwrap();
    let draft_head = br.md.bound_head(&b("draft")).unwrap();
    let td = br.md.head_tip(draft_head).unwrap();
    let bytes = all_updates_md(&br.md);
    (br, td, bytes)
}

// M8: `main` moves onto `draft`'s existing head (`by_tip` hit). A rebind is a pointer swap
// yet must still deliver the diff.
#[test]
fn m8_rebind_to_existing() {
    let (br, td, bytes) = setup_rebind_to_existing();

    let (doc, prec, _s) = plain();
    doc.import(&bytes).unwrap();

    br.resolve_main_to(td);

    assert_parity(&prec, &br.rec, CompareOpts::default());
    assert_eq!(prec.len(), 1, "one event for the fast-forward");
}

// M9: `by`/`origin` compared; a plain doc emits `Import`/"", not `Checkout`/"checkout".

// M9a: import case -> Import + "".
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
    br.resolve_main_with(r.state_frontiers(), ResolveCause::Import { origin: "".into() });

    assert_parity(
        &prec,
        &br.rec,
        CompareOpts {
            compare_by_origin: true,
        },
    );
    assert_eq!(br.rec.rows()[0].by, EventTriggerKind::Import);
    assert_eq!(br.rec.rows()[0].origin, "");
}

// M9b: rebind-to-existing via the access (`resolve`) path -> Import + "resolve" origin.
// The "advance" origin is covered by `resolve_with_advance_tags_import_advance` in
// `multi_head/tests.rs`; the Manual harness has no `merge`.
#[test]
fn m9b_by_origin_resolve() {
    let (br, td, _bytes) = setup_rebind_to_existing();
    br.resolve_main_to(td);
    let rows = br.rec.rows();
    assert_eq!(rows.len(), 1, "one event");
    assert_eq!(rows[0].by, EventTriggerKind::Import, "access move tagged Import");
    assert_eq!(rows[0].origin, "resolve");
}

// M10: the rebind-to-existing move delivers exactly one event.
#[test]
fn m10_exactly_once() {
    let (br, td, _bytes) = setup_rebind_to_existing();
    br.resolve_main_to(td);
    assert_eq!(br.rec.len(), 1, "exactly one event, no double delivery");
}

// M11: the sibling on the shared destination head records no rows when `main` rebinds onto it
// (dispatch is branch-scoped).
#[test]
fn m11_sibling_isolation() {
    let br = Branched::open();
    br.md.create_branch(&b("draft"), &b("main")).unwrap();
    br.md
        .write(&b("draft"), |d| d.get_map("map").insert("d", "1").unwrap())
        .unwrap();
    let draft_head = br.md.bound_head(&b("draft")).unwrap();
    let td = br.md.head_tip(draft_head).unwrap();

    let draft_rec = Recorder::new();
    let _dsub = br
        .md
        .subscribe_branch(&b("draft"), None, draft_rec.root_subscriber())
        .unwrap();

    br.resolve_main_to(td);

    assert_eq!(
        draft_rec.len(),
        0,
        "the sibling on the shared destination head must not see main's jump"
    );
}

// M12: a `main` subscriber re-enters `read` from its callback (dispatch must run after the
// registry lock drops).
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

// M13: access-realized backwards rebind then catch-up. `main` is reduced to the ROOT head at `[]`
// (rebind-to-existing, "everything removed"), then a read re-realizes it (catch-up).
#[test]
fn m13_access_realized_backwards_rebind() {
    let br = Branched::open();
    // A sibling on the root keeps ROOT resting at [], so the backwards move hits by_tip[[]] == ROOT
    // (rebind-to-existing arm, not catch-up).
    br.md.create_branch(&b("draft"), &b("main")).unwrap();
    br.md
        .write(&b("main"), |d| d.get_map("map").insert("a", "1").unwrap())
        .unwrap();
    let y = br.md.head_tip(br.md.bound_head(&b("main")).unwrap()).unwrap();
    let n_after_write = br.rec.len();

    let (doc, prec, _s) = plain();
    doc.get_map("map").insert("a", "1").unwrap();
    doc.commit_then_renew();
    let genesis = Frontiers::default();
    let content = doc.state_frontiers();
    doc.checkout(&genesis).unwrap();
    doc.checkout(&content).unwrap();

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
