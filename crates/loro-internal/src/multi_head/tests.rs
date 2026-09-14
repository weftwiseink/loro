use super::*;
use crate::encoding::ExportMode;
use crate::handler::HandlerTrait;
use crate::lock::{LockKind, LoroLockGroup};
use crate::version::Frontiers;
use crate::LoroDoc;
use loro_common::{LoroError, LoroResult};
use std::sync::{Arc, Mutex};

fn b(s: &str) -> BranchId {
    s.into()
}

fn tlen(md: &MultiHeadDoc<Manual>, branch: &BranchId) -> usize {
    md.read(branch, |d| d.get_text("t").len_unicode()).unwrap()
}

// ------------------------------------------------------------------
// Registry / copy mechanics
// ------------------------------------------------------------------

#[test]
fn free_creation_shares_head_no_copy() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    md.bind(&main, md.root_head_id());
    // create_branch from a live head is a bind, not a copy.
    md.create_branch(&review, &main).unwrap();
    assert_eq!(md.bound_head(&review), Some(0));
    assert_eq!(md.bound_head(&main), Some(0));
    assert_eq!(md.refs_of(&main), Some(2));
    assert_eq!(md.head_count(), 1, "no head was copied on creation");
    assert_eq!(md.is_head_shared(&main), Some(true));
}

#[test]
fn copy_new_inner_fresh_peer_same_oplog() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    md.bind(&main, 0);
    md.bind(&review, 0); // shared
    md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
    let copy_id = md.bound_head(&main).unwrap();
    assert_ne!(copy_id, 0, "a divergent write minted a new head");
    let copy = md.head_doc(copy_id).unwrap();
    let root = md.head_doc(0).unwrap();
    assert_ne!(copy.peer_id(), root.peer_id(), "copy has a fresh peer");
    assert!(
        Arc::ptr_eq(&copy.oplog, &root.oplog),
        "copy shares the same Arc<OpLog>"
    );
}

#[test]
fn on_commit_rekeys_by_tip_and_shares_via_lookup() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, other) = (b("main"), b("other"));
    md.bind(&main, 0);
    // In-place write on a uniquely-owned head; on_commit must re-key by_tip.
    md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
    let t = md.head_tip(0).unwrap();
    assert!(!t.is_empty(), "tip advanced after the write");
    assert!(md.by_tip_maps(&t, 0), "by_tip re-keyed to the new tip");
    assert!(
        !md.by_tip_maps(&Frontiers::default(), 0),
        "the old empty tip was vacated"
    );
    // An unbound branch whose target is that tip SHARES head 0 via the
    // by_tip lookup arm (no new head).
    md.policy().set_target(&other, t.clone());
    let (id, _) = md.resolve(&other, Intent::Read).unwrap();
    assert_eq!(id, 0);
    assert_eq!(md.head_count(), 1);
    assert_eq!(md.refs_of(&other), Some(2));
}

// ------------------------------------------------------------------
// FLOOR: lock-group membership discriminator (+ fresh-group contrast)
// ------------------------------------------------------------------

#[test]
fn lock_group_shared_docstate_before_oplog_panics() {
    // A copied head's DocState lock lives in the SHARED group with the root
    // head's OpLog lock, so a DocState-before-OpLog acquisition ACROSS the
    // two heads is caught by the debug order checker (kind 4 then kind 3).
    // This is the whole point of `fork_in_group`: both heads share one
    // ordering domain.
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    md.bind(&main, 0);
    md.bind(&review, 0);
    md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
    let copy = md.head_doc(md.bound_head(&main).unwrap()).unwrap();
    let root = md.head_doc(0).unwrap();

    // Suppress the expected panic's backtrace noise, then restore.
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let copy_c = copy.clone();
    let root_c = root.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ds = copy_c.state.lock(); // DocState (kind 4)
        let _ol = root_c.oplog.lock(); // OpLog (kind 3) after DocState -> panic
    }));
    std::panic::set_hook(prev);
    assert!(
        result.is_err(),
        "DocState-before-OpLog across two shared-group heads must panic"
    );
    // The order-violation panic poisoned the copy's DocState mutex while its
    // guard was held; `forget` the whole doc so no `Drop` re-locks it (which
    // would double-panic). The clones above drop harmlessly (strong_count>1).
    std::mem::forget(md);
}

#[test]
fn lock_group_fresh_group_docstate_before_oplog_does_not_panic() {
    // CONTRAST: the hazard `fork_in_group` fixes. A DocState lock built in a
    // FRESH group (as `fork_with_new_peer_id`'s bare mutex / a per-handle
    // group would be) is in a different ordering domain, so acquiring it
    // before the shared OpLog lock does NOT panic -- the inversion goes
    // unseen. This is exactly why a head must join the shared group.
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    let root = md.head_doc(0).unwrap();

    let fresh = LoroLockGroup::new();
    let foreign_docstate = fresh.new_lock(0u8, LockKind::DocState);
    let _ds = foreign_docstate.lock(); // DocState (kind 4) in a DIFFERENT domain
    let _ol = root.oplog.lock(); // no panic: different threadlocal
                                 // Reaching here without a panic is the assertion.
}

// ------------------------------------------------------------------
// FLOOR: sink guard
// ------------------------------------------------------------------

#[test]
fn sink_guard_mutation_in_read_closure_refuses_and_baseline_corrupts() {
    // GUARDED: a mutation inside a `read` closure on a shared head returns
    // HeadShared; the co-owner is untouched.
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    md.bind(&main, 0);
    md.bind(&review, 0); // shared, refs 2
    assert_eq!(md.is_head_shared(&main), Some(true));

    let res: LoroResult<()> = md
        .read(&main, |d| d.get_text("t").insert_unicode(0, "X"))
        .unwrap();
    assert!(
        matches!(res, Err(LoroError::HeadShared)),
        "mutation inside read on a shared head is refused, got {res:?}"
    );
    assert_eq!(tlen(&md, &review), 0, "co-owner untouched under the guard");

    // BASELINE (single variable = the guard flag): flip the SAME head to
    // Private (the pre-guard state where a refs>1 head was writable) and
    // the identical mutation now lands and CORRUPTS the co-owner.
    let leaked = md.head_doc(0).unwrap();
    leaked.set_head_shared(false);
    leaked.renew_txn_if_auto_commit(None);
    leaked.get_text("t").insert_unicode(0, "X").unwrap();
    assert_eq!(
        tlen(&md, &review),
        1,
        "baseline: unguarded shared-head write corrupts the co-owner"
    );
}

#[test]
fn sink_guard_leaked_handle_refused_after_flip() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    md.bind(&main, 0); // refs 1, private
    let leaked = md.head_doc(0).unwrap();
    // A write while private is fine.
    leaked.get_text("t").insert_unicode(0, "A").unwrap();

    md.bind(&review, 0); // refs 2 -> Private->Shared flip (commits "A")
    assert_eq!(md.is_head_shared(&main), Some(true));

    // The leaked handle, used AFTER the flip, is refused at both sinks.
    let r_op = leaked.get_text("t").insert_unicode(0, "B");
    assert!(
        matches!(r_op, Err(LoroError::HeadShared)),
        "leaked local op after flip refused, got {r_op:?}"
    );
    // A leaked raw checkout is now refused earlier, by the owned-head
    // checkout gate (before it can reach the apply_diff sink), so the error
    // is `OwnedHeadOp` rather than `HeadShared`. Either way: refused.
    let r_checkout = leaked.checkout(&Frontiers::default());
    assert!(
        matches!(r_checkout, Err(LoroError::OwnedHeadOp("checkout"))),
        "leaked checkout after flip refused, got {r_checkout:?}"
    );
    // The co-owner still reads only the pre-flip content ("A", len 1).
    // (`bind` is a low-level entry that does not update the Manual target,
    // so point `review` at the head's committed tip before reading, or
    // `resolve` would treat the stale empty target as authoritative.)
    md.policy().set_target(&review, md.head_tip(0).unwrap());
    assert_eq!(tlen(&md, &review), 1);
}

#[test]
fn sink_guard_nested_mutation_in_event_callback_refused() {
    let md = Arc::new(MultiHeadDoc::new(Manual::new()));
    let (main, review, writer) = (b("main"), b("review"), b("writer"));
    md.bind(&main, 0);
    md.bind(&review, 0);
    md.bind(&writer, 0); // refs 3, head 0 shared

    // Give `writer` its own private head by diverging once.
    md.write(&writer, |d| d.get_text("t").insert_unicode(0, "1").unwrap())
        .unwrap();
    assert_eq!(md.is_head_shared(&main), Some(true), "head 0 still shared");
    let writer_doc = md.head_doc(md.bound_head(&writer).unwrap()).unwrap();

    // A callback on the writer's (private) head that, when it fires, tries
    // to mutate the SHARED head 0 -- a nested mutation from inside an event.
    let shared_head = md.head_doc(0).unwrap();
    let captured: Arc<Mutex<Option<LoroResult<()>>>> = Arc::new(Mutex::new(None));
    let cap2 = captured.clone();
    let sub = writer_doc.subscribe_root(Arc::new(move |_ev: crate::event::DiffEvent| {
        let r = shared_head.get_text("t").insert_unicode(0, "Z");
        *cap2.lock().unwrap() = Some(r);
    }));

    // Fire the callback with another write on the writer's own head.
    md.write(&writer, |d| d.get_text("t").insert_unicode(0, "2").unwrap())
        .unwrap();
    drop(sub);

    let got = captured.lock().unwrap().take();
    assert!(
        matches!(got, Some(Err(LoroError::HeadShared))),
        "nested mutation of a shared head inside an event callback refused, got {got:?}"
    );
    assert_eq!(tlen(&md, &main), 0, "the shared co-owner was not corrupted");
}

// ------------------------------------------------------------------
// FLOOR: copy-then-write correctness
// ------------------------------------------------------------------

#[test]
fn copy_then_write_coowners_unaffected() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review, draft) = (b("main"), b("review"), b("draft"));
    md.bind(&main, 0);
    md.bind(&review, 0);
    md.bind(&draft, 0); // refs 3, shared
    assert_eq!(md.refs_of(&main), Some(3));
    assert_eq!(md.is_head_shared(&main), Some(true));

    // draft diverges.
    md.write(&draft, |d| d.get_text("t").insert_unicode(0, "D").unwrap())
        .unwrap();
    assert_ne!(md.bound_head(&draft), Some(0), "draft copied off head 0");
    assert_eq!(
        md.refs_of(&main),
        Some(2),
        "head 0 still shared by main + review"
    );
    assert_eq!(md.is_head_shared(&main), Some(true));

    assert_eq!(tlen(&md, &draft), 1, "writer sees its own op");
    assert_eq!(tlen(&md, &main), 0, "co-owner main unaffected");
    assert_eq!(tlen(&md, &review), 0, "co-owner review unaffected");

    // A subsequent op by the writer still does not touch the co-owners.
    md.write(&draft, |d| d.get_text("t").insert_unicode(1, "E").unwrap())
        .unwrap();
    assert_eq!(tlen(&md, &draft), 2);
    assert_eq!(tlen(&md, &main), 0);
    assert_eq!(tlen(&md, &review), 0);
}

// ------------------------------------------------------------------
// All-heads barrier
// ------------------------------------------------------------------

#[test]
fn all_heads_barrier_runs_over_every_head() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    md.bind(&main, 0);
    md.bind(&review, 0);
    md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap(); // 2 heads now
    assert_eq!(md.head_count(), 2);
    let ran = md.with_all_heads_barrier(|| 42);
    assert_eq!(ran, 42);
    // Heads remain usable afterward (txns renewed).
    md.write(&main, |d| d.get_text("t").insert_unicode(1, "B").unwrap())
        .unwrap();
    assert_eq!(tlen(&md, &main), 2);
}

// ------------------------------------------------------------------
// Injected on-commit hook (migration off the synchronous write() path)
// ------------------------------------------------------------------

#[test]
fn injected_on_commit_hook_rekeys_and_after_commits() {
    // Commit DIRECTLY on a head's own handle, bypassing MultiHeadDoc::write
    // entirely. by_tip re-keying and Manual's after_commit can now come ONLY
    // from the injected txn on-commit hook (the synchronous write()-driven
    // path is gone), so their effects prove the hook fired.
    let md = MultiHeadDoc::new(Manual::new());
    let main = b("main");
    md.bind(&main, 0); // refs 1, private, tip empty
    assert!(md.by_tip_maps(&Frontiers::default(), 0));

    let doc = md.head_doc(0).unwrap();
    doc.get_text("t").insert_unicode(0, "A").unwrap();
    doc.commit_then_renew(); // fires the injected on_commit hook

    let t = md.head_tip(0).unwrap();
    assert!(!t.is_empty(), "tip advanced");
    assert!(md.by_tip_maps(&t, 0), "hook re-keyed by_tip to the new tip");
    assert!(
        !md.by_tip_maps(&Frontiers::default(), 0),
        "hook vacated the old empty tip"
    );
    assert_eq!(
        md.policy().target_of(&main),
        Some(t),
        "hook ran after_commit (Manual recorded the new target)"
    );
}

// ------------------------------------------------------------------
// History-only import: no head materialization, co-owner safe, visible
// to a head that advances; sink guard backstops a shared head.
// ------------------------------------------------------------------

fn external_updates(text: &str) -> Vec<u8> {
    let ext = LoroDoc::new();
    ext.start_auto_commit();
    ext.get_text("t").insert_unicode(0, text).unwrap();
    ext.commit_then_renew();
    ext.export(crate::encoding::ExportMode::all_updates())
        .unwrap()
}

fn external_updates_frontier(text: &str) -> (Vec<u8>, Frontiers) {
    let ext = LoroDoc::new();
    ext.start_auto_commit();
    ext.get_text("t").insert_unicode(0, text).unwrap();
    ext.commit_then_renew();
    let f = ext.state_frontiers();
    let bytes = ext
        .export(crate::encoding::ExportMode::all_updates())
        .unwrap();
    (bytes, f)
}

#[test]
fn import_is_history_only_and_coowner_safe() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    md.bind(&main, 0);
    md.bind(&review, 0); // refs 2, head 0 shared
    assert_eq!(md.is_head_shared(&main), Some(true));

    let updates = external_updates("ABC");
    md.import(&updates).unwrap();

    // History-only: the shared head's materialized state is UNTOUCHED, so no
    // co-owner sees the imported ops. (The failure picture: an import that
    // silently materialized into head 0 would corrupt both main and review.)
    assert_eq!(
        tlen(&md, &main),
        0,
        "shared co-owner main untouched by import"
    );
    assert_eq!(
        tlen(&md, &review),
        0,
        "shared co-owner review untouched by import"
    );

    // The ops ARE in the shared history: a fresh branch that advances to
    // include them materializes them.
    let (updates2, f2) = external_updates_frontier("XYZ");
    md.import(&updates2).unwrap();
    let reader = b("reader");
    md.policy().set_target(&reader, f2);
    let seen = md.read(&reader, |d| d.get_text("t").to_string()).unwrap();
    assert_eq!(
        seen, "XYZ",
        "imported ops visible to a head advanced to them"
    );
    // Co-owners of the shared head still see nothing.
    assert_eq!(tlen(&md, &main), 0);
    assert_eq!(tlen(&md, &review), 0);
}

#[test]
fn import_path_sink_guard_backstops_shared_head() {
    // The history-only import never materializes into a head. As a backstop,
    // even a DIRECT attempt to apply imported ops to the shared head (via
    // checkout -> apply_diff) is refused by the sink guard rather than
    // silently corrupting co-owners. This is the guard the history-only rule
    // makes it unnecessary to rely on, shown load-bearing.
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    md.bind(&main, 0);
    md.bind(&review, 0); // shared

    let (updates, f) = external_updates_frontier("ABC");
    md.import(&updates).unwrap();
    assert_eq!(
        tlen(&md, &main),
        0,
        "history-only import left the shared head empty"
    );

    // A direct checkout of the shared head to the imported frontier would be
    // a materialization; the owned-head checkout gate refuses it (before the
    // apply_diff sink guard would).
    let shared = md.head_doc(0).unwrap();
    let r = shared.checkout(&f);
    assert!(
        matches!(r, Err(LoroError::OwnedHeadOp("checkout"))),
        "checkout on a shared owned head refused, got {r:?}"
    );
    assert_eq!(
        tlen(&md, &main),
        0,
        "co-owner still uncorrupted after the refused attempt"
    );
}

#[test]
fn owned_head_refuses_direct_import_and_set_peer_id() {
    // The owner gates: a registry head is mutated only through the registry.
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    let head = md.head_doc(0).unwrap();
    let updates = external_updates("ABC");
    assert!(
        matches!(head.import(&updates), Err(LoroError::OwnedHeadOp("import"))),
        "direct import on an owned head refused"
    );
    assert!(
        matches!(
            head.set_peer_id(12345),
            Err(LoroError::OwnedHeadOp("set_peer_id"))
        ),
        "set_peer_id on an owned head refused"
    );
}

// ------------------------------------------------------------------
// Completed base methods: aggregated subscriptions and snapshot export.
// ------------------------------------------------------------------

#[test]
fn subscribe_history_fires_on_commit_and_import() {
    let md = MultiHeadDoc::new(Manual::new());
    let main = b("main");
    md.bind(&main, 0);
    let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let c2 = count.clone();
    let _sub = md.subscribe_history(Box::new(move |_bytes| {
        c2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        true
    }));
    md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
    let after_commit = count.load(std::sync::atomic::Ordering::SeqCst);
    assert!(after_commit >= 1, "history event fired on a head commit");
    md.import(&external_updates("Z")).unwrap();
    assert!(
        count.load(std::sync::atomic::Ordering::SeqCst) > after_commit,
        "history event fired on import"
    );
}

#[test]
fn subscribe_first_commit_from_peer_aggregates_over_heads() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    let peers = Arc::new(Mutex::new(Vec::<u64>::new()));
    let p2 = peers.clone();
    let _sub = md.subscribe_first_commit_from_peer(Box::new(move |payload| {
        p2.lock().unwrap().push(payload.peer);
        true
    }));
    md.bind(&main, 0);
    md.bind(&review, 0); // shared
                         // main diverges -> a copy with a fresh peer -> a first-commit from it.
    md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
    // review diverges -> another fresh peer.
    md.write(&review, |d| d.get_text("t").insert_unicode(0, "B").unwrap())
        .unwrap();
    let seen = peers.lock().unwrap().clone();
    assert!(
        seen.len() >= 2,
        "one first-commit per write slot (fresh peer per copy), got {seen:?}"
    );
    assert_ne!(seen[0], seen[1], "distinct peers per head copy");
}

#[test]
fn export_snapshot_roundtrips_a_head_state() {
    let md = MultiHeadDoc::new(Manual::new());
    let main = b("main");
    md.bind(&main, 0);
    md.write(&main, |d| {
        d.get_text("t").insert_unicode(0, "hello").unwrap()
    })
    .unwrap();
    // Snapshot carries the root head's state; a fresh doc restores it.
    let snap = md.export(crate::encoding::ExportMode::Snapshot).unwrap();
    let restored = LoroDoc::new();
    restored.import(&snap).unwrap();
    assert_eq!(restored.get_text("t").to_string(), "hello");
    // Updates export is head-independent (shared op log).
    let updates = md
        .export(crate::encoding::ExportMode::all_updates())
        .unwrap();
    let restored2 = LoroDoc::new();
    restored2.import(&updates).unwrap();
    assert_eq!(restored2.get_text("t").to_string(), "hello");
}

// ------------------------------------------------------------------
// refs==0 retirement: non-root heads are dropped, the root is pinned.
// ------------------------------------------------------------------

#[test]
fn dead_tip_rematerializes_via_replay_after_head_dropped() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, feature) = (b("main"), b("feature"));
    md.bind(&main, 0);
    md.bind(&feature, 0); // root shared, refs 2

    // feature diverges onto its OWN sole-owner (non-root) head H.
    md.write(&feature, |d| {
        d.get_text("t").insert_unicode(0, "F").unwrap()
    })
    .unwrap();
    let h = md.bound_head(&feature).unwrap();
    assert_ne!(h, md.root_head_id(), "feature is on a non-root head");
    assert_eq!(md.refs_of(&feature), Some(1), "H is feature's sole owner");
    let h_tip = md.head_tip(h).unwrap();
    let pre_value = md.read(&feature, |d| d.get_text("t").to_string()).unwrap();
    assert_eq!(pre_value, "F");

    // Rebind feature OFF H (target back to the root's empty tip): H reaches
    // refs 0 and is RETIRED (dropped), root is not.
    md.policy().set_target(&feature, Frontiers::default());
    md.resolve(&feature, Intent::Read).unwrap();
    assert!(
        md.head_doc(h).is_none(),
        "sole-owner head retired at refs 0"
    );
    assert!(
        md.head_doc(md.root_head_id()).is_some(),
        "root head is never retired"
    );

    // A later access to H's now-dead tip re-materializes correctly via
    // replay from the pinned root (value equality with the pre-drop state).
    let reader = b("reader");
    md.policy().set_target(&reader, h_tip.clone());
    let seen = md.read(&reader, |d| d.get_text("t").to_string()).unwrap();
    assert_eq!(
        seen, pre_value,
        "dead tip re-materialized to the same value via replay"
    );
    assert_ne!(
        md.bound_head(&reader).unwrap(),
        h,
        "re-materialization built a fresh head, not the dropped one"
    );
}

#[test]
fn root_head_is_pinned_never_retired() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, x) = (b("main"), b("x"));
    md.bind(&main, 0);
    md.bind(&x, 0); // root shared, refs 2

    // x diverges onto its own head Hx; root drops to refs 1 (main only).
    md.write(&x, |d| d.get_text("t").insert_unicode(0, "X").unwrap())
        .unwrap();
    let hx = md.bound_head(&x).unwrap();
    assert_ne!(hx, 0);

    // Move main onto Hx too, driving the root head to refs 0.
    md.policy().set_target(&main, md.head_tip(hx).unwrap());
    md.resolve(&main, Intent::Read).unwrap();
    assert_ne!(md.bound_head(&main), Some(0), "main left the root head");

    // The root reached refs 0 but is PINNED: still present and still the
    // materialize/import/export anchor.
    assert!(
        md.head_doc(0).is_some(),
        "root head is pinned and survives refs == 0"
    );
    // It still anchors resolution: a branch targeting the root's (empty) tip
    // shares it rather than hitting a missing head.
    let cold = b("cold");
    md.policy().set_target(&cold, Frontiers::default());
    let seen = md.read(&cold, |d| d.get_text("t").to_string()).unwrap();
    assert_eq!(seen, "", "pinned root still resolvable at its empty tip");
}

// ------------------------------------------------------------------
// Phase 2: IndexDoc = MultiHeadDoc<SelfRooted>
// ------------------------------------------------------------------

fn d(s: &str) -> DocId {
    s.into()
}

fn doc_has_key(md: &IndexDoc, branch: &BranchId, key: &str) -> bool {
    md.read(branch, |doc| doc.get_map("docs").get_(key).is_some())
        .unwrap()
}

#[test]
fn index_eager_copy_born_refs_one_never_shared() {
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    idx.create_index_branch(&b("feat"), &b("main")).unwrap();
    // Eager: feat is on its OWN head (a copy of main's), refs 1, private.
    assert_ne!(idx.bound_head(&b("feat")), idx.bound_head(&b("main")));
    assert_eq!(idx.refs_of(&b("feat")), Some(1));
    assert_eq!(idx.is_head_shared(&b("feat")), Some(false));
    assert!(idx.branches().contains(&b("main")));
    assert!(idx.branches().contains(&b("feat")));
}

#[test]
fn index_remote_lineage_import_rebinds_branch() {
    // Session A creates `feat` and records content on it.
    let a = IndexDoc::new(SelfRooted::new());
    a.init_genesis(&b("main")).unwrap();
    a.create_index_branch(&b("feat"), &b("main")).unwrap();
    a.record_head(&b("feat"), &d("D"), 111, 7).unwrap();
    let updates = a.export(ExportMode::all_updates()).unwrap();

    // Session B knows only `main`.
    let bb = IndexDoc::new(SelfRooted::new());
    bb.init_genesis(&b("main")).unwrap();
    assert!(
        !bb.branches().contains(&b("feat")),
        "feat unknown before import"
    );

    // Importing A's history discovers `feat` via the lineage scan and
    // rebinds it (after_import -> resolve).
    bb.import(&updates).unwrap();
    assert!(
        bb.branches().contains(&b("feat")),
        "remote branch discovered"
    );
    assert!(
        doc_has_key(&bb, &b("feat"), "D"),
        "feat's index head materialized A's record"
    );
    assert!(
        !doc_has_key(&bb, &b("main"), "D"),
        "main did not absorb feat's record"
    );
}

#[test]
fn index_target_is_join_of_lineage_peers() {
    // Session A creates `shared` and records doc "A" on it.
    let a = IndexDoc::new(SelfRooted::new());
    a.init_genesis(&b("main")).unwrap();
    a.create_index_branch(&b("shared"), &b("main")).unwrap();
    a.record_head(&b("shared"), &d("A"), 1, 1).unwrap();
    let a_updates = a.export(ExportMode::all_updates()).unwrap();

    // Session B independently creates the SAME branch `shared`, records "B".
    let bb = IndexDoc::new(SelfRooted::new());
    bb.init_genesis(&b("main")).unwrap();
    bb.create_index_branch(&b("shared"), &b("main")).unwrap();
    bb.record_head(&b("shared"), &d("B"), 2, 2).unwrap();
    assert!(doc_has_key(&bb, &b("shared"), "B"));
    assert!(!doc_has_key(&bb, &b("shared"), "A"), "B has not seen A yet");

    // Import A's history: `shared` now has TWO lineage peers on B, and its
    // frontier is their JOIN -> B's shared head shows BOTH records.
    bb.import(&a_updates).unwrap();
    assert!(
        doc_has_key(&bb, &b("shared"), "A"),
        "join of lineage peers brought session A's record"
    );
    assert!(
        doc_has_key(&bb, &b("shared"), "B"),
        "join of lineage peers kept session B's record"
    );
}

/// Number of `docs[doc].heads` entries visible on `branch` (0 if absent).
fn head_count(md: &IndexDoc, branch: &BranchId, doc: &str) -> usize {
    md.read(branch, |d| {
        d.get_deep_value()
            .as_map()
            .and_then(|root| root.get("docs").cloned())
            .and_then(|v| v.into_map().ok())
            .and_then(|m| m.get(doc).cloned())
            .and_then(|v| v.as_map().cloned())
            .and_then(|per| per.get("heads").cloned())
            .and_then(|v| v.into_map().ok())
            .map(|h| h.len())
            .unwrap_or(0)
    })
    .unwrap()
}

#[test]
fn index_record_head_same_doc_converges_across_sessions() {
    // Two sessions on the SAME branch record a head for the SAME doc "G"
    // under different head-peer keys. With mergeable map-key children,
    // docs["G"].heads is the SAME container on both sides, so both records
    // survive a cross-import. (Divergent op-id children would LWW-drop one.)
    let a = IndexDoc::new(SelfRooted::new());
    a.init_genesis(&b("main")).unwrap();
    a.create_index_branch(&b("shared"), &b("main")).unwrap();
    a.record_head(&b("shared"), &d("G"), 111, 1).unwrap();

    let bb = IndexDoc::new(SelfRooted::new());
    bb.init_genesis(&b("main")).unwrap();
    bb.create_index_branch(&b("shared"), &b("main")).unwrap();
    bb.record_head(&b("shared"), &d("G"), 222, 2).unwrap();

    let a_updates = a.export(ExportMode::all_updates()).unwrap();
    let b_updates = bb.export(ExportMode::all_updates()).unwrap();
    a.import(&b_updates).unwrap();
    bb.import(&a_updates).unwrap();

    // The discriminating assertion: no head record is lost -- the same
    // mergeable heads container carries BOTH sessions' records on both sides.
    assert_eq!(
        head_count(&a, &b("shared"), "G"),
        2,
        "session A: both head records converged (mergeable child, not LWW-dropped)"
    );
    assert_eq!(
        head_count(&bb, &b("shared"), "G"),
        2,
        "session B: both head records converged"
    );
    // The docs subtree is byte-identical across sessions.
    let docs_of = |md: &IndexDoc| {
        md.read(&b("shared"), |d| {
            d.get_deep_value()
                .as_map()
                .and_then(|r| r.get("docs").cloned())
        })
        .unwrap()
    };
    let a_docs = docs_of(&a);
    let b_docs = docs_of(&bb);
    assert_eq!(
        a_docs, b_docs,
        "docs subtree converges to an identical value"
    );
}

#[test]
fn index_import_then_record_converges() {
    // The import-THEN-record ordering real peer sync produces (the existing
    // convergence test records BEFORE importing, which masked this). A
    // session first receives a remote import -- advancing its refs==1 bound
    // index head via the resolve CATCH-UP arm (checkout -> detached) -- THEN
    // records locally on that head. Without the catch-up detached-clear, the
    // local record fails `AutoCommitNotStarted` and its head record is
    // dropped (head_count wrong). With it, the record lands and converges.
    let a = IndexDoc::new(SelfRooted::new());
    a.init_genesis(&b("main")).unwrap();
    a.create_index_branch(&b("shared"), &b("main")).unwrap();
    a.record_head(&b("shared"), &d("G"), 111, 1).unwrap();
    let a_updates = a.export(ExportMode::all_updates()).unwrap();

    let bb = IndexDoc::new(SelfRooted::new());
    bb.init_genesis(&b("main")).unwrap();
    bb.create_index_branch(&b("shared"), &b("main")).unwrap();

    // IMPORT FIRST: advances bb's `shared` index head (refs==1) to the
    // lineage join through the catch-up arm.
    bb.import(&a_updates).unwrap();
    assert!(
        doc_has_key(&bb, &b("shared"), "G"),
        "import advanced the head to A's record"
    );

    // THEN record locally on that just-advanced head. This must succeed.
    bb.record_head(&b("shared"), &d("G"), 222, 2).unwrap();
    assert_eq!(
        head_count(&bb, &b("shared"), "G"),
        2,
        "import-then-record kept BOTH head records (no dropped record)"
    );
}

// ------------------------------------------------------------------
// Phase-3 precondition: BranchingDocHead surface, Branch signatures,
// read-exit-commit, and the two E1 closures.
// ------------------------------------------------------------------

#[test]
fn branching_doc_head_surface_read_write() {
    // read/write hand the closure a &BranchingDocHead exposing head-safe
    // methods (container access, state reads, local write). The history /
    // attachment methods (import/export/checkout/attach/detach/oplog_*/
    // set_peer_id/fork/diff) are ABSENT at the type level -- see the
    // `compile_fail` doctest on `BranchingDocHead`.
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    let br = md.branch("main");
    br.write(|head: &BranchingDocHead| {
        head.get_text("t").insert_unicode(0, "hi").unwrap();
    })
    .unwrap();
    let (text, deep_is_map, frontier_nonempty) = br
        .read(|head: &BranchingDocHead| {
            // Exercise head-safe reads (peer_id/len_ops are callable too).
            let _ = head.peer_id();
            let _ = head.len_ops();
            (
                head.get_text("t").to_string(),
                head.get_deep_value().is_map(),
                !head.state_frontiers().is_empty(),
            )
        })
        .unwrap();
    assert_eq!(text, "hi");
    assert!(deep_is_map, "get_deep_value returns the doc map");
    assert!(frontier_nonempty, "state advanced after the write");
}

#[test]
fn read_exit_commits_pending_mutation() {
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    let br = md.branch("main");
    // A mutation inside `read` is committed as a normal local edit on exit.
    br.read(|head| head.get_text("t").insert_unicode(0, "X").unwrap())
        .unwrap();
    // After the read returns nothing is left pending, and the edit persists.
    let (pending, text) = br
        .read(|head| (head.get_pending_txn_len(), head.get_text("t").to_string()))
        .unwrap();
    assert_eq!(
        pending, 0,
        "read committed pending ops on exit (nothing left)"
    );
    assert_eq!(text, "X", "the mutation persisted");
}

#[test]
fn e1_raw_checkout_seam_refused_on_owned_head() {
    // The `doc()` re-entry seam: `head.get_text("t").doc()` hands back the
    // raw owned `LoroDoc`. A raw `checkout` on it would move the head's state
    // WITHOUT re-keying the registry, desyncing `tip`/`by_tip` (a sibling
    // bound to the head would read the moved-away state while the registry
    // still says the head's tip). The gate refuses it.
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    md.write(&b("main"), |d| {
        d.get_text("t").insert_unicode(0, "M").unwrap()
    })
    .unwrap();

    // Drive the raw seam from inside a read closure.
    let r: LoroResult<()> = md
        .branch("main")
        .read(|head| {
            let raw = head.get_text("t").doc().expect("handler has a doc");
            raw.checkout(&Frontiers::default())
        })
        .unwrap();
    assert!(
        matches!(r, Err(LoroError::OwnedHeadOp("checkout"))),
        "raw checkout on an owned head refused, got {r:?}"
    );
    // The registry is NOT desynced: main still reads its own state.
    assert_eq!(
        md.branch("main")
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "M",
        "the head was not moved out from under the registry"
    );
}

#[test]
fn e1_detach_noop_on_owned_head() {
    // `detach()` is gated (no-op) on a registry-owned head -- part of the
    // deferred checkout/attach/detach trio: a branch head's attachment is the
    // registry's, not the raw doc's, to change.
    //
    // (The attach/checkout_to_latest no-op gates are defense-in-depth: after
    // the resolve arms all clear `detached`, and external checkout/detach are
    // gated, no reachable owned head is left detached to observe them on --
    // the discriminating case returns with the deferred create_branch_at
    // read-only-view head. The observable E1-A guard is the raw-checkout-seam
    // test above.)
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    md.write(&b("main"), |d| {
        d.get_text("t").insert_unicode(0, "M").unwrap()
    })
    .unwrap();
    let head = md.head_doc(0).unwrap();
    assert!(!head.is_detached(), "owned head starts attached");
    head.detach();
    assert!(!head.is_detached(), "detach refused (no-op) on owned head");
}

#[test]
fn e1_site_b_branch_scoped_new_version() {
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    md.bind(&b("feat"), 0); // shared
    md.write(&b("feat"), |d| {
        d.get_text("t").insert_unicode(0, "F").unwrap()
    })
    .unwrap(); // feat -> own head; main keeps root
    md.write(&b("main"), |d| {
        d.get_text("t").insert_unicode(0, "M").unwrap()
    })
    .unwrap();

    let captured: Arc<Mutex<Option<Frontiers>>> = Arc::new(Mutex::new(None));
    let cap = captured.clone();
    let sub = md
        .branch("main")
        .subscribe_root(Arc::new(move |ev: crate::event::DiffEvent| {
            *cap.lock().unwrap() = Some(ev.event_meta.to.clone());
        }))
        .unwrap();

    // A sibling advances the shared union; main's subscriber must not adopt it.
    md.write(&b("feat"), |d| {
        d.get_text("t").insert_unicode(1, "2").unwrap()
    })
    .unwrap();
    // main commits -> its subscriber fires with a BRANCH-SCOPED new_version.
    md.write(&b("main"), |d| {
        d.get_text("t").insert_unicode(1, "2").unwrap()
    })
    .unwrap();
    drop(sub);

    let to = captured
        .lock()
        .unwrap()
        .take()
        .expect("main's subscriber fired");
    let main_frontier = md.branch("main").read(|h| h.state_frontiers()).unwrap();
    let union = md.head_doc(0).unwrap().oplog().lock().frontiers().clone();
    assert_eq!(to, main_frontier, "new_version is main's own frontier");
    assert_ne!(
        to, union,
        "new_version is NOT the shared union (feat excluded)"
    );
}

#[test]
fn branch_fork_ejects_coherent_checkout_able_doc() {
    // main's head is behind the shared union (feat diverged), yet still
    // "attached" (detached flag false). `Branch::fork` must eject via
    // `fork_at(&state_frontiers())` so the ejected snapshot's state matches
    // its own frontier label -- not `LoroDoc::fork()`, which would label
    // main's content with the union frontier (incoherent -> panics on
    // checkout).
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    md.bind(&b("feat"), 0); // shared
    md.write(&b("feat"), |d| {
        d.get_text("t").insert_unicode(0, "F").unwrap()
    })
    .unwrap(); // feat diverges; union advances
    md.write(&b("main"), |d| {
        d.get_text("t").insert_unicode(0, "M").unwrap()
    })
    .unwrap(); // main behind the union

    let forked = md.branch("main").fork().unwrap();
    assert_eq!(forked.get_text("t").to_string(), "M");
    // The ejected doc is coherent and checkout-able (would panic if the
    // state/frontier label were mismatched).
    let f = forked.state_frontiers();
    forked.checkout(&Frontiers::default()).unwrap();
    assert_eq!(forked.get_text("t").to_string(), "");
    forked.checkout(&f).unwrap();
    assert_eq!(forked.get_text("t").to_string(), "M");
    // It aliases no registry head: main is untouched.
    assert_eq!(
        md.branch("main")
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "M"
    );
}

// ------------------------------------------------------------------
// Phase 3 body: refs>1 copy+advance detached-clear (base), and the
// Delegated engine (repo create / write / read / merge / advance).
// ------------------------------------------------------------------

#[test]
fn refs_gt1_copy_advance_clears_detached() {
    // The refs>1 copy+advance arm: a SHARED head whose branch target moved
    // off the shared tip is copied and advanced (checkout -> detached). The
    // copy is the branch's live writable head, so a subsequent write must
    // succeed (the clear); WITHOUT it the write fails on a detached head.
    let md = MultiHeadDoc::new(Manual::new());
    // main + feat SHARE the root head; bind `x` too, then `x` diverges (root
    // is shared, so its write copies off) onto its own head and builds "AB",
    // leaving root empty and shared by main + feat (refs 2).
    md.bind(&b("main"), 0);
    md.bind(&b("feat"), 0);
    md.bind(&b("x"), 0);
    md.write(&b("x"), |d| d.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap(); // x diverges off the shared root onto its own head
    let t_a = md.head_tip(md.bound_head(&b("x")).unwrap()).unwrap();
    md.write(&b("x"), |d| d.get_text("t").insert_unicode(1, "B").unwrap())
        .unwrap();
    assert_eq!(md.refs_of(&b("main")), Some(2));

    // main's target -> the intermediate frontier: resolve(Write) hits the
    // refs>1 copy+advance arm.
    md.policy().set_target(&b("main"), t_a.clone());
    let inner = md
        .write(&b("main"), |d| d.get_text("t").insert_unicode(0, "M"))
        .unwrap();
    assert!(
        inner.is_ok(),
        "write on the advanced shared-copy head succeeded (detached cleared): {inner:?}"
    );
    assert_eq!(
        md.read(&b("main"), |d| d.get_text("t").to_string())
            .unwrap(),
        "MA",
        "the advanced head materialized 'A' and accepted 'M'"
    );
    assert_eq!(
        md.read(&b("feat"), |d| d.get_text("t").to_string())
            .unwrap(),
        "",
        "feat, still on the shared root, is unaffected"
    );
}

#[test]
fn repo_create_write_read_free_creation() {
    let repo = BranchingDocRepo::open().unwrap();
    let doc = repo.open_doc("G".into());
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "main").unwrap())
        .unwrap();

    // Free creation: draft shares main's content head until it writes.
    repo.create_branch(&b("draft"), &b(GENESIS_BRANCH)).unwrap();
    assert!(repo.branches().contains(&b("draft")));
    assert_eq!(
        doc.branch("draft")
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "main",
        "draft sees main's content (shared head, no copy)"
    );

    // draft diverges (copy-on-divergence); main is unaffected.
    doc.branch("draft")
        .write(|h| h.get_text("t").insert_unicode(4, "-draft").unwrap())
        .unwrap();
    assert_eq!(
        doc.branch("draft")
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "main-draft"
    );
    assert_eq!(
        doc.branch(GENESIS_BRANCH)
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "main",
        "main unaffected by draft's divergent write"
    );
}

#[test]
fn merge_does_not_drop_ops() {
    let repo = BranchingDocRepo::open().unwrap();
    let doc = repo.open_doc("G".into());
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();
    repo.create_branch(&b("feature"), &b(GENESIS_BRANCH))
        .unwrap();

    // Both branches diverge with a concurrent edit at the same position.
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(4, "M").unwrap())
        .unwrap();
    doc.branch("feature")
        .write(|h| h.get_text("t").insert_unicode(4, "F").unwrap())
        .unwrap();
    assert!(!doc.contains(&b(GENESIS_BRANCH), &b("feature")).unwrap());

    let outcome = doc.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap();
    assert_eq!(outcome, MergeOutcome::Merged);

    // main now contains BOTH edits -- no dropped op.
    let text = doc
        .branch(GENESIS_BRANCH)
        .read(|h| h.get_text("t").to_string())
        .unwrap();
    assert_eq!(
        text.chars().count(),
        6,
        "base + both 1-char edits, got {text:?}"
    );
    assert!(
        text.starts_with("base") && text.contains('M') && text.contains('F'),
        "merge kept both branches' ops, got {text:?}"
    );
    // feature is now contained in main.
    assert!(doc.contains(&b(GENESIS_BRANCH), &b("feature")).unwrap());
}

#[test]
fn merge_fast_forward_and_already_contained() {
    let repo = BranchingDocRepo::open().unwrap();
    let doc = repo.open_doc("G".into());
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "x").unwrap())
        .unwrap();
    repo.create_branch(&b("feature"), &b(GENESIS_BRANCH))
        .unwrap();
    // Only feature advances; main is behind -> fast-forward.
    doc.branch("feature")
        .write(|h| h.get_text("t").insert_unicode(1, "y").unwrap())
        .unwrap();
    assert_eq!(
        doc.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap(),
        MergeOutcome::FastForward
    );
    assert_eq!(
        doc.branch(GENESIS_BRANCH)
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "xy"
    );
    // Merging again: already contained.
    assert_eq!(
        doc.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap(),
        MergeOutcome::AlreadyContained
    );
}

// ------------------------------------------------------------------
// N1: two-peer Delegated content-sync convergence (the raison d'etre).
// ------------------------------------------------------------------

#[test]
fn two_peer_content_sync_converges() {
    // Peer A edits branch main on doc G; export A's index + content; peer B
    // imports both. B's `after_import` (Delegated) must re-resolve main so its
    // branch head converges to A's ACTUAL content -- not merely a matching
    // index head_count.
    let a = BranchingDocRepo::open().unwrap();
    let ga = a.open_doc("G".into());
    ga.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "hello").unwrap())
        .unwrap();
    let a_index = a.index().export(ExportMode::all_updates()).unwrap();
    let a_content = ga.export(ExportMode::all_updates()).unwrap();

    let bb = BranchingDocRepo::open().unwrap();
    let gb = bb.open_doc("G".into());
    assert_eq!(
        gb.branch(GENESIS_BRANCH)
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "",
        "B starts empty"
    );

    // Sync: index first (B learns where main is), then content (the ops).
    bb.index().import(&a_index).unwrap();
    gb.import(&a_content).unwrap();

    // Inspect main's bound head DIRECTLY (no branch read/resolve), so this
    // isolates `after_import`'s EAGER re-resolve/advance -- a lazy read would
    // re-resolve and converge on its own, masking the ingest.
    let head_id = gb.bound_head(&GENESIS_BRANCH.into()).expect("main bound");
    let head_content = gb.head_doc(head_id).unwrap().get_text("t").to_string();
    assert_eq!(
        head_content, "hello",
        "after_import eagerly advanced main's head to A's actual content"
    );
}

#[test]
fn delete_branch_no_dangling_registry() {
    let repo = BranchingDocRepo::open().unwrap();
    let doc = repo.open_doc("G".into());
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();
    repo.create_branch(&b("draft"), &b(GENESIS_BRANCH)).unwrap();
    // Access draft on the content doc so it binds AND diverges onto its own
    // head (refs 1) -- the case a broken delete would leave dangling.
    doc.branch("draft")
        .write(|h| h.get_text("t").insert_unicode(4, "-d").unwrap())
        .unwrap();
    assert!(repo.branches().contains(&b("draft")));
    assert!(
        repo.index().bound_head(&b("draft")).is_some(),
        "index binds draft"
    );
    assert!(
        doc.bound_head(&b("draft")).is_some(),
        "content doc binds draft"
    );

    repo.delete_branch(&b("draft")).unwrap();

    // No dangling / desynced entry anywhere.
    assert!(
        !repo.branches().contains(&b("draft")),
        "draft removed from index lineage"
    );
    assert_eq!(
        repo.index().bound_head(&b("draft")),
        None,
        "index: draft unbound (no dangling bound entry)"
    );
    assert_eq!(
        doc.bound_head(&b("draft")),
        None,
        "content doc: draft unbound (no dangling bound entry)"
    );
    // Pinned root intact; the other branch is unaffected.
    assert!(doc.head_doc(0).is_some(), "pinned root intact");
    assert_eq!(
        doc.branch(GENESIS_BRANCH)
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "base",
        "main unaffected by the delete"
    );
}

#[test]
fn create_branch_at_writable_fork_at_historical_frontier() {
    let repo = BranchingDocRepo::open().unwrap();
    let doc = repo.open_doc("G".into());
    // main: "A" then "AB"; capture the frontier after "A".
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
    let at_a = doc.frontier_of(&b(GENESIS_BRANCH)).unwrap();
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(1, "B").unwrap())
        .unwrap();

    // Fork a new branch at the historical "A" frontier.
    doc.create_branch_at(&b("hist"), &at_a).unwrap();
    assert!(repo.branches().contains(&b("hist")));
    // hist materializes to the CHOSEN frontier ("A"), not main's current "AB".
    assert_eq!(
        doc.branch("hist")
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "A"
    );
    // head_of resolves hist's index head to a valid id.
    assert!(repo.index().head_of(&b("hist")).is_ok());
    // It is a WRITABLE fork (RFP use-case 3): a write succeeds and diverges.
    doc.branch("hist")
        .write(|h| h.get_text("t").insert_unicode(1, "X").unwrap())
        .unwrap();
    assert_eq!(
        doc.branch("hist")
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "AX"
    );
    // main is unaffected.
    assert_eq!(
        doc.branch(GENESIS_BRANCH)
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "AB"
    );
}

#[test]
fn subscribe_survives_rebind() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    md.bind(&b("feat"), 0); // root shared, refs 2

    // Subscribe on feat BEFORE it diverges (installed on the shared root).
    let count = Arc::new(AtomicUsize::new(0));
    let c2 = count.clone();
    let sub = md
        .branch("feat")
        .subscribe_root(Arc::new(move |_ev: crate::event::DiffEvent| {
            c2.fetch_add(1, SeqCst);
        }))
        .unwrap();

    // feat writes -> copy-on-divergence -> feat rebinds to a NEW head. The
    // subscription must re-install on the new head and fire for this write.
    md.branch("feat")
        .write(|h| h.get_text("t").insert_unicode(0, "F").unwrap())
        .unwrap();
    assert!(
        count.load(SeqCst) >= 1,
        "subscription fired on feat's new head after copy-on-divergence rebind"
    );
    drop(sub);
}

#[test]
fn create_branch_at_multi_peer_parent_excludes_other_peers() {
    // Peer A: main writes "A1" on G.
    let a = BranchingDocRepo::open().unwrap();
    let ga = a.open_doc("G".into());
    ga.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "A1").unwrap())
        .unwrap();
    let at_a = ga.frontier_of(&b(GENESIS_BRANCH)).unwrap(); // only A's op
    let a_index = a.index().export(ExportMode::all_updates()).unwrap();
    let a_content = ga.export(ExportMode::all_updates()).unwrap();

    // Peer B: main writes "B1", then syncs A in -> main is MULTI-PEER.
    let bb = BranchingDocRepo::open().unwrap();
    let gb = bb.open_doc("G".into());
    gb.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "B1").unwrap())
        .unwrap();
    bb.index().import(&a_index).unwrap();
    gb.import(&a_content).unwrap();
    let main_content = gb
        .branch(GENESIS_BRANCH)
        .read(|h| h.get_text("t").to_string())
        .unwrap();
    assert!(
        main_content.contains("A1") && main_content.contains("B1"),
        "parent is multi-peer: {main_content:?}"
    );

    // Fork at A's single-peer frontier: must be A's op ONLY, not the parent's
    // full multi-peer content (the additive record_frontier bug).
    gb.create_branch_at(&b("hist"), &at_a).unwrap();
    assert_eq!(
        gb.branch("hist")
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "A1",
        "fork at [A] excludes peer B's op"
    );
}

#[test]
fn create_branch_at_existing_name_errors() {
    let repo = BranchingDocRepo::open().unwrap();
    let doc = repo.open_doc("G".into());
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "x").unwrap())
        .unwrap();
    let f = doc.frontier_of(&b(GENESIS_BRANCH)).unwrap();
    // genesis already exists -> error (no silent reposition).
    assert!(doc.create_branch_at(&b(GENESIS_BRANCH), &f).is_err());
    // a fresh name works; re-creating it errors.
    doc.create_branch_at(&b("hist"), &f).unwrap();
    assert!(
        doc.create_branch_at(&b("hist"), &f).is_err(),
        "re-create of an existing branch errors"
    );
}
