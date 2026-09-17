use super::*;
use crate::encoding::ExportMode;
use crate::handler::HandlerTrait;
use crate::lock::{LockKind, LoroLockGroup};
use crate::version::Frontiers;
use crate::LoroDoc;
use loro_common::{LoroError, LoroResult, PeerID, ID};
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

// ------------------------------------------------------------------
// Behavior spike: can the same index-doc peer land in >1 branch's
// `lineage`, and if so does `target()` compute a wrong frontier?
// ------------------------------------------------------------------

#[test]
fn index_create_existing_branch_errors_and_no_peer_serves_two_branches() {
    // S2. Single session, many branches (some forked from each other), plus
    // content writes that call record_head. Re-forking an EXISTING name must
    // error (the existence guard, mirroring create_branch_at) instead of
    // silently minting a second creation marker for it; and via the normal
    // API no index peer ever attributes to two branches.
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());

    repo.create_branch(&b("b1"), &b(GENESIS_BRANCH)).unwrap();
    repo.create_branch(&b("b2"), &b(GENESIS_BRANCH)).unwrap();
    repo.create_branch(&b("b1a"), &b("b1")).unwrap();
    let tips_before = repo.index().tips();
    assert!(
        matches!(
            repo.create_branch(&b("b1a"), &b("b2")),
            Err(LoroError::ArgErr(_))
        ),
        "re-forking an existing branch name errors"
    );
    assert_eq!(
        repo.index().tips(),
        tips_before,
        "a refused re-fork moves no branch's frontier"
    );

    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "main").unwrap())
        .unwrap();
    g.branch("b1")
        .write(|h| h.get_text("t").insert_unicode(0, "one").unwrap())
        .unwrap();
    g.branch("b2")
        .write(|h| h.get_text("t").insert_unicode(0, "two").unwrap())
        .unwrap();
    g.branch("b1a")
        .write(|h| h.get_text("t").insert_unicode(0, "onea").unwrap())
        .unwrap();

    let idx = repo.index();
    let attr = idx.policy().attribution(idx).lock();
    let multi: Vec<(PeerID, usize)> = attr
        .runs
        .iter()
        .filter(|(_, runs)| runs.len() != 1)
        .map(|(p, runs)| (*p, runs.len()))
        .collect();
    assert!(
        multi.is_empty(),
        "via the normal API every index peer has exactly one attribution run: {multi:?}"
    );
    assert!(attr.quarantined.is_empty());
    assert_eq!(attr.tips.len(), 4, "main, b1, b2, b1a");
}

#[test]
fn index_crafted_shared_peer_across_two_markers_attributes_causally() {
    // S1. A hand-crafted history (bypassing the IndexDoc API) in which ONE
    // peer writes the creation marker of `alpha`, then of `beta`, then an
    // unrelated op. Under peer-roster attribution both branches' frontiers
    // collapsed onto the unrelated op (the peer's latest op anywhere). Under
    // causal attribution each op belongs to the nearest marker in its past:
    // tips[alpha] = [create(alpha)], tips[beta] = [unrelated].
    const SHARED_PEER: PeerID = 999;
    let ext = LoroDoc::new();
    ext.start_auto_commit();
    ext.set_peer_id(SHARED_PEER).unwrap();
    ext.get_list("lineage:alpha").push(SHARED_PEER as i64).unwrap();
    ext.commit_then_renew();
    let create_alpha = ext.state_frontiers();
    ext.get_list("lineage:beta").push(SHARED_PEER as i64).unwrap();
    ext.commit_then_renew();
    ext.get_map("docs").insert("unrelated", "poison").unwrap();
    ext.commit_then_renew();
    let unrelated_tip = ext.state_frontiers();
    let payload = ext.export(ExportMode::all_updates()).unwrap();

    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    idx.import(&payload).unwrap();

    assert!(idx.branches().contains(&b("alpha")));
    assert!(idx.branches().contains(&b("beta")));
    let target_alpha = idx.policy().target(&idx, &b("alpha")).unwrap();
    let target_beta = idx.policy().target(&idx, &b("beta")).unwrap();
    assert_eq!(
        target_alpha, create_alpha,
        "alpha's frontier is its own creation marker, not the peer's latest op"
    );
    assert_eq!(
        target_beta, unrelated_tip,
        "beta's frontier is the unrelated op, whose nearest marker names beta"
    );
    assert_ne!(target_alpha, target_beta, "the two branches are kept apart");
    assert!(
        idx.quarantine_report().is_empty(),
        "a well-formed (if odd) history quarantines nothing"
    );
}

#[test]
fn spike_scenario2_diamond_topology_forks_merges_are_stable() {
    // A (genesis) -> fork B, fork C. Write on B, write on C. Merge B into A,
    // merge C into A. Fork D from A's now-merged state. Verify D inherits
    // both contributions (nothing dropped, nothing duplicated) and that
    // target(B)/target(C) remain stable (still just their own lineage) after
    // A absorbed them.
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();
    let a_after_base = g.frontier_of(&b(GENESIS_BRANCH)).unwrap();

    repo.create_branch(&b("B"), &b(GENESIS_BRANCH)).unwrap();
    repo.create_branch(&b("C"), &b(GENESIS_BRANCH)).unwrap();

    g.branch("B")
        .write(|h| h.get_text("t").insert_unicode(4, "-b").unwrap())
        .unwrap();
    g.branch("C")
        .write(|h| h.get_text("t").insert_unicode(4, "-c").unwrap())
        .unwrap();
    let b_frontier_before = g.frontier_of(&b("B")).unwrap();
    let c_frontier_before = g.frontier_of(&b("C")).unwrap();

    assert_eq!(
        g.merge(&b(GENESIS_BRANCH), &b("B")).unwrap(),
        MergeOutcome::FastForward
    );
    assert_eq!(
        g.merge(&b(GENESIS_BRANCH), &b("C")).unwrap(),
        MergeOutcome::Merged
    );

    let a_text = g
        .branch(GENESIS_BRANCH)
        .read(|h| h.get_text("t").to_string())
        .unwrap();
    assert!(
        a_text.contains("-b") && a_text.contains("-c") && a_text.starts_with("base"),
        "A absorbed both B's and C's contributions, got {a_text:?}"
    );
    assert_eq!(
        a_text.chars().count(),
        8,
        "base(4) + -b(2) + -c(2), nothing dropped/duplicated, got {a_text:?}"
    );

    // Fork D from A's now-merged state.
    repo.create_branch(&b("D"), &b(GENESIS_BRANCH)).unwrap();
    let d_text = g
        .branch("D")
        .read(|h| h.get_text("t").to_string())
        .unwrap();
    assert_eq!(
        d_text, a_text,
        "D inherits exactly A's merged state, both contributions present once"
    );

    // B and C themselves are STABLE: still just their own (unmerged) content,
    // unaffected by A's absorption of them.
    assert_eq!(
        g.frontier_of(&b("B")).unwrap(),
        b_frontier_before,
        "B's own frontier unaffected by being merged into A"
    );
    assert_eq!(
        g.frontier_of(&b("C")).unwrap(),
        c_frontier_before,
        "C's own frontier unaffected by being merged into A"
    );
    assert_eq!(
        g.branch("B").read(|h| h.get_text("t").to_string()).unwrap(),
        "base-b",
        "B still reads only its own lineage's content"
    );
    assert_eq!(
        g.branch("C").read(|h| h.get_text("t").to_string()).unwrap(),
        "base-c",
        "C still reads only its own lineage's content"
    );
    // Sanity: A's post-base frontier is properly an ancestor (both merges
    // strictly advanced it).
    assert_ne!(a_after_base, g.frontier_of(&b(GENESIS_BRANCH)).unwrap());
}

#[test]
fn spike_scenario3_repeated_merge_into_same_target_no_drop_no_dup() {
    // Fork b from main, write, merge b into main. Continue writing on b.
    // Merge b into main AGAIN. Confirm main reflects both merges (nothing
    // dropped, nothing double-counted), i.e. merge does not assume "b is
    // merged only once".
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();
    repo.create_branch(&b("feature"), &b(GENESIS_BRANCH))
        .unwrap();

    g.branch("feature")
        .write(|h| h.get_text("t").insert_unicode(4, "-1").unwrap())
        .unwrap();
    assert_eq!(
        g.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap(),
        MergeOutcome::FastForward
    );
    assert_eq!(
        g.branch(GENESIS_BRANCH)
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "base-1"
    );

    // Continue writing on feature past the point already merged.
    g.branch("feature")
        .write(|h| h.get_text("t").insert_unicode(6, "-2").unwrap())
        .unwrap();
    // main also advances independently, concurrently with feature's second
    // edit, so the second merge is a real join (not another fast-forward).
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(6, "-m").unwrap())
        .unwrap();

    let outcome2 = g.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap();
    assert_eq!(outcome2, MergeOutcome::Merged, "second merge is a real join");

    let final_text = g
        .branch(GENESIS_BRANCH)
        .read(|h| h.get_text("t").to_string())
        .unwrap();
    assert!(
        final_text.contains("-1") && final_text.contains("-2") && final_text.contains("-m"),
        "both merges' content present, got {final_text:?}"
    );
    assert_eq!(
        final_text.chars().count(),
        10,
        "base(4) + -1(2) + -2(2) + -m(2) = 10, no drop/dup across repeated merges, \
         got {final_text:?}"
    );
    // A third merge (already contained) must be a true no-op.
    assert_eq!(
        g.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap(),
        MergeOutcome::AlreadyContained
    );
    assert_eq!(
        g.branch(GENESIS_BRANCH)
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        final_text,
        "a no-op AlreadyContained merge changes nothing"
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

#[test]
fn index_snapshot_restore_preserves_lineage_for_every_branch() {
    // Behavior spike: does `SelfRooted`'s runtime `lineage` cache (rebuilt by
    // replaying `after_import` over `ImportStatus::success`) survive a
    // Snapshot export/import into a COMPLETELY FRESH `IndexDoc`? Build a
    // non-trivial multi-branch history (3 forks, one nested, one genuine
    // divergent merge, and one branch with a MULTI-PEER lineage from two
    // independent sessions), snapshot it, import the snapshot into a doc
    // that has NEVER seen an incremental update, and compare `branches()` /
    // `target(b)` before vs after exactly.
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());

    // main (genesis) gets a base write.
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();

    // Two forks off genesis, plus a NESTED fork off one of them.
    repo.create_branch(&b("feat1"), &b(GENESIS_BRANCH)).unwrap();
    repo.create_branch(&b("feat2"), &b(GENESIS_BRANCH)).unwrap();
    repo.create_branch(&b("feat1-nested"), &b("feat1")).unwrap();

    g.branch("feat1")
        .write(|h| h.get_text("t").insert_unicode(4, "-f1").unwrap())
        .unwrap();
    g.branch("feat2")
        .write(|h| h.get_text("t").insert_unicode(4, "-f2").unwrap())
        .unwrap();
    g.branch("feat1-nested")
        .write(|h| h.get_text("t").insert_unicode(4, "-n").unwrap())
        .unwrap();

    // A genuine divergent merge (not a fast-forward): main advances
    // independently of feat2 before the merge, forcing a real join.
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(4, "-m").unwrap())
        .unwrap();
    let outcome = g.merge(&b(GENESIS_BRANCH), &b("feat2")).unwrap();
    assert_eq!(
        outcome,
        MergeOutcome::Merged,
        "the merge must be a real divergent join, not a fast-forward"
    );

    // A branch with a MULTI-PEER lineage: a second, fully independent
    // session creates the SAME branch name off its own genesis, then syncs
    // with the first via a plain index-history import (mirrors
    // `index_target_is_join_of_lineage_peers`). This exercises the exact
    // "peer roster per branch" attribution the spike is questioning.
    let session_b = IndexDoc::new(SelfRooted::new());
    session_b.init_genesis(&b(GENESIS_BRANCH)).unwrap();
    session_b
        .create_index_branch(&b("shared"), &b(GENESIS_BRANCH))
        .unwrap();
    session_b
        .record_head(&b("shared"), &d("OTHER"), 999, 3)
        .unwrap();
    let b_updates = session_b.export(ExportMode::all_updates()).unwrap();
    repo.create_branch(&b("shared"), &b(GENESIS_BRANCH)).unwrap();
    repo.index().import(&b_updates).unwrap();
    assert_eq!(
        repo.index().tips()[&b("shared")].len(),
        2,
        "sanity: 'shared' must be a genuine two-lineage (two-id) frontier before the snapshot"
    );

    // --- BEFORE baseline: branches() + target(b) for every branch --------
    let idx = repo.index();
    let mut before_branches = idx.branches();
    before_branches.sort();
    let before_targets: Vec<(BranchId, LoroResult<Frontiers>)> = before_branches
        .iter()
        .map(|br| (br.clone(), idx.policy().target(idx, br)))
        .collect();
    for (br, t) in &before_targets {
        assert!(t.is_ok(), "pre-snapshot target({br}) must resolve");
    }

    // --- Export a SNAPSHOT (not just updates) and import into a doc that --
    // --- has NEVER seen ANY incremental update (only the snapshot bytes). -
    let snap = idx.export(ExportMode::Snapshot).unwrap();
    let restored = IndexDoc::new(SelfRooted::new());
    let status = restored.import(&snap).unwrap();
    // The concern under test: does a snapshot import's `ImportStatus::success`
    // cover the FULL history (so `after_import`'s attribution walk sees every
    // `lineage:<b>` op), or does it come back empty/partial?
    assert!(
        !status.success.is_empty(),
        "snapshot import's ImportStatus::success must be non-empty for after_import \
         to rebuild lineage at all"
    );

    let mut after_branches = restored.branches();
    after_branches.sort();
    assert_eq!(
        before_branches, after_branches,
        "branches() must match exactly after a fresh snapshot restore"
    );

    for (br, before) in &before_targets {
        let after = restored.policy().target(&restored, br);
        match (before, after) {
            (Ok(bf), Ok(af)) => assert_eq!(
                bf, &af,
                "target({br}) frontier must match exactly before vs after snapshot restore"
            ),
            other => panic!("target({br}) resolution mismatch before/after: {other:?}"),
        }
    }
}

// ------------------------------------------------------------------
// S0: a local record on a REMOTELY-DISCOVERED branch must survive the next
// resolve and reach the peer. Control: `index_import_then_record_converges`
// (identical, except session B CREATES `feat` locally before importing).
// ------------------------------------------------------------------

#[test]
fn index_remote_discovered_branch_local_record_survives() {
    // Session A creates `feat` and records doc D on it.
    let a = IndexDoc::new(SelfRooted::new());
    a.init_genesis(&b("main")).unwrap();
    a.create_index_branch(&b("feat"), &b("main")).unwrap();
    a.record_head(&b("feat"), &d("D"), 111, 7).unwrap();
    let a_updates = a.export(ExportMode::all_updates()).unwrap();

    // Session B never creates `feat`: it LEARNS it from A's import (the
    // `loro_repo.ts` `ensureFsBranch` path), so its `feat` index head is minted
    // by the resolve MATERIALIZE arm, which no `lineage:feat` op names.
    let bb = IndexDoc::new(SelfRooted::new());
    bb.init_genesis(&b("main")).unwrap();
    bb.import(&a_updates).unwrap();
    let after_import = bb.recorded_ids(&b("feat"), &d("D")).unwrap();
    assert_eq!(
        after_import,
        vec![ID::new(111, 7)],
        "B discovered feat and materialized A's record"
    );

    // B records locally on the discovered branch.
    bb.record_head(&b("feat"), &d("D"), 222, 9).unwrap();
    let mut after_record = bb.recorded_ids(&b("feat"), &d("D")).unwrap();
    after_record.sort();
    assert_eq!(
        after_record,
        vec![ID::new(111, 7), ID::new(222, 9)],
        "B's local record on a remotely-discovered branch is LOST on B: {after_record:?}"
    );

    // ... and the record reaches A.
    let b_updates = bb.export(ExportMode::all_updates()).unwrap();
    a.import(&b_updates).unwrap();
    let mut on_a = a.recorded_ids(&b("feat"), &d("D")).unwrap();
    on_a.sort();
    assert_eq!(
        on_a,
        vec![ID::new(111, 7), ID::new(222, 9)],
        "B's record never reached A: {on_a:?}"
    );
}

// ------------------------------------------------------------------
// S4: the single-pass lamport-ordered fold attributes EVERY change of a
// multi-peer, multi-branch history (diamond + repeated merge + two-session
// same-branch) on a cold import, with zero quarantines and tips equal to the
// source's.
// ------------------------------------------------------------------

/// Build the S4 history on a fresh repo and return its index.
fn s4_history() -> BranchingDocRepo {
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();

    // Diamond: B, C fork from main; both merge into main; D forks from the
    // post-merge main and writes.
    repo.create_branch(&b("B"), &b(GENESIS_BRANCH)).unwrap();
    repo.create_branch(&b("C"), &b(GENESIS_BRANCH)).unwrap();
    g.branch("B")
        .write(|h| h.get_text("t").insert_unicode(4, "-b").unwrap())
        .unwrap();
    g.branch("C")
        .write(|h| h.get_text("t").insert_unicode(4, "-c").unwrap())
        .unwrap();
    assert_eq!(
        g.merge(&b(GENESIS_BRANCH), &b("B")).unwrap(),
        MergeOutcome::FastForward
    );
    assert_eq!(
        g.merge(&b(GENESIS_BRANCH), &b("C")).unwrap(),
        MergeOutcome::Merged
    );
    repo.create_branch(&b("D"), &b(GENESIS_BRANCH)).unwrap();
    g.branch("D")
        .write(|h| h.get_text("t").insert_unicode(0, "d:").unwrap())
        .unwrap();

    // Repeated merge: feature merged into main twice (fast-forward, then a
    // real join), then a redundant third merge.
    repo.create_branch(&b("feature"), &b(GENESIS_BRANCH))
        .unwrap();
    g.branch("feature")
        .write(|h| h.get_text("t").insert_unicode(0, "f1").unwrap())
        .unwrap();
    assert_eq!(
        g.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap(),
        MergeOutcome::FastForward
    );
    g.branch("feature")
        .write(|h| h.get_text("t").insert_unicode(0, "f2").unwrap())
        .unwrap();
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "m").unwrap())
        .unwrap();
    assert_eq!(
        g.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap(),
        MergeOutcome::Merged
    );
    assert_eq!(
        g.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap(),
        MergeOutcome::AlreadyContained
    );

    // Two sessions on one branch, with the S0 shape on the second: session 2
    // learns `shared` from the repo, records on it (a materialize-arm peer),
    // and the repo takes that record back; then the repo records again on top.
    repo.create_branch(&b("shared"), &b(GENESIS_BRANCH)).unwrap();
    repo.index()
        .record_head(&b("shared"), &d("OTHER"), 500, 1)
        .unwrap();
    let session2 = IndexDoc::new(SelfRooted::new());
    session2.init_genesis(&b(GENESIS_BRANCH)).unwrap();
    session2
        .import(&repo.index().export(ExportMode::all_updates()).unwrap())
        .unwrap();
    session2
        .record_head(&b("shared"), &d("OTHER"), 600, 2)
        .unwrap();
    repo.index()
        .import(&session2.export(ExportMode::all_updates()).unwrap())
        .unwrap();
    repo.index()
        .record_head(&b("shared"), &d("OTHER"), 500, 3)
        .unwrap();
    repo
}

#[test]
fn index_cold_import_of_diamond_repeated_merge_two_session_history_attributes_all() {
    let repo = s4_history();
    let src = repo.index();
    assert!(
        src.quarantine_report().is_empty(),
        "source quarantined: {:?}",
        src.quarantine_report()
    );
    let src_tips = src.tips();
    assert_eq!(src_tips.len(), 6, "main, B, C, D, feature, shared");

    let bytes = src.export(ExportMode::all_updates()).unwrap();
    let cold = IndexDoc::new(SelfRooted::new());
    cold.import(&bytes).unwrap();

    let q = cold.quarantine_report();
    assert!(q.is_empty(), "cold import quarantined {} span(s): {q:?}", q.len());
    assert_eq!(cold.tips(), src_tips, "tips equal on the cold import");
    let mut sb = src.branches();
    sb.sort();
    let mut cb = cold.branches();
    cb.sort();
    assert_eq!(sb, cb);
    // The two-session branch reads back both sessions' records on the cold side.
    let mut ids = cold.recorded_ids(&b("shared"), &d("OTHER")).unwrap();
    ids.sort();
    assert_eq!(ids, vec![ID::new(500, 3), ID::new(600, 2)]);
}

// ------------------------------------------------------------------
// S7: quarantine is loud and local. A crafted NON-marker change whose deps
// span two branches' frontiers is reported, its ops stay in the op log, and
// neither branch's tips moves.
// ------------------------------------------------------------------

#[test]
fn index_change_spanning_two_branches_is_quarantined_without_moving_tips() {
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    idx.create_index_branch(&b("alpha"), &b("main")).unwrap();
    idx.create_index_branch(&b("beta"), &b("main")).unwrap();
    idx.record_head(&b("alpha"), &d("A"), 1, 1).unwrap();
    idx.record_head(&b("beta"), &d("B"), 2, 2).unwrap();
    let tips_before = idx.tips();

    // A plain LoroDoc holding the index's whole history sits at the join of
    // every branch tip; a local write there depends on alpha's AND beta's tips.
    let ext = LoroDoc::new();
    ext.start_auto_commit();
    ext.import(&idx.export(ExportMode::all_updates()).unwrap())
        .unwrap();
    let ext_deps = ext.state_frontiers();
    assert!(
        ext_deps.contains(&tips_before[&b("alpha")].as_single().unwrap())
            && ext_deps.contains(&tips_before[&b("beta")].as_single().unwrap()),
        "the crafted change depends on both branches' tips: {ext_deps:?}"
    );
    ext.get_map("docs").insert("poison", 1).unwrap();
    ext.commit_then_renew();
    let poison = ext.state_frontiers().as_single().unwrap();
    let ext_bytes = ext.export(ExportMode::updates(&idx.oplog_vv())).unwrap();

    idx.import(&ext_bytes).unwrap();

    let q = idx.quarantine_report();
    assert_eq!(q.len(), 1, "exactly the crafted change is quarantined: {q:?}");
    assert_eq!(q[0].0, poison);
    assert!(
        matches!(&q[0].1, QuarantineReason::DepsDisagree(bs) if bs.len() == 2),
        "reason names the disagreeing branches: {:?}",
        q[0].1
    );
    assert_eq!(idx.tips(), tips_before, "no branch's tips moved");
    assert!(
        idx.oplog_vv().get_last(poison.peer) == Some(poison.counter),
        "the quarantined ops stay in the op log"
    );
    // Both branches still read their own records only.
    assert_eq!(
        idx.recorded_ids(&b("alpha"), &d("A")).unwrap(),
        vec![ID::new(1, 1)]
    );
    assert!(idx.recorded_ids(&b("alpha"), &d("B")).unwrap().is_empty());
}

// ------------------------------------------------------------------
// Phase 2: the `branch.name` root-map marker replaces the per-branch
// `lineage:<b>` root list. Two root containers total, no branch-name charset
// constraint, and a DUAL-READ fold so a pre-Phase-2 history still attributes.
// ------------------------------------------------------------------

/// Build a small multi-branch index history as a raw op stream, using either
/// the legacy `lineage:<b>` marker (`new == false`) or the Phase 2 `branch.name`
/// marker (`new == true`). The peer/counter layout is IDENTICAL between the two
/// (only the marker op's shape differs), so the two histories fold to
/// byte-identical `tips` -- exactly the dual-read guarantee: a pre-Phase-2
/// history yields the same attribution under the current engine as its Phase 2
/// equivalent.
fn dual_read_history(new: bool) -> Vec<Vec<u8>> {
    fn marker(doc: &LoroDoc, new: bool, name: &str) {
        if new {
            doc.get_map("branch").insert("name", name).unwrap();
        } else {
            doc.get_list(format!("lineage:{name}").as_str())
                .push(0i64)
                .unwrap();
        }
        doc.commit_then_renew();
    }

    // main: peer 100, one creation marker.
    let m = LoroDoc::new();
    m.start_auto_commit();
    m.set_peer_id(100).unwrap();
    marker(&m, new, "main");
    let m_bytes = m.export(ExportMode::all_updates()).unwrap();

    // feat: peer 200, forks main (imports its history), then its marker and a
    // NON-marker op (a plain `docs` write, attributed to feat by its deps).
    let f = LoroDoc::new();
    f.start_auto_commit();
    f.set_peer_id(200).unwrap();
    f.import(&m_bytes).unwrap();
    marker(&f, new, "feat");
    f.get_map("docs").insert("x", 1).unwrap();
    f.commit_then_renew();
    let f_bytes = f.export(ExportMode::all_updates()).unwrap();

    // feat2: peer 300, an independent fork off main, marker only.
    let f2 = LoroDoc::new();
    f2.start_auto_commit();
    f2.set_peer_id(300).unwrap();
    f2.import(&m_bytes).unwrap();
    marker(&f2, new, "feat2");
    let f2_bytes = f2.export(ExportMode::all_updates()).unwrap();

    vec![m_bytes, f_bytes, f2_bytes]
}

#[test]
fn index_dual_read_old_and_new_marker_fold_identically() {
    // The transition fold reads BOTH marker encodings. A history built with the
    // legacy `lineage:<b>` marker and its `branch.name` twin (same peers, same
    // counters) fold to IDENTICAL tips under the current engine, with no
    // quarantine on either -- the guarantee that existing pre-Phase-2 dogfood
    // and snapshot data still attributes correctly.
    let fold = |hs: Vec<Vec<u8>>| {
        let idx = IndexDoc::new(SelfRooted::new());
        for h in &hs {
            idx.import(h).unwrap();
        }
        idx
    };
    let old = fold(dual_read_history(false));
    let new = fold(dual_read_history(true));

    assert!(
        old.quarantine_report().is_empty(),
        "old-encoding fold quarantined: {:?}",
        old.quarantine_report()
    );
    assert!(
        new.quarantine_report().is_empty(),
        "new-encoding fold quarantined: {:?}",
        new.quarantine_report()
    );

    assert_eq!(
        old.tips(),
        new.tips(),
        "legacy lineage:<b> and branch.name markers fold to identical tips"
    );

    let mut ob = old.branches();
    ob.sort();
    let mut nb = new.branches();
    nb.sort();
    assert_eq!(ob, nb);
    assert_eq!(ob, vec![b("feat"), b("feat2"), b("main")]);

    // Concrete frontiers (identical id layout in both encodings): main and feat2
    // are single-op peers; feat's marker + non-marker op merge into one change,
    // so its tip is the change's last op.
    assert_eq!(old.tips()[&b("main")].as_single(), Some(ID::new(100, 0)));
    assert_eq!(old.tips()[&b("feat")].as_single(), Some(ID::new(200, 1)));
    assert_eq!(old.tips()[&b("feat2")].as_single(), Some(ID::new(300, 0)));
}

#[test]
fn index_arbitrary_branch_names_round_trip() {
    // Phase 2 retires the branch-name charset constraint (`branch_name_codec`):
    // the name travels as a map VALUE, not a container id, so a name with
    // spaces, unicode, ':' or '/' -- all restricted by the legacy `lineage:<b>`
    // container-name codec -- round-trips through creation, snapshot export,
    // cold import, and reads.
    let names = ["a branch with spaces", "café ☕ 名前", "boc/abc123", "a:b:c"];
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    for (i, n) in names.iter().enumerate() {
        idx.create_index_branch(&b(n), &b("main")).unwrap();
        idx.record_head(&b(n), &d("G"), 1000 + i as u64, i as i32 + 1)
            .unwrap();
    }

    let snap = idx.export(ExportMode::Snapshot).unwrap();
    let restored = IndexDoc::new(SelfRooted::new());
    restored.import(&snap).unwrap();

    assert!(restored.quarantine_report().is_empty());
    for (i, n) in names.iter().enumerate() {
        assert!(
            restored.branches().contains(&b(n)),
            "branch {n:?} restored from the snapshot"
        );
        assert!(
            restored
                .policy()
                .target(&restored, &b(n))
                .unwrap()
                .as_single()
                .is_some(),
            "target({n:?}) resolves"
        );
        assert_eq!(
            restored.recorded_ids(&b(n), &d("G")).unwrap(),
            vec![ID::new(1000 + i as u64, i as i32 + 1)],
            "record for {n:?} round-trips"
        );
    }
}

#[test]
fn index_root_container_count_is_two_regardless_of_branch_count() {
    // Phase 2 collapses the wire encoding to TWO root containers total (`branch`
    // + `docs`), independent of branch count: the legacy `1 + branches_alltime`
    // (one `lineage:<b>` root list per branch) is gone.
    use crate::arena::LoadAllFlag;
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    for i in 0..50 {
        idx.create_index_branch(&b(&format!("feat{i}")), &b("main"))
            .unwrap();
    }
    idx.record_head(&b("main"), &d("G"), 1, 1).unwrap();

    let roots = idx.arena.top_level_root_containers(LoadAllFlag);
    let names: Vec<_> = roots.iter().filter_map(|c| idx.arena.idx_to_id(*c)).collect();
    assert_eq!(
        roots.len(),
        2,
        "exactly `branch` and `docs` after 50 branches, not one root per branch: {names:?}"
    );
}

// ------------------------------------------------------------------
// Phase 3: repo-wide `IndexDoc::merge` is a SINGLE marker op (checkout + join),
// not a per-doc `record_frontier` loop. Spike S5's per-key union of
// `docs[D].heads` is the load-bearing premise; it lands here as a permanent
// regression.
// ------------------------------------------------------------------

/// Total ops in the shared index op log (sum of the per-peer VV counters).
fn total_ops(idx: &IndexDoc) -> i64 {
    idx.oplog_vv().values().map(|c| *c as i64).sum()
}

#[test]
fn index_merge_one_marker_unions_docs_per_key() {
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    idx.create_index_branch(&b("into"), &b("main")).unwrap();
    idx.create_index_branch(&b("from"), &b("main")).unwrap();

    // The same doc D is recorded CONCURRENTLY on both branches (distinct peer
    // keys), plus a doc E touched only on `from`.
    idx.record_head(&b("into"), &d("D"), 111, 5).unwrap();
    idx.record_head(&b("from"), &d("D"), 222, 7).unwrap();
    idx.record_head(&b("from"), &d("E"), 333, 3).unwrap();

    let ops_before = total_ops(&idx);
    let outcome = idx.merge(&b("into"), &b("from")).unwrap();
    assert_eq!(outcome, MergeOutcome::Merged, "into and from diverged");

    // Exactly ONE op was written by the merge (the marker), NOT one-per-doc.
    assert_eq!(
        total_ops(&idx) - ops_before,
        1,
        "merge writes a single index marker op, not one per merged doc"
    );

    // Per-key UNION of docs[D].heads survives (S5 property, now permanent): both
    // concurrent writers' entries are present on `into` after the merge.
    let mut d_ids = idx.recorded_ids(&b("into"), &d("D")).unwrap();
    d_ids.sort();
    assert_eq!(
        d_ids,
        vec![ID::new(111, 5), ID::new(222, 7)],
        "both concurrent same-doc writers survive the merge (per-key VV union)"
    );
    // The doc touched only on `from` crossed the merge for free.
    assert_eq!(
        idx.recorded_ids(&b("into"), &d("E")).unwrap(),
        vec![ID::new(333, 3)]
    );
    // `from` is untouched: a merge never moves the source.
    assert_eq!(
        idx.recorded_ids(&b("from"), &d("D")).unwrap(),
        vec![ID::new(222, 7)]
    );
    assert_eq!(
        idx.recorded_ids(&b("from"), &d("E")).unwrap(),
        vec![ID::new(333, 3)]
    );

    // tips read correctly: into moved to a single marker, from unchanged, both
    // branches still known, nothing quarantined.
    assert!(
        idx.tips()[&b("into")].as_single().is_some(),
        "into's tip is the single merge marker"
    );
    let mut branches = idx.branches();
    branches.sort();
    assert_eq!(branches, vec![b("from"), b("into"), b("main")]);
    assert!(idx.quarantine_report().is_empty());

    // The merge marker is a new KEY on the `branch` map, not a new container:
    // the P2 root-container count (branch + docs) is unchanged.
    assert_eq!(
        idx.arena
            .top_level_root_containers(crate::arena::LoadAllFlag)
            .len(),
        2,
        "a merge marker adds no root container"
    );

    // The merge marker (deps = the cross-branch join) folds cleanly on a COLD
    // import: it is a marker, so the fold attributes it to `into` with no dep
    // agreement needed and no quarantine.
    let bytes = idx.export(ExportMode::all_updates()).unwrap();
    let cold = IndexDoc::new(SelfRooted::new());
    cold.import(&bytes).unwrap();
    assert!(
        cold.quarantine_report().is_empty(),
        "cold import of a merge marker quarantines nothing: {:?}",
        cold.quarantine_report()
    );
    assert_eq!(cold.tips(), idx.tips(), "tips equal on the cold import");
    let mut cold_d = cold.recorded_ids(&b("into"), &d("D")).unwrap();
    cold_d.sort();
    assert_eq!(cold_d, vec![ID::new(111, 5), ID::new(222, 7)]);
}

#[test]
fn index_merge_fast_forward_and_already_contained() {
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    idx.create_index_branch(&b("feat"), &b("main")).unwrap();
    // Only feat advances -> main merging feat is a FAST-FORWARD.
    idx.record_head(&b("feat"), &d("D"), 111, 5).unwrap();

    let ops_before = total_ops(&idx);
    assert_eq!(
        idx.merge(&b("main"), &b("feat")).unwrap(),
        MergeOutcome::FastForward
    );
    assert_eq!(total_ops(&idx) - ops_before, 1, "ff still writes one marker");
    assert_eq!(
        idx.recorded_ids(&b("main"), &d("D")).unwrap(),
        vec![ID::new(111, 5)],
        "main fast-forwarded to feat's record"
    );

    // A second, redundant merge is AlreadyContained and writes NOTHING.
    let ops_now = total_ops(&idx);
    assert_eq!(
        idx.merge(&b("main"), &b("feat")).unwrap(),
        MergeOutcome::AlreadyContained
    );
    assert_eq!(total_ops(&idx) - ops_now, 0, "already-contained writes no op");
}

#[test]
fn repo_merge_branch_content_catches_up() {
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());

    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();
    repo.create_branch(&b("feat"), &b(GENESIS_BRANCH)).unwrap();

    // Diverge: feat and main each edit the same doc.
    g.branch("feat")
        .write(|h| h.get_text("t").insert_unicode(4, "-f").unwrap())
        .unwrap();
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(4, "-m").unwrap())
        .unwrap();

    let idx_ops_before = total_ops(repo.index());
    let outcome = repo.merge_branch(&b(GENESIS_BRANCH), &b("feat")).unwrap();
    assert_eq!(outcome, MergeOutcome::Merged);
    // Exactly one INDEX op for the whole repo-wide merge (the marker).
    assert_eq!(
        total_ops(repo.index()) - idx_ops_before,
        1,
        "repo-wide merge is a single index marker, not one record per doc"
    );

    // main's content head caught up: it now contains BOTH divergent edits.
    let text = g
        .branch(GENESIS_BRANCH)
        .read(|h| h.get_text("t").to_string())
        .unwrap();
    assert!(
        text.contains("base") && text.contains("-f") && text.contains("-m"),
        "merged main content contains base + both edits: {text:?}"
    );
    // feat is untouched by the merge.
    assert!(repo.branches().contains(&b("feat")));
}

// ------------------------------------------------------------------
// Phase 3 FIX (Round 4): the merge marker must ALWAYS emit an op, even when
// `into` already won the branch.name LWW at the join. A same-value map set is a
// no-op -> after_commit never fires -> the whole merge (docs union + from-only
// docs) is silently lost, locally AND on cold import (no op propagates). These
// three tests reproduce that loss on the pre-fix tree.
// ------------------------------------------------------------------

#[test]
fn index_merge_into_lww_winner_still_emits_and_propagates() {
    // merge(feat, main) with into = feat: feat's branch.name set is CAUSALLY
    // AFTER main's (eager copy), so branch.name == "feat" at the join
    // deterministically (no peer luck). Re-setting branch.name = "feat" would be
    // a no-op. The merge marker must still emit so main's record reaches feat and
    // propagates to a cold peer.
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    idx.create_index_branch(&b("feat"), &b("main")).unwrap();
    idx.record_head(&b("main"), &d("D"), 111, 5).unwrap();
    idx.record_head(&b("feat"), &d("E"), 222, 7).unwrap();

    let ops_before = total_ops(&idx);
    assert_eq!(
        idx.merge(&b("feat"), &b("main")).unwrap(),
        MergeOutcome::Merged
    );
    assert_eq!(
        total_ops(&idx) - ops_before,
        1,
        "the merge emits exactly one marker op even when `into` wins the name LWW"
    );

    // main's record crossed into feat (the docs union); feat kept its own.
    assert_eq!(
        idx.recorded_ids(&b("feat"), &d("D")).unwrap(),
        vec![ID::new(111, 5)],
        "main's record must reach feat"
    );
    assert_eq!(
        idx.recorded_ids(&b("feat"), &d("E")).unwrap(),
        vec![ID::new(222, 7)]
    );

    // Propagation: a remote peer sees only the op stream; with no marker op it
    // would never learn the merge happened.
    let cold = IndexDoc::new(SelfRooted::new());
    cold.import(&idx.export(ExportMode::all_updates()).unwrap())
        .unwrap();
    assert!(cold.quarantine_report().is_empty());
    assert_eq!(cold.tips(), idx.tips(), "merge propagates: cold tips equal");
    assert_eq!(
        cold.recorded_ids(&b("feat"), &d("D")).unwrap(),
        vec![ID::new(111, 5)],
        "the merge reaches a cold-importing peer"
    );
}

#[test]
fn index_merge_preserves_docs_in_both_sibling_creation_orders() {
    // Two siblings off main; merge `from` into `into`. Their branch.name sets are
    // CONCURRENT, so the LWW tie-break by (lamport, peer) decides the join value:
    // the later-created sibling wins. Whichever order makes `into` win is the
    // bug's trigger; BOTH orders must preserve the merge after the fix.
    for (first, second) in [("into", "from"), ("from", "into")] {
        let idx = IndexDoc::new(SelfRooted::new());
        idx.init_genesis(&b("main")).unwrap();
        idx.create_index_branch(&b(first), &b("main")).unwrap();
        idx.create_index_branch(&b(second), &b("main")).unwrap();
        idx.record_head(&b("into"), &d("D"), 111, 5).unwrap();
        idx.record_head(&b("from"), &d("E"), 222, 7).unwrap();

        assert_eq!(
            idx.merge(&b("into"), &b("from")).unwrap(),
            MergeOutcome::Merged,
            "order {first}->{second}"
        );
        assert_eq!(
            idx.recorded_ids(&b("into"), &d("E")).unwrap(),
            vec![ID::new(222, 7)],
            "from's record E lost in order {first}->{second}"
        );
        assert_eq!(
            idx.recorded_ids(&b("into"), &d("D")).unwrap(),
            vec![ID::new(111, 5)],
            "into's own record D in order {first}->{second}"
        );

        let cold = IndexDoc::new(SelfRooted::new());
        cold.import(&idx.export(ExportMode::all_updates()).unwrap())
            .unwrap();
        assert_eq!(
            cold.tips(),
            idx.tips(),
            "cold tips equal in order {first}->{second}"
        );
        assert_eq!(
            cold.recorded_ids(&b("into"), &d("E")).unwrap(),
            vec![ID::new(222, 7)],
            "from's record E lost on cold import, order {first}->{second}"
        );
    }
}

#[test]
fn index_repeated_merge_into_same_target_each_emits() {
    // Two merges into the SAME target, with `from` advancing between them. The
    // second merge writes the same branch NAME as the first, so a name-keyed
    // marker would no-op (dropping the second merge). The seq'd merge marker must
    // emit on BOTH.
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    idx.create_index_branch(&b("into"), &b("main")).unwrap();
    idx.create_index_branch(&b("from"), &b("main")).unwrap();

    idx.record_head(&b("from"), &d("D"), 111, 1).unwrap();
    idx.record_head(&b("into"), &d("Z"), 999, 1).unwrap(); // into diverges so 1st merge is real
    assert_eq!(idx.merge(&b("into"), &b("from")).unwrap(), MergeOutcome::Merged);
    assert_eq!(idx.recorded_ids(&b("into"), &d("D")).unwrap(), vec![ID::new(111, 1)]);

    // `from` advances, then a SECOND merge into the same target.
    idx.record_head(&b("from"), &d("E"), 222, 2).unwrap();
    let ops_before = total_ops(&idx);
    assert_eq!(idx.merge(&b("into"), &b("from")).unwrap(), MergeOutcome::Merged);
    assert_eq!(
        total_ops(&idx) - ops_before,
        1,
        "the second merge into the same target still emits one op"
    );
    assert_eq!(
        idx.recorded_ids(&b("into"), &d("E")).unwrap(),
        vec![ID::new(222, 2)],
        "the second merge's new doc reaches into"
    );

    let cold = IndexDoc::new(SelfRooted::new());
    cold.import(&idx.export(ExportMode::all_updates()).unwrap()).unwrap();
    assert_eq!(cold.tips(), idx.tips());
    assert_eq!(cold.recorded_ids(&b("into"), &d("E")).unwrap(), vec![ID::new(222, 2)]);
}
