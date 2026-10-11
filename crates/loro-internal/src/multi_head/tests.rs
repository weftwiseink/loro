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

// Registry / copy mechanics.

#[test]
fn free_creation_shares_head_no_copy() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    md.bind(&main, md.root_head_id());
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
    md.bind(&review, 0);
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
    md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
    let t = md.head_tip(0).unwrap();
    assert!(!t.is_empty(), "tip advanced after the write");
    assert!(md.by_tip_maps(&t, 0), "by_tip re-keyed to the new tip");
    assert!(
        !md.by_tip_maps(&Frontiers::default(), 0),
        "the old empty tip was vacated"
    );
    // An unbound branch targeting that tip shares head 0 via the by_tip lookup arm (no new head).
    md.policy().set_target(&other, t.clone());
    let (id, _) = md.resolve(&other, Intent::Read).unwrap();
    assert_eq!(id, 0);
    assert_eq!(md.head_count(), 1);
    assert_eq!(md.refs_of(&other), Some(2));
}

// Lock-group membership discriminator (+ fresh-group contrast).

#[test]
fn lock_group_shared_docstate_before_oplog_panics() {
    // A copied head's DocState lock shares the root OpLog's group (`fork_in_group`), so a DocState-
    // before-OpLog acquisition across the two heads is caught by the debug order checker.
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    md.bind(&main, 0);
    md.bind(&review, 0);
    md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
    let copy = md.head_doc(md.bound_head(&main).unwrap()).unwrap();
    let root = md.head_doc(0).unwrap();

    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let copy_c = copy.clone();
    let root_c = root.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ds = copy_c.state.lock();
        let _ol = root_c.oplog.lock();
    }));
    std::panic::set_hook(prev);
    assert!(
        result.is_err(),
        "DocState-before-OpLog across two shared-group heads must panic"
    );
    // The panic poisoned the copy's DocState mutex; `forget` the doc so no `Drop` re-locks it
    // (double-panic).
    std::mem::forget(md);
}

#[test]
fn lock_group_fresh_group_docstate_before_oplog_does_not_panic() {
    // CONTRAST: a lock in a fresh group is a different ordering domain, so the inversion goes
    // unseen; this is why a head must join the shared group.
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    let root = md.head_doc(0).unwrap();

    let fresh = LoroLockGroup::new();
    let foreign_docstate = fresh.new_lock(0u8, LockKind::DocState);
    let _ds = foreign_docstate.lock();
    let _ol = root.oplog.lock(); // reaching here without a panic is the assertion
}

// Sink guard.

#[test]
fn sink_guard_mutation_in_read_closure_refuses_and_baseline_corrupts() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    md.bind(&main, 0);
    md.bind(&review, 0);
    assert_eq!(md.is_head_shared(&main), Some(true));

    let res: LoroResult<()> = md
        .read(&main, |d| d.get_text("t").insert_unicode(0, "X"))
        .unwrap();
    assert!(
        matches!(res, Err(LoroError::HeadShared)),
        "mutation inside read on a shared head is refused, got {res:?}"
    );
    assert_eq!(tlen(&md, &review), 0, "co-owner untouched under the guard");

    // Baseline: flipping the same head to Private (a refs>1 head writable) lets the identical
    // mutation land and corrupt the co-owner.
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
    md.bind(&main, 0);
    let leaked = md.head_doc(0).unwrap();
    leaked.get_text("t").insert_unicode(0, "A").unwrap();

    md.bind(&review, 0); // refs 2: Private->Shared flip commits "A"
    assert_eq!(md.is_head_shared(&main), Some(true));

    let r_op = leaked.get_text("t").insert_unicode(0, "B");
    assert!(
        matches!(r_op, Err(LoroError::HeadShared)),
        "leaked local op after flip refused, got {r_op:?}"
    );
    // Refused earlier by the owned-head checkout gate (`OwnedHeadOp`), not `HeadShared`.
    let r_checkout = leaked.checkout(&Frontiers::default());
    assert!(
        matches!(r_checkout, Err(LoroError::OwnedHeadOp("checkout"))),
        "leaked checkout after flip refused, got {r_checkout:?}"
    );
    // `bind` does not update the Manual target; point `review` at the head's committed tip before
    // reading, or `resolve` treats the stale empty target as authoritative.
    md.policy().set_target(&review, md.head_tip(0).unwrap());
    assert_eq!(tlen(&md, &review), 1);
}

#[test]
fn sink_guard_nested_mutation_in_event_callback_refused() {
    let md = Arc::new(MultiHeadDoc::new(Manual::new()));
    let (main, review, writer) = (b("main"), b("review"), b("writer"));
    md.bind(&main, 0);
    md.bind(&review, 0);
    md.bind(&writer, 0);

    md.write(&writer, |d| d.get_text("t").insert_unicode(0, "1").unwrap())
        .unwrap();
    assert_eq!(md.is_head_shared(&main), Some(true), "head 0 still shared");
    let writer_doc = md.head_doc(md.bound_head(&writer).unwrap()).unwrap();

    // Callback on the writer's private head that mutates the SHARED head 0 from inside an event.
    let shared_head = md.head_doc(0).unwrap();
    let captured: Arc<Mutex<Option<LoroResult<()>>>> = Arc::new(Mutex::new(None));
    let cap2 = captured.clone();
    let sub = writer_doc.subscribe_root(Arc::new(move |_ev: crate::event::DiffEvent| {
        let r = shared_head.get_text("t").insert_unicode(0, "Z");
        *cap2.lock().unwrap() = Some(r);
    }));

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

// Copy-then-write correctness.

#[test]
fn copy_then_write_coowners_unaffected() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review, draft) = (b("main"), b("review"), b("draft"));
    md.bind(&main, 0);
    md.bind(&review, 0);
    md.bind(&draft, 0);
    assert_eq!(md.refs_of(&main), Some(3));
    assert_eq!(md.is_head_shared(&main), Some(true));

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

    md.write(&draft, |d| d.get_text("t").insert_unicode(1, "E").unwrap())
        .unwrap();
    assert_eq!(tlen(&md, &draft), 2);
    assert_eq!(tlen(&md, &main), 0);
    assert_eq!(tlen(&md, &review), 0);
}

// All-heads barrier.

#[test]
fn all_heads_barrier_runs_over_every_head() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    md.bind(&main, 0);
    md.bind(&review, 0);
    md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
    assert_eq!(md.head_count(), 2);
    let ran = md.with_all_heads_barrier(|| 42);
    assert_eq!(ran, 42);
    md.write(&main, |d| d.get_text("t").insert_unicode(1, "B").unwrap())
        .unwrap();
    assert_eq!(tlen(&md, &main), 2);
}

// Injected on-commit hook.

#[test]
fn injected_on_commit_hook_rekeys_and_after_commits() {
    // Commit directly on a head's handle, bypassing `MultiHeadDoc::write`: by_tip re-keying and
    // Manual's after_commit can then only come from the injected on-commit hook.
    let md = MultiHeadDoc::new(Manual::new());
    let main = b("main");
    md.bind(&main, 0);
    assert!(md.by_tip_maps(&Frontiers::default(), 0));

    let doc = md.head_doc(0).unwrap();
    doc.get_text("t").insert_unicode(0, "A").unwrap();
    doc.commit_then_renew();

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

// History-only import: no head materialization; co-owners unaffected.

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
    md.bind(&review, 0);
    assert_eq!(md.is_head_shared(&main), Some(true));

    let updates = external_updates("ABC");
    md.import(&updates).unwrap();

    // An import that materialized into head 0 would corrupt both main and review.
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

    let (updates2, f2) = external_updates_frontier("XYZ");
    md.import(&updates2).unwrap();
    let reader = b("reader");
    md.policy().set_target(&reader, f2);
    let seen = md.read(&reader, |d| d.get_text("t").to_string()).unwrap();
    assert_eq!(
        seen, "XYZ",
        "imported ops visible to a head advanced to them"
    );
    assert_eq!(tlen(&md, &main), 0);
    assert_eq!(tlen(&md, &review), 0);
}

#[test]
fn import_path_sink_guard_backstops_shared_head() {
    // Backstop: a direct checkout -> apply_diff onto a shared head is refused.
    let md = MultiHeadDoc::new(Manual::new());
    let (main, review) = (b("main"), b("review"));
    md.bind(&main, 0);
    md.bind(&review, 0);

    let (updates, f) = external_updates_frontier("ABC");
    md.import(&updates).unwrap();
    assert_eq!(
        tlen(&md, &main),
        0,
        "history-only import left the shared head empty"
    );

    // Refused by the owned-head checkout gate.
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

// Aggregated subscriptions and snapshot export.

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
    md.bind(&review, 0);
    md.write(&main, |d| d.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
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

// refs == 0 retirement: non-root heads are dropped, the root is pinned.

#[test]
fn dead_tip_rematerializes_via_replay_after_head_dropped() {
    let md = MultiHeadDoc::new(Manual::new());
    let (main, feature) = (b("main"), b("feature"));
    md.bind(&main, 0);
    md.bind(&feature, 0);

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

    // Retire H via `unbind`: the fast-forward-only guard holds a backward resolve of a bound head,
    // so the head stays on H.
    md.unbind(&feature);
    assert!(
        md.head_doc(h).is_none(),
        "sole-owner head retired at refs 0"
    );
    assert!(
        md.head_doc(md.root_head_id()).is_some(),
        "root head is never retired"
    );

    // Re-access of H's dead tip re-materializes by replay from the pinned root.
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
    md.bind(&x, 0);

    md.write(&x, |d| d.get_text("t").insert_unicode(0, "X").unwrap())
        .unwrap();
    let hx = md.bound_head(&x).unwrap();
    assert_ne!(hx, 0);

    md.policy().set_target(&main, md.head_tip(hx).unwrap());
    md.resolve(&main, Intent::Read).unwrap();
    assert_ne!(md.bound_head(&main), Some(0), "main left the root head");

    // The root reached refs 0 but is pinned (the materialize/import/export anchor).
    assert!(
        md.head_doc(0).is_some(),
        "root head is pinned and survives refs == 0"
    );
    // A branch targeting the root's empty tip shares it rather than hitting a missing head.
    let cold = b("cold");
    md.policy().set_target(&cold, Frontiers::default());
    let seen = md.read(&cold, |d| d.get_text("t").to_string()).unwrap();
    assert_eq!(seen, "", "pinned root still resolvable at its empty tip");
}

// IndexDoc = MultiHeadDoc<SelfRooted>.

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
    assert_ne!(idx.bound_head(&b("feat")), idx.bound_head(&b("main")));
    assert_eq!(idx.refs_of(&b("feat")), Some(1));
    assert_eq!(idx.is_head_shared(&b("feat")), Some(false));
    assert!(idx.branches().contains(&b("main")));
    assert!(idx.branches().contains(&b("feat")));
}

#[test]
fn index_remote_lineage_import_rebinds_branch() {
    let a = IndexDoc::new(SelfRooted::new());
    a.init_genesis(&b("main")).unwrap();
    a.create_index_branch(&b("feat"), &b("main")).unwrap();
    a.record_head(&b("feat"), &d("D"), 111, 7).unwrap();
    let updates = a.export(ExportMode::all_updates()).unwrap();

    let bb = IndexDoc::new(SelfRooted::new());
    bb.init_genesis(&b("main")).unwrap();
    assert!(
        !bb.branches().contains(&b("feat")),
        "feat unknown before import"
    );

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
fn index_target_is_join_of_same_name_branch_markers() {
    let a = IndexDoc::new(SelfRooted::new());
    a.init_genesis(&b("main")).unwrap();
    a.create_index_branch(&b("shared"), &b("main")).unwrap();
    a.record_head(&b("shared"), &d("A"), 1, 1).unwrap();
    let a_updates = a.export(ExportMode::all_updates()).unwrap();

    let bb = IndexDoc::new(SelfRooted::new());
    bb.init_genesis(&b("main")).unwrap();
    bb.create_index_branch(&b("shared"), &b("main")).unwrap();
    bb.record_head(&b("shared"), &d("B"), 2, 2).unwrap();
    assert!(doc_has_key(&bb, &b("shared"), "B"));
    assert!(!doc_has_key(&bb, &b("shared"), "A"), "B has not seen A yet");

    // Import A: `shared` has two creation markers on B and its frontier is their join, so it shows
    // both records.
    bb.import(&a_updates).unwrap();
    assert!(
        doc_has_key(&bb, &b("shared"), "A"),
        "join of both markers brought session A's record"
    );
    assert!(
        doc_has_key(&bb, &b("shared"), "B"),
        "join of both markers kept session B's record"
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
    // Same doc "G" recorded under different head-peer keys on one branch. Mergeable map-key
    // children make docs["G"].heads one container on both sides, so both records survive (divergent
    // op-id children would LWW-drop one).
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
    // Import-THEN-record ordering (as real sync produces): the catch-up resolve arm leaves the head
    // detached, and without the detached-clear the local record fails `AutoCommitNotStarted`.
    let a = IndexDoc::new(SelfRooted::new());
    a.init_genesis(&b("main")).unwrap();
    a.create_index_branch(&b("shared"), &b("main")).unwrap();
    a.record_head(&b("shared"), &d("G"), 111, 1).unwrap();
    let a_updates = a.export(ExportMode::all_updates()).unwrap();

    let bb = IndexDoc::new(SelfRooted::new());
    bb.init_genesis(&b("main")).unwrap();
    bb.create_index_branch(&b("shared"), &b("main")).unwrap();

    bb.import(&a_updates).unwrap();
    assert!(
        doc_has_key(&bb, &b("shared"), "G"),
        "import advanced the head to A's record"
    );

    bb.record_head(&b("shared"), &d("G"), 222, 2).unwrap();
    assert_eq!(
        head_count(&bb, &b("shared"), "G"),
        2,
        "import-then-record kept BOTH head records (no dropped record)"
    );
}

// BranchingDocHead surface, read-exit commit, union-frontier closures.

#[test]
fn branching_doc_head_surface_read_write() {
    // History/attachment methods are absent from `BranchingDocHead` at the type level.
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    let br = md.branch("main");
    br.write(|head: &BranchingDocHead| {
        head.get_text("t").insert_unicode(0, "hi").unwrap();
    })
    .unwrap();
    let (text, deep_is_map, frontier_nonempty) = br
        .read(|head: &BranchingDocHead| {
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
    br.read(|head| head.get_text("t").insert_unicode(0, "X").unwrap())
        .unwrap();
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
    // `head.get_text("t").doc()` exposes the raw owned `LoroDoc`; a raw `checkout` would move head
    // state without re-keying the registry (desyncing tip/by_tip). The gate refuses it.
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    md.write(&b("main"), |d| {
        d.get_text("t").insert_unicode(0, "M").unwrap()
    })
    .unwrap();

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
    // `detach()` is a no-op on a registry-owned head: attachment is the registry's to change.
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
    md.bind(&b("feat"), 0);
    md.write(&b("feat"), |d| {
        d.get_text("t").insert_unicode(0, "F").unwrap()
    })
    .unwrap();
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
    // main's head is behind the union; see `Branch::fork`.
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    md.bind(&b("feat"), 0);
    md.write(&b("feat"), |d| {
        d.get_text("t").insert_unicode(0, "F").unwrap()
    })
    .unwrap();
    md.write(&b("main"), |d| {
        d.get_text("t").insert_unicode(0, "M").unwrap()
    })
    .unwrap();

    let forked = md.branch("main").fork().unwrap();
    assert_eq!(forked.get_text("t").to_string(), "M");
    let f = forked.state_frontiers();
    forked.checkout(&Frontiers::default()).unwrap();
    assert_eq!(forked.get_text("t").to_string(), "");
    forked.checkout(&f).unwrap();
    assert_eq!(forked.get_text("t").to_string(), "M");
    assert_eq!(
        md.branch("main")
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "M"
    );
}

// refs>1 copy+advance detached-clear; Delegated engine (create / write / read / merge / advance).

#[test]
fn refs_gt1_copy_advance_clears_detached() {
    // Copy+advance arm: the detached-clear lets a subsequent write succeed.
    let md = MultiHeadDoc::new(Manual::new());
    // x diverges onto its own head building "AB"; main + feat stay on the root (refs 2).
    md.bind(&b("main"), 0);
    md.bind(&b("feat"), 0);
    md.bind(&b("x"), 0);
    md.write(&b("x"), |d| d.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
    let t_a = md.head_tip(md.bound_head(&b("x")).unwrap()).unwrap();
    md.write(&b("x"), |d| d.get_text("t").insert_unicode(1, "B").unwrap())
        .unwrap();
    assert_eq!(md.refs_of(&b("main")), Some(2));

    // main's target -> the intermediate frontier: resolve(Write) hits that arm.
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

    repo.create_branch(&b("draft"), &b(GENESIS_BRANCH)).unwrap();
    assert!(repo.branches().contains(&b("draft")));
    assert_eq!(
        doc.branch("draft")
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "main",
        "draft sees main's content (shared head, no copy)"
    );

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

    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(4, "M").unwrap())
        .unwrap();
    doc.branch("feature")
        .write(|h| h.get_text("t").insert_unicode(4, "F").unwrap())
        .unwrap();
    assert!(!doc.contains(&b(GENESIS_BRANCH), &b("feature")).unwrap());

    let outcome = doc.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap();
    assert_eq!(outcome, MergeOutcome::Merged);

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
    assert_eq!(
        doc.merge(&b(GENESIS_BRANCH), &b("feature")).unwrap(),
        MergeOutcome::AlreadyContained
    );
}

// Two-peer Delegated content-sync convergence.

#[test]
fn two_peer_content_sync_converges() {
    // B's `after_import` (Delegated) must re-resolve main so its head converges to A's actual
    // content, not merely a matching index head_count.
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

    bb.index().import(&a_index).unwrap();
    gb.import(&a_content).unwrap();

    // Inspect main's bound head directly: a lazy read would re-resolve on its own and mask the
    // eager ingest.
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
    // Bind and diverge draft onto its own head (refs 1): the case a broken delete would leave
    // dangling.
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
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
    let at_a = doc.frontier_of(&b(GENESIS_BRANCH)).unwrap();
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(1, "B").unwrap())
        .unwrap();

    doc.create_branch_at(&b("hist"), &at_a).unwrap();
    assert!(repo.branches().contains(&b("hist")));
    assert_eq!(
        doc.branch("hist")
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "A"
    );
    assert!(repo.index().head_of(&b("hist")).is_ok());
    doc.branch("hist")
        .write(|h| h.get_text("t").insert_unicode(1, "X").unwrap())
        .unwrap();
    assert_eq!(
        doc.branch("hist")
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "AX"
    );
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
    md.bind(&b("feat"), 0);

    let count = Arc::new(AtomicUsize::new(0));
    let c2 = count.clone();
    let sub = md
        .branch("feat")
        .subscribe_root(Arc::new(move |_ev: crate::event::DiffEvent| {
            c2.fetch_add(1, SeqCst);
        }))
        .unwrap();

    // feat's write copies off the shared root; the subscription must re-install on the new head.
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
    let a = BranchingDocRepo::open().unwrap();
    let ga = a.open_doc("G".into());
    ga.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "A1").unwrap())
        .unwrap();
    let at_a = ga.frontier_of(&b(GENESIS_BRANCH)).unwrap();
    let a_index = a.index().export(ExportMode::all_updates()).unwrap();
    let a_content = ga.export(ExportMode::all_updates()).unwrap();

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
fn index_create_existing_branch_errors_and_no_peer_serves_two_branches() {
    // Single session, many branches, plus content writes: re-forking an existing name must error
    // rather than mint a second creation marker, and no index peer attributes to two branches.
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
    assert_eq!(attr.tips.len(), 4, "main, b1, b2, b1a");
}

#[test]
fn index_crafted_shared_peer_across_two_markers_attributes_causally() {
    // Hand-crafted history: one peer writes alpha's creation marker, then beta's, then an unrelated
    // op. Attribution keys on the nearest marker in each op's causal past, not peer identity:
    // tips[alpha] = [create(alpha)], tips[beta] = [unrelated].
    const SHARED_PEER: PeerID = 999;
    let ext = LoroDoc::new();
    ext.start_auto_commit();
    ext.set_peer_id(SHARED_PEER).unwrap();
    ext.get_map("branch").insert("name", "alpha").unwrap();
    ext.commit_then_renew();
    let create_alpha = ext.state_frontiers();
    ext.get_map("branch").insert("name", "beta").unwrap();
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
}

#[test]
fn spike_scenario2_diamond_topology_forks_merges_are_stable() {
    // A forks B and C; merge both into A; fork D from A: D inherits both, B/C unchanged.
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

    repo.create_branch(&b("D"), &b(GENESIS_BRANCH)).unwrap();
    let d_text = g
        .branch("D")
        .read(|h| h.get_text("t").to_string())
        .unwrap();
    assert_eq!(
        d_text, a_text,
        "D inherits exactly A's merged state, both contributions present once"
    );

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
    assert_ne!(a_after_base, g.frontier_of(&b(GENESIS_BRANCH)).unwrap());
}

#[test]
fn spike_scenario3_repeated_merge_into_same_target_no_drop_no_dup() {
    // Fork b, merge into main, keep writing on b, merge again: main reflects both merges.
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

    g.branch("feature")
        .write(|h| h.get_text("t").insert_unicode(6, "-2").unwrap())
        .unwrap();
    // main advances concurrently, so the second merge is a real join, not a fast-forward.
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
    doc.create_branch_at(&b("hist"), &f).unwrap();
    assert!(
        doc.create_branch_at(&b("hist"), &f).is_err(),
        "re-create of an existing branch errors"
    );
}

#[test]
fn index_snapshot_restore_preserves_tips_for_every_branch() {
    // Derived `tips` must survive a Snapshot export/import into a fresh `IndexDoc`, over a
    // nested fork, a divergent merge, and a branch created by two sessions.
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());

    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();

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

    // Divergent merge (main advances independently of feat2), forcing a real join.
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(4, "-m").unwrap())
        .unwrap();
    let outcome = g.merge(&b(GENESIS_BRANCH), &b("feat2")).unwrap();
    assert_eq!(
        outcome,
        MergeOutcome::Merged,
        "the merge must be a real divergent join, not a fast-forward"
    );

    // A branch created by two independent sessions (two creation markers), synced via index-history
    // import, so `shared`'s frontier is the join of both.
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
        "sanity: 'shared' must be a genuine two-marker (two-id) frontier before the snapshot"
    );

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

    let snap = idx.export(ExportMode::Snapshot).unwrap();
    let restored = IndexDoc::new(SelfRooted::new());
    let status = restored.import(&snap).unwrap();
    assert!(
        !status.success.is_empty(),
        "snapshot import's ImportStatus::success must be non-empty for after_import \
         to rebuild tips at all"
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

// Control: `index_import_then_record_converges`.

#[test]
fn index_remote_discovered_branch_local_record_survives() {
    let a = IndexDoc::new(SelfRooted::new());
    a.init_genesis(&b("main")).unwrap();
    a.create_index_branch(&b("feat"), &b("main")).unwrap();
    a.record_head(&b("feat"), &d("D"), 111, 7).unwrap();
    let a_updates = a.export(ExportMode::all_updates()).unwrap();

    // B learns `feat` from A's import, so its head is minted by the resolve MATERIALIZE arm, which
    // no `branch.name` marker names.
    let bb = IndexDoc::new(SelfRooted::new());
    bb.init_genesis(&b("main")).unwrap();
    bb.import(&a_updates).unwrap();
    let after_import = bb.recorded_ids(&b("feat"), &d("D")).unwrap();
    assert_eq!(
        after_import,
        vec![ID::new(111, 7)],
        "B discovered feat and materialized A's record"
    );

    bb.record_head(&b("feat"), &d("D"), 222, 9).unwrap();
    let mut after_record = bb.recorded_ids(&b("feat"), &d("D")).unwrap();
    after_record.sort();
    assert_eq!(
        after_record,
        vec![ID::new(111, 7), ID::new(222, 9)],
        "B's local record on a remotely-discovered branch is LOST on B: {after_record:?}"
    );

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

/// Build the diamond + repeated-merge + two-session history on a fresh repo and return its index.
fn s4_history() -> BranchingDocRepo {
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();

    // Diamond: B, C fork from main and merge back; D forks from post-merge main and writes.
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

    // feature merged into main twice (fast-forward, then real join), then a redundant third merge.
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

    // Session 2 learns `shared` from the repo, records on it (materialize-arm peer), the repo takes
    // that record back, then records again.
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
    let src_tips = src.tips();
    assert_eq!(src_tips.len(), 6, "main, B, C, D, feature, shared");

    let bytes = src.export(ExportMode::all_updates()).unwrap();
    let cold = IndexDoc::new(SelfRooted::new());
    // Clean cold import: `unwrap` asserts the fold rejected nothing.
    cold.import(&bytes).unwrap();

    assert_eq!(cold.tips(), src_tips, "tips equal on the cold import");
    let mut sb = src.branches();
    sb.sort();
    let mut cb = cold.branches();
    cb.sort();
    assert_eq!(sb, cb);
    let mut ids = cold.recorded_ids(&b("shared"), &d("OTHER")).unwrap();
    ids.sort();
    assert_eq!(ids, vec![ID::new(500, 3), ID::new(600, 2)]);
}

#[test]
fn index_change_spanning_two_branches_is_rejected_without_moving_tips() {
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    idx.create_index_branch(&b("alpha"), &b("main")).unwrap();
    idx.create_index_branch(&b("beta"), &b("main")).unwrap();
    idx.record_head(&b("alpha"), &d("A"), 1, 1).unwrap();
    idx.record_head(&b("beta"), &d("B"), 2, 2).unwrap();
    let tips_before = idx.tips();

    // A plain LoroDoc holding the whole index history sits at the join of every tip; a local write
    // there depends on alpha's and beta's tips.
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

    let result = idx.import(&ext_bytes);
    assert!(
        result.is_err(),
        "an op spanning two branches must reject the import, not attribute silently"
    );

    assert_eq!(idx.tips(), tips_before, "no branch's tips moved");
    assert!(
        idx.oplog_vv().get_last(poison.peer) == Some(poison.counter),
        "the rejected op stays in the op log"
    );
    assert_eq!(
        idx.recorded_ids(&b("alpha"), &d("A")).unwrap(),
        vec![ID::new(1, 1)]
    );
    assert!(idx.recorded_ids(&b("alpha"), &d("B")).unwrap().is_empty());
}

// `branch.name` marker: two root containers total, no branch-name charset constraint.

#[test]
fn index_arbitrary_branch_names_round_trip() {
    // The name is a map value, not a container id, so spaces/unicode/':'/'/' round-trip through
    // creation, snapshot, cold import and reads.
    let names = ["a branch with spaces", "café ☕ 名前", "boc/abc123", "a:b:c"];
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    for (i, n) in names.iter().enumerate() {
        idx.create_index_branch(&b(n), &b("main")).unwrap();
        idx.record_head(&b(n), &d("G"), 1000 + i as u64, i as i32 + 1)
            .unwrap();
    }
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

// Repo-wide `IndexDoc::merge` is a single marker op.

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

    // Doc D recorded concurrently on both branches (distinct peer keys), doc E only on `from`.
    idx.record_head(&b("into"), &d("D"), 111, 5).unwrap();
    idx.record_head(&b("from"), &d("D"), 222, 7).unwrap();
    idx.record_head(&b("from"), &d("E"), 333, 3).unwrap();

    let ops_before = total_ops(&idx);
    let outcome = idx.merge(&b("into"), &b("from")).unwrap();
    assert_eq!(outcome, MergeOutcome::Merged, "into and from diverged");

    assert_eq!(
        total_ops(&idx) - ops_before,
        1,
        "merge writes a single index marker op, not one per merged doc"
    );

    let mut d_ids = idx.recorded_ids(&b("into"), &d("D")).unwrap();
    d_ids.sort();
    assert_eq!(
        d_ids,
        vec![ID::new(111, 5), ID::new(222, 7)],
        "both concurrent same-doc writers survive the merge (per-key VV union)"
    );
    assert_eq!(
        idx.recorded_ids(&b("into"), &d("E")).unwrap(),
        vec![ID::new(333, 3)]
    );
    assert_eq!(
        idx.recorded_ids(&b("from"), &d("D")).unwrap(),
        vec![ID::new(222, 7)]
    );
    assert_eq!(
        idx.recorded_ids(&b("from"), &d("E")).unwrap(),
        vec![ID::new(333, 3)]
    );

    assert!(
        idx.tips()[&b("into")].as_single().is_some(),
        "into's tip is the single merge marker"
    );
    let mut branches = idx.branches();
    branches.sort();
    assert_eq!(branches, vec![b("from"), b("into"), b("main")]);

    assert_eq!(
        idx.arena
            .top_level_root_containers(crate::arena::LoadAllFlag)
            .len(),
        2,
        "a merge marker adds no root container"
    );

    // The marker folds cleanly on a cold import (attributed to `into`, no dep agreement needed).
    let bytes = idx.export(ExportMode::all_updates()).unwrap();
    let cold = IndexDoc::new(SelfRooted::new());
    cold.import(&bytes).unwrap();
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

    g.branch("feat")
        .write(|h| h.get_text("t").insert_unicode(4, "-f").unwrap())
        .unwrap();
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(4, "-m").unwrap())
        .unwrap();

    let idx_ops_before = total_ops(repo.index());
    let outcome = repo.merge_branch(&b(GENESIS_BRANCH), &b("feat")).unwrap();
    assert_eq!(outcome, MergeOutcome::Merged);
    assert_eq!(
        total_ops(repo.index()) - idx_ops_before,
        1,
        "repo-wide merge is a single index marker, not one record per doc"
    );

    let text = g
        .branch(GENESIS_BRANCH)
        .read(|h| h.get_text("t").to_string())
        .unwrap();
    assert!(
        text.contains("base") && text.contains("-f") && text.contains("-m"),
        "merged main content contains base + both edits: {text:?}"
    );
    assert!(repo.branches().contains(&b("feat")));
}

#[test]
fn index_merge_into_lww_winner_still_emits_and_propagates() {
    // merge(feat, main): feat's branch.name already equals "feat" at the join, so re-setting it
    // would be a no-op; the marker must still emit so main's record reaches feat.
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

    assert_eq!(
        idx.recorded_ids(&b("feat"), &d("D")).unwrap(),
        vec![ID::new(111, 5)],
        "main's record must reach feat"
    );
    assert_eq!(
        idx.recorded_ids(&b("feat"), &d("E")).unwrap(),
        vec![ID::new(222, 7)]
    );

    // A remote peer sees only the op stream; without a marker op it would never learn of the merge.
    let cold = IndexDoc::new(SelfRooted::new());
    cold.import(&idx.export(ExportMode::all_updates()).unwrap())
        .unwrap();
    assert_eq!(cold.tips(), idx.tips(), "merge propagates: cold tips equal");
    assert_eq!(
        cold.recorded_ids(&b("feat"), &d("D")).unwrap(),
        vec![ID::new(111, 5)],
        "the merge reaches a cold-importing peer"
    );
}

#[test]
fn index_merge_preserves_docs_in_both_sibling_creation_orders() {
    // Two siblings off main: their branch.name sets are concurrent, so the LWW tie-break by
    // (lamport, peer) decides the join value (later-created wins). Both orders must preserve the
    // merge.
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
    // Two merges into the same target with `from` advancing between: the second writes the same
    // branch NAME, so a name-keyed marker would no-op. The seq'd marker must emit on both.
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    idx.create_index_branch(&b("into"), &b("main")).unwrap();
    idx.create_index_branch(&b("from"), &b("main")).unwrap();

    idx.record_head(&b("from"), &d("D"), 111, 1).unwrap();
    idx.record_head(&b("into"), &d("Z"), 999, 1).unwrap(); // into diverges so 1st merge is real
    assert_eq!(idx.merge(&b("into"), &b("from")).unwrap(), MergeOutcome::Merged);
    assert_eq!(idx.recorded_ids(&b("into"), &d("D")).unwrap(), vec![ID::new(111, 1)]);

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

// Derived reads: fork_point, docs_modified_on_branch, docs_changed_since_main.

#[test]
fn index_fork_point_and_docs_modified() {
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    // Genesis has no fork point (its creation marker is the root op).
    assert_eq!(idx.fork_point(&b("main")), Frontiers::default());

    idx.record_head(&b("main"), &d("D"), 111, 5).unwrap();
    let main_after_d = idx.tips()[&b("main")].clone();

    idx.create_index_branch(&b("feat"), &b("main")).unwrap();
    assert_eq!(idx.fork_point(&b("feat")), main_after_d);

    idx.record_head(&b("feat"), &d("E"), 222, 7).unwrap();
    // docs_modified_on_branch excludes docs inherited unchanged from main; only E is feat's.
    assert_eq!(idx.docs_modified_on_branch(&b("feat")).unwrap(), vec![d("E")]);
    assert_eq!(idx.docs_modified_on_branch(&b("main")).unwrap(), vec![d("D")]);
    assert!(idx.docs_modified_on_branch(&b("nope")).unwrap().is_empty());
    assert_eq!(idx.fork_point(&b("nope")), Frontiers::default());
}

#[test]
fn index_docs_modified_stack_a_b_c() {
    // main -> b -> c. docs_modified_on_branch is the top-of-stack set; docs_changed_since_main is
    // the whole-stack union.
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    idx.record_head(&b("main"), &d("M"), 1, 1).unwrap();

    idx.create_index_branch(&b("bb"), &b("main")).unwrap();
    idx.record_head(&b("bb"), &d("B1"), 10, 1).unwrap();

    idx.create_index_branch(&b("cc"), &b("bb")).unwrap();
    idx.record_head(&b("cc"), &d("C1"), 20, 1).unwrap();

    assert_eq!(idx.docs_modified_on_branch(&b("main")).unwrap(), vec![d("M")]);
    assert_eq!(idx.docs_modified_on_branch(&b("bb")).unwrap(), vec![d("B1")]);
    assert_eq!(idx.docs_modified_on_branch(&b("cc")).unwrap(), vec![d("C1")]);

    // c changed B1 and C1 since leaving main; main's own M is inherited, not "changed since main".
    assert_eq!(
        idx.docs_changed_since_main(&b("cc")).unwrap(),
        vec![d("B1"), d("C1")]
    );
    assert_eq!(idx.docs_changed_since_main(&b("bb")).unwrap(), vec![d("B1")]);
    assert!(idx.docs_changed_since_main(&b("main")).unwrap().is_empty());
}

#[test]
fn index_docs_modified_recomputes_across_a_merge() {
    let idx = IndexDoc::new(SelfRooted::new());
    idx.init_genesis(&b("main")).unwrap();
    idx.create_index_branch(&b("into"), &b("main")).unwrap();
    idx.create_index_branch(&b("from"), &b("main")).unwrap();
    idx.record_head(&b("into"), &d("I"), 10, 1).unwrap();
    idx.record_head(&b("from"), &d("F"), 20, 1).unwrap();

    assert_eq!(idx.docs_modified_on_branch(&b("into")).unwrap(), vec![d("I")]);
    idx.merge(&b("into"), &b("from")).unwrap();
    assert_eq!(
        idx.docs_modified_on_branch(&b("into")).unwrap(),
        vec![d("F"), d("I")]
    );
}

#[test]
fn index_docs_modified_excludes_concurrent_parent_op() {
    // After `child` forks from `parent`, `parent` records a doc concurrently (not an ancestor of
    // the fork point). A naive copy-parent's-live-set-at-fork could leak it into child depending on
    // lamport fold order; the on-demand fork-point diff cannot.
    let a = IndexDoc::new(SelfRooted::new());
    a.init_genesis(&b("main")).unwrap();
    a.create_index_branch(&b("parent"), &b("main")).unwrap();
    a.record_head(&b("parent"), &d("P0"), 10, 1).unwrap();
    let a_updates = a.export(ExportMode::all_updates()).unwrap();

    let bb = IndexDoc::new(SelfRooted::new());
    bb.init_genesis(&b("main")).unwrap();
    bb.import(&a_updates).unwrap();
    bb.create_index_branch(&b("child"), &b("parent")).unwrap();
    bb.record_head(&b("child"), &d("CH"), 30, 1).unwrap();

    // Session A records another doc on `parent` concurrently with the fork, then the two sync.
    a.record_head(&b("parent"), &d("PCONC"), 11, 1).unwrap();
    bb.import(&a.export(ExportMode::all_updates()).unwrap())
        .unwrap();

    assert_eq!(
        bb.docs_modified_on_branch(&b("child")).unwrap(),
        vec![d("CH")],
        "child must not absorb the parent's concurrent post-fork doc"
    );
    assert_eq!(
        bb.docs_modified_on_branch(&b("parent")).unwrap(),
        vec![d("P0"), d("PCONC")]
    );
}

#[test]
fn index_local_commit_after_import_preserves_both_lineages() {
    // `after_commit` replaces tips[b] with the single committed op while `after_import` joins. Safe
    // because a local write resolves b to its current tips first, so the new op causally dominates
    // the join. Interleave an import (JOIN to two ids) with a local commit (REPLACE to one) and
    // show no record lost.
    let a = IndexDoc::new(SelfRooted::new());
    a.init_genesis(&b("main")).unwrap();
    a.create_index_branch(&b("shared"), &b("main")).unwrap();
    a.record_head(&b("shared"), &d("D"), 111, 1).unwrap();
    let a_updates = a.export(ExportMode::all_updates()).unwrap();

    let bb = IndexDoc::new(SelfRooted::new());
    bb.init_genesis(&b("main")).unwrap();
    bb.create_index_branch(&b("shared"), &b("main")).unwrap();
    bb.record_head(&b("shared"), &d("D"), 222, 2).unwrap();

    bb.import(&a_updates).unwrap();
    assert_eq!(bb.tips()[&b("shared")].len(), 2, "join of two lineages");
    let mut both = bb.recorded_ids(&b("shared"), &d("D")).unwrap();
    both.sort();
    assert_eq!(both, vec![ID::new(111, 1), ID::new(222, 2)]);

    bb.record_head(&b("shared"), &d("E"), 222, 3).unwrap();
    assert_eq!(
        bb.tips()[&b("shared")].len(),
        1,
        "after_commit replaced the joined tips with one id"
    );
    let mut still = bb.recorded_ids(&b("shared"), &d("D")).unwrap();
    still.sort();
    assert_eq!(
        still,
        vec![ID::new(111, 1), ID::new(222, 2)],
        "both lineages' D records survive the tips REPLACE"
    );
    assert_eq!(
        bb.recorded_ids(&b("shared"), &d("E")).unwrap(),
        vec![ID::new(222, 3)]
    );

    // Converges on a cold peer: the single-tip op carries the full past.
    let cold = IndexDoc::new(SelfRooted::new());
    cold.import(&bb.export(ExportMode::all_updates()).unwrap())
        .unwrap();
    let mut cd = cold.recorded_ids(&b("shared"), &d("D")).unwrap();
    cd.sort();
    assert_eq!(cd, vec![ID::new(111, 1), ID::new(222, 2)]);
}

#[test]
fn resolve_with_advance_tags_import_advance() {
    // The merge/advance cause maps to `("advance", Import)`: a subscriber sees a merge-driven move
    // as live `Import`, not `Checkout`. The parity harness cannot reach this cause.
    use crate::event::{DiffEvent, EventTriggerKind};
    let md = MultiHeadDoc::new(Manual::new());
    md.bind(&b("main"), 0);
    md.create_branch(&b("draft"), &b("main")).unwrap();
    md.write(&b("draft"), |d| d.get_text("t").insert_unicode(0, "X").unwrap())
        .unwrap();
    let td = md.head_tip(md.bound_head(&b("draft")).unwrap()).unwrap();

    let seen: Arc<Mutex<Vec<(EventTriggerKind, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let s2 = seen.clone();
    let _sub = md
        .subscribe_branch(
            &b("main"),
            None,
            Arc::new(move |ev: DiffEvent| {
                s2.lock()
                    .unwrap()
                    .push((ev.event_meta.by, ev.event_meta.origin.to_string()));
            }),
        )
        .unwrap();

    md.policy().set_target(&b("main"), td);
    md.resolve_with(&b("main"), Intent::Read, ResolveCause::Advance)
        .unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "one advance event delivered");
    assert_eq!(seen[0].0, EventTriggerKind::Import, "advance tagged Import");
    assert_eq!(seen[0].1, "advance", "advance origin");
}

// Branch-diff producer surface.

use crate::event::Diff as EvDiff;
use crate::handler::TextDelta;
use crate::undo::DiffBatch;

fn doc_id(s: &str) -> DocId {
    s.into()
}

fn text_deltas(batch: &DiffBatch) -> Vec<TextDelta> {
    batch
        .iter()
        .flat_map(|(_, diff)| match diff {
            EvDiff::Text(t) => TextDelta::from_text_diff(t.iter()),
            _ => Vec::new(),
        })
        .collect()
}

fn inserted_text(batch: &DiffBatch) -> String {
    text_deltas(batch)
        .into_iter()
        .filter_map(|td| match td {
            TextDelta::Insert { insert, .. } => Some(insert),
            _ => None,
        })
        .collect()
}

fn delete_count(batch: &DiffBatch) -> usize {
    text_deltas(batch)
        .into_iter()
        .map(|td| match td {
            TextDelta::Delete { delete } => delete,
            _ => 0,
        })
        .sum()
}

#[test]
fn branch_sources_creation_marker_kind_and_parent() {
    let repo = BranchingDocRepo::open().unwrap();
    repo.create_branch(&b("feat"), &b(GENESIS_BRANCH)).unwrap();

    let sources = repo.index().branch_sources().unwrap();
    let feat = sources.get(&b("feat")).expect("feat present");
    assert_eq!(feat.kind, BranchSourceKind::Create);
    assert_eq!(feat.parent, Some(b(GENESIS_BRANCH)), "feat forked from main");
    assert_eq!(
        feat.op.counter, 0,
        "creation marker is the first op of feat's fresh index peer"
    );

    let main = sources.get(&b(GENESIS_BRANCH)).expect("genesis present");
    assert_eq!(main.kind, BranchSourceKind::Create);
    assert_eq!(main.parent, None, "genesis has no parent (empty deps)");
}

#[test]
fn branch_sources_agree_with_fork_point() {
    // Pins the shared min-lamport-run-start helper: get_deps_of(op) == fork_point.
    let repo = BranchingDocRepo::open().unwrap();
    repo.create_branch(&b("feat"), &b(GENESIS_BRANCH)).unwrap();
    repo.create_branch(&b("feat2"), &b("feat")).unwrap();

    let sources = repo.index().branch_sources().unwrap();
    for (name, src) in &sources {
        let deps = repo
            .index()
            .oplog
            .lock()
            .get_deps_of(src.op)
            .unwrap_or_default();
        assert_eq!(
            deps,
            repo.index().fork_point(name),
            "get_deps_of(branch_sources()[{name}].op) == fork_point({name})"
        );
    }
}

#[test]
fn branch_sources_stable_across_cross_merge() {
    // A branch created off main, then main cross-merged into it, must still report parent == main
    // and the original creation op-id; a tips/frontier-based impl would drift.
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
    repo.create_branch(&b("feat"), &b(GENESIS_BRANCH)).unwrap();

    let op_before = repo.index().branch_sources().unwrap()[&b("feat")].op;

    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(1, "B").unwrap())
        .unwrap();
    repo.merge_branch(&b("feat"), &b(GENESIS_BRANCH)).unwrap();

    let after = repo.index().branch_sources().unwrap();
    assert_eq!(
        after[&b("feat")].op, op_before,
        "creation-marker op-id is stable across a cross-merge (min-lamport causal root)"
    );
    assert_eq!(
        after[&b("feat")].parent,
        Some(b(GENESIS_BRANCH)),
        "parent stays main after cross-merge"
    );
}

#[test]
fn diff_docs_symmetric_over_key_union() {
    let repo = BranchingDocRepo::open().unwrap();
    let d1 = repo.open_doc("D1".into());
    let d2 = repo.open_doc("D2".into());
    let d3 = repo.open_doc("D3".into());
    repo.create_branch(&b("feat"), &b(GENESIS_BRANCH)).unwrap();

    d1.branch("feat")
        .write(|h| h.get_text("t").insert_unicode(0, "x").unwrap())
        .unwrap();
    d2.branch("feat")
        .write(|h| h.get_text("t").insert_unicode(0, "y").unwrap())
        .unwrap();
    d3.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "z").unwrap())
        .unwrap();

    assert_eq!(
        repo.index().diff_docs(&b(GENESIS_BRANCH), &b("feat")).unwrap(),
        vec![doc_id("D1"), doc_id("D2"), doc_id("D3")],
        "diff_docs captures both feat's D1/D2 and main-only D3"
    );
    assert_eq!(
        repo.index().docs_modified_on_branch(&b("feat")).unwrap(),
        vec![doc_id("D1"), doc_id("D2")],
        "docs_modified_on_branch is one-sided (drops D3)"
    );
    assert!(repo
        .index()
        .diff_docs(&b("feat"), &b("feat"))
        .unwrap()
        .is_empty());
}

#[test]
fn branching_doc_head_diff_passthrough_state_preserving() {
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "AB").unwrap())
        .unwrap();
    let f0 = g.frontier_of(&b(GENESIS_BRANCH)).unwrap();
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(2, "CD").unwrap())
        .unwrap();
    let f1 = g.frontier_of(&b(GENESIS_BRANCH)).unwrap();

    let (before, batch) = g
        .branch(GENESIS_BRANCH)
        .read(|h| (h.state_frontiers(), h.diff(&f0, &f1).unwrap()))
        .unwrap();
    let after = g.branch(GENESIS_BRANCH).read(|h| h.state_frontiers()).unwrap();

    assert_eq!(before, after, "diff self-restores the head's frontier");
    assert_eq!(inserted_text(&batch), "CD", "f0 -> f1 inserts CD");
}

#[test]
fn diff_from_fork_dynamic_gca_baseline() {
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();
    repo.create_branch(&b("feat"), &b(GENESIS_BRANCH)).unwrap();
    g.branch("feat")
        .write(|h| h.get_text("t").insert_unicode(4, "-feat").unwrap())
        .unwrap();
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(4, "-main").unwrap())
        .unwrap();

    let fork = g.diff_from_fork(&b("feat"), &b(GENESIS_BRANCH)).unwrap();
    assert_eq!(
        inserted_text(&fork.diff),
        "-feat",
        "delta is feat's post-fork insert against the GCA base only"
    );
    assert_eq!(delete_count(&fork.diff), 0, "no deletion of main's concurrent edit");
    assert_eq!(fork.merge_base_length, 4, "merge_base_length = len('base')");
}

#[test]
fn diff_from_fork_dynamic_gca_moves_after_ff_merge() {
    // The baseline is the dynamic GCA, not the static fork_point: after main fast-forwards into
    // feat the GCA is main's head ("base-main"); a static baseline would report "-main-feat".
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();
    repo.create_branch(&b("feat"), &b(GENESIS_BRANCH)).unwrap();
    // feat doesn't edit concurrently, so the merge is a clean fast-forward and the GCA moves up.
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(4, "-main").unwrap())
        .unwrap();
    repo.merge_branch(&b("feat"), &b(GENESIS_BRANCH)).unwrap();
    g.branch("feat")
        .write(|h| h.get_text("t").insert_unicode(9, "-feat").unwrap())
        .unwrap();

    let fork = g.diff_from_fork(&b("feat"), &b(GENESIS_BRANCH)).unwrap();
    let inserted = inserted_text(&fork.diff);
    assert_eq!(
        inserted, "-feat",
        "dynamic GCA = main's merged head: only feat's post-merge insert"
    );
    assert!(
        !inserted.contains("-main"),
        "a static fork_point baseline would wrongly include -main"
    );
    assert_eq!(
        fork.merge_base_length, 9,
        "merge_base_length = len('base-main') (dynamic base), not len('base')"
    );
    assert_eq!(delete_count(&fork.diff), 0);
}

#[test]
fn merge_preview_additive_and_empty_for_contained() {
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();
    repo.create_branch(&b("feat"), &b(GENESIS_BRANCH)).unwrap();
    g.branch("feat")
        .write(|h| h.get_text("t").insert_unicode(4, "-feat").unwrap())
        .unwrap();
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(4, "-main").unwrap())
        .unwrap();

    // Merging feat into main adds feat's insert and deletes none of main's "-main" edit (union is a
    // superset of the target).
    let preview = g.merge_preview(&b(GENESIS_BRANCH), &b("feat")).unwrap();
    assert_eq!(inserted_text(&preview), "-feat", "merge adds feat's insert");
    assert_eq!(
        delete_count(&preview),
        0,
        "no spurious deletion of the target's own edit"
    );

    repo.merge_branch(&b("feat"), &b(GENESIS_BRANCH)).unwrap();
    let empty = g.merge_preview(&b("feat"), &b(GENESIS_BRANCH)).unwrap();
    assert_eq!(inserted_text(&empty), "", "already-contained preview is empty");
    assert_eq!(delete_count(&empty), 0);
}

#[test]
fn merge_preview_loud_error_on_unresolvable_frontier() {
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "x").unwrap())
        .unwrap();
    let valid = g.frontier_of(&b(GENESIS_BRANCH)).unwrap();
    let bogus = Frontiers::from_id(ID::new(999_999, 999));
    assert!(
        g.union_frontier(&valid, &bogus).is_err(),
        "an unresolvable frontier is a LOUD error, never a silently-empty union"
    );
    assert!(g.union_frontier(&bogus, &valid).is_err());
}

#[test]
fn diff_branches_head_vs_head() {
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();
    repo.create_branch(&b("feat"), &b(GENESIS_BRANCH)).unwrap();
    g.branch("feat")
        .write(|h| h.get_text("t").insert_unicode(4, "-feat").unwrap())
        .unwrap();
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(4, "-main").unwrap())
        .unwrap();

    // Head-vs-head counts feat's side as a deletion (why it must not back divergence / merge
    // preview).
    let batch = g.diff_branches(&b("feat"), &b(GENESIS_BRANCH)).unwrap();
    assert_eq!(inserted_text(&batch), "-main");
    assert!(
        delete_count(&batch) > 0,
        "head-vs-head double-counts each side's concurrent change"
    );
}

#[test]
fn export_oplog_slice_round_trips() {
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "AA").unwrap())
        .unwrap();
    let from = g.frontier_of(&b(GENESIS_BRANCH)).unwrap();
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(2, "BB").unwrap())
        .unwrap();
    let to = g.frontier_of(&b(GENESIS_BRANCH)).unwrap();

    let base = g
        .export_oplog_slice(&Frontiers::default(), Some(&from))
        .unwrap();
    let slice = g.export_oplog_slice(&from, Some(&to)).unwrap();
    assert!(!slice.is_empty());

    let plain = LoroDoc::new();
    plain.import(&base).unwrap();
    plain.import(&slice).unwrap();
    assert_eq!(
        plain.get_text("t").to_string(),
        "AABB",
        "base + bounded slice reconstruct the full content"
    );

    let full = g
        .export_oplog_slice(&Frontiers::default(), None)
        .unwrap();
    let plain2 = LoroDoc::new();
    plain2.import(&full).unwrap();
    assert_eq!(plain2.get_text("t").to_string(), "AABB");
}

#[test]
fn conveniences_leave_head_frontiers_unchanged() {
    let repo = BranchingDocRepo::open().unwrap();
    let g = repo.open_doc("G".into());
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();
    repo.create_branch(&b("feat"), &b(GENESIS_BRANCH)).unwrap();
    g.branch("feat")
        .write(|h| h.get_text("t").insert_unicode(4, "-feat").unwrap())
        .unwrap();
    g.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(4, "-main").unwrap())
        .unwrap();

    let before_feat = g.frontier_of(&b("feat")).unwrap();
    let before_main = g.frontier_of(&b(GENESIS_BRANCH)).unwrap();
    let sf_feat = g.branch("feat").read(|h| h.state_frontiers()).unwrap();

    let _ = g.diff_from_fork(&b("feat"), &b(GENESIS_BRANCH)).unwrap();
    let _ = g.merge_preview(&b(GENESIS_BRANCH), &b("feat")).unwrap();
    let _ = g.diff_branches(&b("feat"), &b(GENESIS_BRANCH)).unwrap();

    assert_eq!(g.frontier_of(&b("feat")).unwrap(), before_feat);
    assert_eq!(g.frontier_of(&b(GENESIS_BRANCH)).unwrap(), before_main);
    assert_eq!(
        g.branch("feat").read(|h| h.state_frontiers()).unwrap(),
        sf_feat,
        "the scratch-head diffs never disturb feat's live head"
    );
}

// Branch.lens(): copy-on-open pin, fast-forward-only guard, lens-observer delivery.

/// A forward import advances a bound lens in place; lens and branch subscribers fire.
#[test]
fn lens_forward_import_advances_and_fires() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

    let a = BranchingDocRepo::open().unwrap();
    let ga = a.open_doc("G".into());
    ga.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "hello").unwrap())
        .unwrap();
    let idx_base = a.index().export(ExportMode::all_updates()).unwrap();
    let content_base = ga.export(ExportMode::all_updates()).unwrap();
    let vv_before = ga.oplog_vv();
    ga.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(5, " world").unwrap())
        .unwrap();
    let idx_ahead = a.index().export(ExportMode::all_updates()).unwrap();
    let content_fwd = ga
        .export(ExportMode::updates_owned(vv_before))
        .unwrap();

    let bb = BranchingDocRepo::open().unwrap();
    let gb = bb.open_doc("G".into());
    bb.index().import(&idx_base).unwrap();
    gb.import(&content_base).unwrap();
    let lens = gb.branch(GENESIS_BRANCH).lens().unwrap();
    assert_eq!(lens.get_text("t").to_string(), "hello", "lens opens at base");
    let lens_head = gb.bound_head(&b(GENESIS_BRANCH)).unwrap();

    let lens_fires = Arc::new(AtomicUsize::new(0));
    let lf = lens_fires.clone();
    let _lens_sub = lens.subscribe_root(Arc::new(move |_ev: crate::event::DiffEvent| {
        lf.fetch_add(1, SeqCst);
    }));
    let branch_fires = Arc::new(AtomicUsize::new(0));
    let bf = branch_fires.clone();
    let _branch_sub = gb
        .branch(GENESIS_BRANCH)
        .subscribe_root(Arc::new(move |_ev: crate::event::DiffEvent| {
            bf.fetch_add(1, SeqCst);
        }))
        .unwrap();

    gb.import(&content_fwd).unwrap();
    bb.import_index(&idx_ahead).unwrap();

    assert_eq!(
        lens.get_text("t").to_string(),
        "hello world",
        "the bound lens advanced FORWARD in place on import"
    );
    assert_eq!(
        gb.bound_head(&b(GENESIS_BRANCH)),
        Some(lens_head),
        "the lens head is pinned: no rebind on the forward advance"
    );
    assert!(
        lens_fires.load(SeqCst) >= 1,
        "the forward advance was delivered through the lens's OWN observer (B.4)"
    );
    assert_eq!(
        branch_fires.load(SeqCst),
        1,
        "the branch subscription fired EXACTLY once per advance (no double-delivery)"
    );
    assert!(!gb.lens_held(&b(GENESIS_BRANCH)), "caught up: lensHeld is false");
}

/// A lineage-ahead import HOLDS the bound lens (no regress, no event); content catch-up fills it.
#[test]
fn lens_lineage_ahead_holds_then_fills() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

    let a = BranchingDocRepo::open().unwrap();
    let ga = a.open_doc("G".into());
    ga.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "BASE").unwrap())
        .unwrap();
    let idx_base = a.index().export(ExportMode::all_updates()).unwrap();
    let content_base = ga.export(ExportMode::all_updates()).unwrap();
    let vv_before = ga.oplog_vv();
    ga.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(4, " MORE").unwrap())
        .unwrap();
    let idx_ahead = a.index().export(ExportMode::all_updates()).unwrap();
    let content_fwd = ga.export(ExportMode::updates_owned(vv_before)).unwrap();

    let bb = BranchingDocRepo::open().unwrap();
    let gb = bb.open_doc("G".into());
    bb.index().import(&idx_base).unwrap();
    gb.import(&content_base).unwrap();
    let lens = gb.branch(GENESIS_BRANCH).lens().unwrap();
    assert_eq!(lens.get_text("t").to_string(), "BASE");

    let branch_fires = Arc::new(AtomicUsize::new(0));
    let bf = branch_fires.clone();
    let _branch_sub = gb
        .branch(GENESIS_BRANCH)
        .subscribe_root(Arc::new(move |_ev: crate::event::DiffEvent| {
            bf.fetch_add(1, SeqCst);
        }))
        .unwrap();

    bb.import_index(&idx_ahead).unwrap();
    assert_eq!(
        lens.get_text("t").to_string(),
        "BASE",
        "HOLD: the lens did not regress on a lineage-ahead frame"
    );
    assert_eq!(
        branch_fires.load(SeqCst),
        0,
        "no delete-all event reached the subscription (nothing to suppress downstream)"
    );
    assert!(
        gb.lens_held(&b(GENESIS_BRANCH)),
        "lensHeld is TRUE while behind the recorded content"
    );

    gb.import(&content_fwd).unwrap();
    let _ = gb
        .branch(GENESIS_BRANCH)
        .read(|h| h.get_text("t").to_string());
    assert_eq!(
        lens.get_text("t").to_string(),
        "BASE MORE",
        "the lens filled FORWARD once content caught up"
    );
    assert!(
        !gb.lens_held(&b(GENESIS_BRANCH)),
        "lensHeld is FALSE after catch-up"
    );
}

/// A sibling converging on an open lens head's frontier neither shares into nor retires it.
#[test]
fn lens_sibling_convergence_does_not_break_open_head() {
    let repo = BranchingDocRepo::open().unwrap();
    let doc = repo.open_doc("G".into());
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();

    let lens = doc.branch(GENESIS_BRANCH).lens().unwrap();
    let main_head = doc.bound_head(&b(GENESIS_BRANCH)).unwrap();
    assert_eq!(doc.refs_of(&b(GENESIS_BRANCH)), Some(1));

    repo.create_branch(&b("sib"), &b(GENESIS_BRANCH)).unwrap();
    let _ = doc.branch("sib").read(|h| h.get_text("t").to_string());
    assert_eq!(
        doc.refs_of(&b(GENESIS_BRANCH)),
        Some(1),
        "main's lens head stayed refs == 1 (no sibling shared into it)"
    );
    assert_eq!(
        doc.bound_head(&b(GENESIS_BRANCH)),
        Some(main_head),
        "main's lens head was not retired or rebound"
    );
    assert_ne!(
        doc.bound_head(&b("sib")),
        Some(main_head),
        "the sibling got its own head, not the lens head"
    );

    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(4, "!").unwrap())
        .unwrap();
    assert_eq!(lens.get_text("t").to_string(), "base!", "lens write survived");
}

/// A backward `advance` still regresses: it is exempt from the fast-forward guard.
#[test]
fn lens_backward_advance_is_guard_exempt() {
    let repo = BranchingDocRepo::open().unwrap();
    let doc = repo.open_doc("G".into());
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "A").unwrap())
        .unwrap();
    let at_a = doc.frontier_of(&b(GENESIS_BRANCH)).unwrap();
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(1, "B").unwrap())
        .unwrap();
    let lens = doc.branch(GENESIS_BRANCH).lens().unwrap();
    assert_eq!(lens.get_text("t").to_string(), "AB");
    doc.advance(&b(GENESIS_BRANCH), &at_a).unwrap();
    assert_eq!(
        doc.branch(GENESIS_BRANCH)
            .read(|h| h.get_text("t").to_string())
            .unwrap(),
        "A",
        "a backward advance regressed the branch (guard-exempt)"
    );
}

/// `import_index` advances an OPEN doc's lens and does not open a never-opened doc.
#[test]
fn lens_import_index_advances_open_doc_only() {
    let a = BranchingDocRepo::open().unwrap();
    let ga = a.open_doc("G".into());
    let ha = a.open_doc("H".into());
    ga.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "g").unwrap())
        .unwrap();
    ha.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "h").unwrap())
        .unwrap();
    let idx = a.index().export(ExportMode::all_updates()).unwrap();
    let g_content = ga.export(ExportMode::all_updates()).unwrap();

    let bb = BranchingDocRepo::open().unwrap();
    let gb = bb.open_doc("G".into());
    gb.import(&g_content).unwrap();
    let lens = gb.branch(GENESIS_BRANCH).lens().unwrap();
    bb.import_index(&idx).unwrap();
    assert_eq!(
        lens.get_text("t").to_string(),
        "g",
        "G's lens advanced to its held content on the index import"
    );
    // H was never opened: import_index loops only already-open docs.
    assert!(bb.branches().contains(&b(GENESIS_BRANCH)));
}

/// `lens_generation` is stable across `lens()` calls and moves after a rebind.
#[test]
fn lens_generation_stable_then_bumps_on_rebind() {
    let repo = BranchingDocRepo::open().unwrap();
    let doc = repo.open_doc("G".into());
    doc.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(0, "base").unwrap())
        .unwrap();
    repo.create_branch(&b("draft"), &b(GENESIS_BRANCH)).unwrap();
    let _l1 = doc.branch("draft").lens().unwrap();
    let g1 = doc.lens_generation(&b("draft"));
    let _l2 = doc.branch("draft").lens().unwrap();
    let g2 = doc.lens_generation(&b("draft"));
    assert_eq!(g1, g2, "generation stable across lens() calls with no rebind");
    repo.delete_branch(&b("draft")).unwrap();
    let g3 = doc.lens_generation(&b("draft"));
    assert!(g3 > g2, "generation bumped on the delete/unbind rebind");
}

/// A container-scoped `Branch.subscribe` hears only its container: "t" fires, untouched "u" not.
#[test]
fn lens_container_scoped_subscribe_parity_after_observer_switch() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

    let a = BranchingDocRepo::open().unwrap();
    let ga = a.open_doc("G".into());
    ga.branch(GENESIS_BRANCH)
        .write(|h| {
            h.get_text("t").insert_unicode(0, "hi").unwrap();
            h.get_map("u").insert("k", "v").unwrap();
        })
        .unwrap();
    let idx_base = a.index().export(ExportMode::all_updates()).unwrap();
    let c_base = ga.export(ExportMode::all_updates()).unwrap();
    let vv = ga.oplog_vv();
    ga.branch(GENESIS_BRANCH)
        .write(|h| h.get_text("t").insert_unicode(2, "!").unwrap())
        .unwrap();
    let idx_ahead = a.index().export(ExportMode::all_updates()).unwrap();
    let c_fwd = ga.export(ExportMode::updates_owned(vv)).unwrap();

    let bb = BranchingDocRepo::open().unwrap();
    let gb = bb.open_doc("G".into());
    bb.index().import(&idx_base).unwrap();
    gb.import(&c_base).unwrap();
    let lens = gb.branch(GENESIS_BRANCH).lens().unwrap();
    let cid_t = gb.branch(GENESIS_BRANCH).read(|h| h.get_text("t").id()).unwrap();
    let cid_u = gb.branch(GENESIS_BRANCH).read(|h| h.get_map("u").id()).unwrap();

    let t_fires = Arc::new(AtomicUsize::new(0));
    let tf = t_fires.clone();
    let _t_sub = gb
        .subscribe_branch(
            &b(GENESIS_BRANCH),
            Some(cid_t),
            Arc::new(move |_ev: crate::event::DiffEvent| {
                tf.fetch_add(1, SeqCst);
            }),
        )
        .unwrap();
    let u_fires = Arc::new(AtomicUsize::new(0));
    let uf = u_fires.clone();
    let _u_sub = gb
        .subscribe_branch(
            &b(GENESIS_BRANCH),
            Some(cid_u),
            Arc::new(move |_ev: crate::event::DiffEvent| {
                uf.fetch_add(1, SeqCst);
            }),
        )
        .unwrap();

    gb.import(&c_fwd).unwrap();
    bb.import_index(&idx_ahead).unwrap();

    assert_eq!(
        lens.get_text("t").to_string(),
        "hi!",
        "the lens advanced forward on the 't' edit"
    );
    assert!(
        t_fires.load(SeqCst) >= 1,
        "the 't' container subscriber fired for the advance touching 't'"
    );
    assert_eq!(
        u_fires.load(SeqCst),
        0,
        "the 'u' container subscriber did NOT fire (container filtering preserved through the head observer)"
    );
}
