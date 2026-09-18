export * from "loro-wasm";
export type * from "loro-wasm";
import {
  AwarenessWasm,
  Branch,
  BranchingDoc,
  BranchingDocHead,
  BranchingDocRepo,
  BranchingIndex,
  EphemeralStoreWasm,
  PeerID,
  Container,
  ContainerID,
  ContainerType,
  LoroCounter,
  LoroDoc,
  LoroList,
  LoroMap,
  LoroMovableList,
  LoroText,
  LoroTree,
  LoroTreeNode,
  OpId,
  Value,
  AwarenessListener,
  EphemeralListener,
  EphemeralLocalListener,
  UndoManager,
  callPendingEvents,
  isValidContainerId,
} from "loro-wasm";

/**
 * @deprecated Please use LoroDoc
 */
export class Loro extends LoroDoc {}

const CONTAINER_TYPES = [
  "Map",
  "Text",
  "List",
  "Tree",
  "MovableList",
  "Counter",
];

export function isContainerId(s: string): s is ContainerID {
  return isValidContainerId(s);
}

/**  Whether the value is a container.
 *
 * # Example
 *
 * ```ts
 * const doc = new LoroDoc();
 * const map = doc.getMap("map");
 * const list = doc.getList("list");
 * const text = doc.getText("text");
 * isContainer(map); // true
 * isContainer(list); // true
 * isContainer(text); // true
 * isContainer(123); // false
 * isContainer("123"); // false
 * isContainer({}); // false
 * ```
 */
export function isContainer(value: any): value is Container {
  if (typeof value !== "object" || value == null) {
    return false;
  }

  const p = Object.getPrototypeOf(value);
  if (p == null || typeof p !== "object" || typeof p["kind"] !== "function") {
    return false;
  }

  return CONTAINER_TYPES.includes(value.kind());
}

/**  Get the type of a value that may be a container.
 *
 * # Example
 *
 * ```ts
 * const doc = new LoroDoc();
 * const map = doc.getMap("map");
 * const list = doc.getList("list");
 * const text = doc.getText("text");
 * getType(map); // "Map"
 * getType(list); // "List"
 * getType(text); // "Text"
 * getType(123); // "Json"
 * getType("123"); // "Json"
 * getType({}); // "Json"
 * ```
 */
export function getType<T>(
  value: T,
): T extends LoroText
  ? "Text"
  : T extends LoroMap<any>
    ? "Map"
    : T extends LoroTree<any>
      ? "Tree"
      : T extends LoroList<any>
        ? "List"
        : T extends LoroCounter
          ? "Counter"
          : "Json" {
  if (isContainer(value)) {
    return value.kind() as unknown as any;
  }

  return "Json" as any;
}

export function newContainerID(id: OpId, type: ContainerType): ContainerID {
  return `cid:${id.counter}@${id.peer}:${type}`;
}

export function newRootContainerID(
  name: string,
  type: ContainerType,
): ContainerID {
  return `cid:root-${name}:${type}`;
}

/**
 * @deprecated Please use `EphemeralStore` instead.
 *
 * Awareness is a structure that allows to track the ephemeral state of the peers.
 *
 * If we don't receive a state update from a peer within the timeout, we will remove their state.
 * The timeout is in milliseconds. This can be used to handle the offline state of a peer.
 */
export class Awareness<T extends Value = Value> {
  inner: AwarenessWasm<T>;
  private peer: PeerID;
  private timer: number | undefined;
  private timeout: number;
  private listeners: Set<AwarenessListener> = new Set();
  constructor(peer: PeerID, timeout: number = 30000) {
    this.inner = new AwarenessWasm(peer, timeout);
    this.peer = peer;
    this.timeout = timeout;
  }

  apply(bytes: Uint8Array, origin = "remote") {
    const { updated, added } = this.inner.apply(bytes);
    this.listeners.forEach((listener) => {
      listener({ updated, added, removed: [] }, origin);
    });

    this.startTimerIfNotEmpty();
  }

  setLocalState(state: T) {
    const wasEmpty = this.inner.getState(this.peer) == null;
    this.inner.setLocalState(state);
    if (wasEmpty) {
      this.listeners.forEach((listener) => {
        listener(
          { updated: [], added: [this.inner.peer()], removed: [] },
          "local",
        );
      });
    } else {
      this.listeners.forEach((listener) => {
        listener(
          { updated: [this.inner.peer()], added: [], removed: [] },
          "local",
        );
      });
    }

    this.startTimerIfNotEmpty();
  }

  getLocalState(): T | undefined {
    return this.inner.getState(this.peer);
  }

  getAllStates(): Record<PeerID, T> {
    return this.inner.getAllStates();
  }

  encode(peers: PeerID[]): Uint8Array {
    return this.inner.encode(peers);
  }

  encodeAll(): Uint8Array {
    return this.inner.encodeAll();
  }

  addListener(listener: AwarenessListener) {
    this.listeners.add(listener);
  }

  removeListener(listener: AwarenessListener) {
    this.listeners.delete(listener);
  }

  peers(): PeerID[] {
    return this.inner.peers();
  }

  destroy() {
    clearInterval(this.timer);
    this.listeners.clear();
  }

  private startTimerIfNotEmpty() {
    if (this.inner.isEmpty() || this.timer != null) {
      return;
    }

    this.timer = setInterval(() => {
      const removed = this.inner.removeOutdated();
      if (removed.length > 0) {
        this.listeners.forEach((listener) => {
          listener({ updated: [], added: [], removed }, "timeout");
        });
      }
      if (this.inner.isEmpty()) {
        clearInterval(this.timer);
        this.timer = undefined;
      }
    }, this.timeout / 2) as unknown as number;
  }
}

/**
 * EphemeralStore tracks ephemeral key-value state across peers.
 *
 * - Use it for lightweight presence/state like cursors, selections, and UI hints.
 * - Conflict resolution is timestamp-based LWW (Last-Write-Wins) per key.
 * - Timeout unit: milliseconds.
 * - After timeout: keys are considered expired. They are omitted from
 *   `encode(key)`, `encodeAll()` and `getAllStates()`. A periodic cleanup runs
 *   while the store is non-empty and removes expired keys; when removals happen
 *   subscribers receive an event with `by: "timeout"` and the `removed` keys.
 *
 * See: https://loro.dev/docs/tutorial/ephemeral
 *
 * @param timeout Inactivity timeout in milliseconds (default: 30000). If a key
 * doesn't receive updates within this duration, it will expire and be removed
 * on the next cleanup tick.
 *
 * @example
 * ```ts
 * const store = new EphemeralStore();
 * const store2 = new EphemeralStore();
 * // Subscribe to local updates and forward over the wire
 * store.subscribeLocalUpdates((data) => {
 *   store2.apply(data);
 * });
 * // Subscribe to all updates (including removals by timeout)
 * store2.subscribe((event) => {
 *   console.log("event:", event);
 * });
 * // Set a value
 * store.set("key", "value");
 * // Encode the value
 * const encoded = store.encode("key");
 * // Apply the encoded value
 * store2.apply(encoded);
 * ```
 */
export class EphemeralStore<
  T extends Record<string, Value> = Record<string, Value>,
> {
  inner: EphemeralStoreWasm;
  private timer: number | undefined;
  private timeout: number;
  constructor(timeout: number = 30000) {
    this.inner = new EphemeralStoreWasm(timeout);
    this.timeout = timeout;
  }

  apply(bytes: Uint8Array) {
    this.inner.apply(bytes);
    this.startTimerIfNotEmpty();
  }

  set<K extends keyof T>(key: K, value: T[K]) {
    this.inner.set(key as string, value);
    this.startTimerIfNotEmpty();
  }

  delete<K extends keyof T>(key: K) {
    this.inner.delete(key as string);
  }

  get<K extends keyof T>(key: K): T[K] | undefined {
    return this.inner.get(key as string);
  }

  getAllStates(): Partial<T> {
    return this.inner.getAllStates();
  }

  encode<K extends keyof T>(key: K): Uint8Array {
    return this.inner.encode(key as string);
  }

  encodeAll(): Uint8Array {
    return this.inner.encodeAll();
  }

  keys(): string[] {
    return this.inner.keys();
  }

  destroy() {
    clearInterval(this.timer);
  }

  subscribe(listener: EphemeralListener) {
    return this.inner.subscribe(listener);
  }

  subscribeLocalUpdates(listener: EphemeralLocalListener) {
    return this.inner.subscribeLocalUpdates(listener);
  }

  private startTimerIfNotEmpty() {
    if (this.inner.isEmpty() || this.timer != null) {
      return;
    }

    this.timer = setInterval(() => {
      this.inner.removeOutdated();
      if (this.inner.isEmpty()) {
        clearInterval(this.timer);
        this.timer = undefined;
      }
    }, this.timeout / 2) as unknown as number;
  }
}

LoroDoc.prototype.toJsonWithReplacer = function (
  replacer: (
    key: string | number,
    value: Value | Container,
  ) => Value | Container | undefined,
) {
  const processed = new Set<string>();
  const doc = this;
  const m = (key: string | number, value: Value): Value | undefined => {
    if (typeof value === "string") {
      if (isContainerId(value) && !processed.has(value)) {
        processed.add(value);
        const container = doc.getContainerById(value);
        if (container == null) {
          throw new Error(`ContainerID not found: ${value}`);
        }

        const ans = replacer(key, container);
        if (ans === container) {
          const ans = container.getShallowValue();
          if (typeof ans === "object") {
            return run(ans as any);
          }

          return ans;
        }

        if (isContainer(ans)) {
          throw new Error(
            "Using new container is not allowed in toJsonWithReplacer",
          );
        }

        if (typeof ans === "object" && ans != null) {
          return run(ans as any);
        }

        return ans;
      }
    }

    if (typeof value === "object" && value != null) {
      return run(value as Record<string, Value>);
    }

    const ans = replacer(key, value);
    if (isContainer(ans)) {
      throw new Error(
        "Using new container is not allowed in toJsonWithReplacer",
      );
    }

    return ans;
  };

  const run = (layer: Record<string, Value> | Value[]): Value => {
    if (Array.isArray(layer)) {
      return layer
        .map((item, index) => {
          return m(index, item);
        })
        .filter((item): item is NonNullable<typeof item> => item !== undefined);
    }

    const result: Record<string, Value> = {};
    for (const [key, value] of Object.entries(layer)) {
      const ans = m(key, value);
      if (ans !== undefined) {
        result[key] = ans;
      }
    }

    return result;
  };

  const layer = doc.getShallowValue();
  return run(layer);
};

// A head exposes the same head-safe `toJsonWithReplacer` as a doc; reuse the exact impl.
BranchingDocHead.prototype.toJsonWithReplacer =
  LoroDoc.prototype.toJsonWithReplacer;

export function idStrToId(idStr: `${number}@${PeerID}`): OpId {
  const [counter, peer] = idStr.split("@");
  return {
    counter: parseInt(counter),
    peer: peer as PeerID,
  };
}

const CALL_PENDING_EVENTS_WRAPPED = Symbol("loro.callPendingEventsWrapped");

function decorateMethod(prototype: object, method: PropertyKey) {
  const descriptor = Object.getOwnPropertyDescriptor(prototype, method);
  if (!descriptor || typeof descriptor.value !== "function") {
    return;
  }

  const original = descriptor.value as (...args: unknown[]) => unknown;
  if ((original as any)[CALL_PENDING_EVENTS_WRAPPED]) {
    return;
  }

  const wrapped = function (this: unknown, ...args: unknown[]) {
    let result;
    try {
      result = original.apply(this, args);
      return result;
    } finally {
      if (result && typeof (result as Promise<unknown>).then === "function") {
        (result as Promise<unknown>).finally(() => {
          callPendingEvents();
        });
      } else {
        callPendingEvents();
      }
    }
  };

  (wrapped as any)[CALL_PENDING_EVENTS_WRAPPED] = true;

  Object.defineProperty(prototype, method, {
    ...descriptor,
    value: wrapped,
  });
}

function decorateMethods(prototype: object, methods: PropertyKey[]) {
  for (const method of methods) {
    decorateMethod(prototype, method);
  }
}

function decorateAllPrototypeMethods(prototype: object) {
  const visited = new Set<PropertyKey>();
  let current: object | null = prototype;
  while (
    current &&
    current !== Object.prototype &&
    current !== Function.prototype
  ) {
    for (const property of Object.getOwnPropertyNames(current)) {
      if (property === "constructor" || visited.has(property)) {
        continue;
      }
      visited.add(property);
      decorateMethod(current, property);
    }

    for (const symbol of Object.getOwnPropertySymbols(current)) {
      if (visited.has(symbol)) {
        continue;
      }
      visited.add(symbol);
      decorateMethod(current, symbol);
    }

    current = Object.getPrototypeOf(current) as object | null;
  }
}

decorateMethods(LoroDoc.prototype, [
  "setDetachedEditing",
  "attach",
  "detach",
  "fork",
  "forkAt",
  "checkoutToLatest",
  "checkout",
  "commit",
  "getCursorPos",
  "revertTo",
  "export",
  "exportJsonUpdates",
  "exportJsonInIdSpan",
  "importJsonUpdates",
  "import",
  "importUpdateBatch",
  "importBatch",
  "travelChangeAncestors",
  "getChangedContainersIn",
  "diff",
  "applyDiff",
  "setPeerId",
]);

decorateMethods(EphemeralStoreWasm.prototype, [
  "set",
  "delete",
  "apply",
  "removeOutdated",
]);

decorateMethods(UndoManager.prototype, ["undo", "redo"]);

// A head shares LoroDoc's pending-event decoration for the head-safe mutating methods.
decorateMethods(BranchingDocHead.prototype, [
  "commit",
  "getCursorPos",
  "revertTo",
  "applyDiff",
]);

// `Branch` methods that can enqueue branch-subscriber events, auto-flushed here so a consumer
// observes them with NO manual `callPendingEvents()`:
//   - `write` commits its resolved content head, enqueuing the local-commit event.
//   - `read`/`subscribe`/`subscribeRoot` each RESOLVE the branch first, and a resolve that
//     MOVES the branch (a lazy realization on access -- the common content-then-lineage
//     arrival order, where the move is realized on the next `read`) now dispatches its diff.
//     Without this decoration that event would sit in the global pending queue and be flushed
//     by an unrelated later decorated call (possibly on another repo), never inside the `read`.
decorateMethods(Branch.prototype, ["write", "read", "subscribe", "subscribeRoot"]);

// `BranchingDoc` mutating methods that enqueue branch-subscriber events, auto-flushed here:
//   - `import`: lands remote ops then EAGERLY re-resolves every known branch (the `Delegated`
//     policy's `after_import`); same precedent as the decorated `LoroDoc.import`.
//   - `merge` / `advance`: every advance arm (catch-up, copy+advance, and the fast-forward
//     REBIND-to-existing arm) now synthesizes and dispatches the branch's `diff(X -> Y)`,
//     tagged `by:"import"` with origin `"advance"`. Without this flush a subscriber goes STALE
//     on a merge; a fast-forward is no longer silent (the rebind arm delivers a synthesized
//     diff, so there IS an event to flush here).
decorateMethods(BranchingDoc.prototype, ["import", "merge", "advance"]);

// `BranchingIndex.import` lands lineage ops into the shared index op log. On a policy whose
// `after_import` re-resolves content branches this can enqueue branch-subscriber events; flush
// them at `import`'s exit so a lineage import delivers its consequences before returning to JS.
decorateMethods(BranchingIndex.prototype, ["import"]);

// `BranchingDocRepo.createBranch` commits a branch-marker op onto the shared INDEX op log (the
// repo's `create_index_branch` sets the new branch's `branch.name` marker and `commit_then_renew`s).
// That commit synchronously enqueues a `BranchingIndex.subscribeLocalUpdates` local-update
// callback into the global pending-event queue, but the queue only flushes when a decorated
// method runs `callPendingEvents()`. Without this decoration the enqueued frame is never
// delivered (0 fires) and the scheduled microtask check logs `[LORO_INTERNAL_ERROR] Event not
// called`, so a wire adaptor driven off the index stream never learns of a locally-created
// branch. Auto-flushing here makes `createBranch(...)` deliver the index marker frame with NO
// manual `callPendingEvents()`, mirroring the decorated content-mutating methods above.
// `deleteBranch` is CURRENTLY local-only registry cleanup (`delete_index_branch` commits no op —
// durable cross-peer deletion is a Phase-5 wrapper-lifecycle follow-up), so its decoration flushes
// an empty queue today; it is included to future-proof the method for when a durable delete op lands.
decorateMethods(BranchingDocRepo.prototype, ["createBranch", "deleteBranch"]);

// ---------------------------------------------------------------------------
// Head-safety drift guard (compile-time).
//
// `BranchingDocHead` must expose exactly LoroDoc's surface MINUS the history-bearing
// methods (`HistoryKeys`). Two checks enforce this at `tsc` time:
//   1. name-set exactness: every history method is absent from the head and every
//      head-safe method is present (adding/removing either flips the keyof set);
//   2. signature-shape compatibility of the head-safe surface.
// This is the wasm-layer enforcement of the head-safe/history partition. It deliberately
// does NOT use an exact `Equals<Omit<LoroDoc, HistoryKeys>, BranchingDocHead>`: LoroDoc's
// published type is a class+generic-interface declaration-merge that only type-checks
// under `skipLibCheck`, so an independently-generated `BranchingDocHead` cannot reach
// exact structural identity with an `Omit` of it. See the branchingdocrepo devlog.
type _Equals<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B
  ? 1
  : 2)
  ? true
  : false;
type _MutualExtends<A, B> = [A] extends [B]
  ? [B] extends [A]
    ? true
    : false
  : false;

// The exact excluded set: every LoroDoc instance method that is NOT head-safe.
// A LoroDoc method that is neither forwarded onto the head nor listed here trips the
// name-set check below — that is the drift signal.
type HistoryKeys =
  | "setDetachedEditing"
  | "isDetachedEditingEnabled"
  | "setRecordTimestamp"
  | "setChangeMergeInterval"
  | "configTextStyle"
  | "configDefaultTextStyle"
  | "attach"
  | "isDetached"
  | "detach"
  | "fork"
  | "forkAt"
  | "checkoutToLatest"
  | "travelChangeAncestors"
  | "findIdSpansBetween"
  | "checkout"
  | "setPeerId"
  | "subscribeJsonpath"
  | "shallowSinceVV"
  | "isShallow"
  | "shallowSinceFrontiers"
  | "oplogVersion"
  | "oplogFrontiers"
  | "cmpFrontiers"
  | "export"
  | "exportJsonUpdates"
  | "exportJsonInIdSpan"
  | "importJsonUpdates"
  | "import"
  | "importUpdateBatch"
  | "importBatch"
  | "subscribe"
  | "subscribeLocalUpdates"
  | "debugHistory"
  | "getAllChanges"
  | "getChangeAt"
  | "getChangeAtLamport"
  | "getOpsInChange"
  | "frontiersToVV"
  | "vvToFrontiers"
  | "getChangedContainersIn"
  | "diff"
  | "getUncommittedOpsAsJson"
  | "subscribeFirstCommitFromPeer"
  | "deleteRootContainer"
  | "setHideEmptyRootContainers"
  | "toContainerTree";

// (C.1) name-set exactness.
const _headKeysMatchOmit: _Equals<
  keyof Omit<LoroDoc, HistoryKeys>,
  keyof BranchingDocHead
> = true;
// (C.2) signature-shape compatibility of the head-safe surface.
const _headShapeMatchesOmit: _MutualExtends<
  Omit<LoroDoc, HistoryKeys>,
  BranchingDocHead
> = true;
void _headKeysMatchOmit;
void _headShapeMatchesOmit;
