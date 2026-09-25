import { expect } from "bun:test"
import { Cause, Effect, Exit, Fiber, Layer } from "effect"
import { EventV2Bridge } from "../../src/event-v2-bridge"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { Permission } from "../../src/permission"
import { Plugin } from "../../src/plugin"
import { InstanceBootstrap } from "../../src/project/bootstrap"
import { InstanceStore } from "../../src/project/instance-store"
import { testEffect } from "../lib/effect"
import { SessionID } from "../../src/session/schema"
import { AppNodeBuilder } from "@opencode-ai/core/effect/app-node-builder"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"

/** Recorded hook dispatches, reset per test. */
let calls: { name: string; input: any }[] = []
/** What the stand-in handler answers with. */
let verdict: "ask" | "allow" | "deny" = "ask"

const pluginMock = Layer.mock(Plugin.Service, {
  trigger: ((name: string, input: unknown, output: { status: string }) => {
    calls.push({ name, input })
    if (name === "permission.ask") output.status = verdict
    return Effect.succeed(output)
  }) as unknown as Plugin.Interface["trigger"],
})

const noopBootstrap = Layer.succeed(InstanceBootstrap.Service, InstanceBootstrap.Service.of({ run: Effect.void }))
const env = AppNodeBuilder.build(
  LayerNode.group([Permission.node, EventV2Bridge.node, CrossSpawnSpawner.node, InstanceStore.node]),
  [
    [InstanceStore.bootstrapNode, noopBootstrap],
    [Plugin.node, pluginMock],
  ],
)
const it = testEffect(env)

const ask = (input: Parameters<Permission.Interface["ask"]>[0]) =>
  Effect.gen(function* () {
    const permission = yield* Permission.Service
    return yield* permission.ask(input)
  })

const base = {
  sessionID: SessionID.make("session_test"),
  permission: "bash",
  metadata: {},
  always: [],
}

const reset = (next: typeof verdict = "ask") => {
  calls = []
  verdict = next
}

const hookCalls = () => calls.filter((c) => c.name === "permission.ask")

it.instance(
  "a statically allowed request never reaches the hook", () =>
  Effect.gen(function* () {
    reset()
    yield* ask({ ...base, patterns: ["ls"], ruleset: [{ permission: "bash", pattern: "*", action: "allow" }] })
    expect(hookCalls()).toHaveLength(0)
  }),
  { git: true },
)

it.instance(
  "a statically denied request never reaches the hook", () =>
  Effect.gen(function* () {
    reset()
    const exit = yield* Effect.exit(
      ask({ ...base, patterns: ["rm -rf /"], ruleset: [{ permission: "bash", pattern: "*", action: "deny" }] }),
    )
    expect(Exit.isFailure(exit)).toBe(true)
    expect(hookCalls()).toHaveLength(0)
  }),
  { git: true },
)

it.instance(
  "a residual ask dispatches the hook exactly once", () =>
  Effect.gen(function* () {
    reset("allow") // resolve it immediately so the effect does not block on a user
    yield* ask({ ...base, patterns: ["ls"], ruleset: [] })
    expect(hookCalls()).toHaveLength(1)
    expect(hookCalls()[0]!.input.permission).toBe("bash")
    expect(hookCalls()[0]!.input.patterns).toEqual(["ls"])
  }),
  { git: true },
)

it.instance(
  "an allow verdict resolves without prompting the user", () =>
  Effect.gen(function* () {
    reset("allow")
    const permission = yield* Permission.Service
    yield* ask({ ...base, patterns: ["ls"], ruleset: [] })
    // Nothing was left waiting for a human.
    expect(yield* permission.list()).toHaveLength(0)
  }),
  { git: true },
)

it.instance(
  "a deny verdict fails the request", () =>
  Effect.gen(function* () {
    reset("deny")
    const exit = yield* Effect.exit(ask({ ...base, patterns: ["ls"], ruleset: [] }))
    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) expect(String(Cause.squash(exit.cause))).toContain("Denied")
  }),
  { git: true },
)

it.instance(
  "an untouched verdict still reaches the user",
  () =>
    Effect.gen(function* () {
      reset("ask")
      const permission = yield* Permission.Service
      const fiber = yield* ask({ ...base, patterns: ["ls"], ruleset: [] }).pipe(Effect.forkScoped)

      // Poll rather than yield once: the fiber registers the pending request asynchronously.
      let pending = yield* permission.list()
      while (pending.length === 0) {
        yield* Effect.sleep("5 millis")
        pending = yield* permission.list()
      }

      expect(pending).toHaveLength(1)
      yield* permission.reply({ requestID: pending[0]!.id, reply: "once" })
      yield* Fiber.await(fiber)
      expect(hookCalls()).toHaveLength(1)
    }),
  { git: true },
)
