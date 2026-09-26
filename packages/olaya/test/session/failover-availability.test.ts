/**
 * Availability, fail-back, waiting and stalls (olaya-provider-failover, gate F5).
 */
import { afterEach, describe, expect, test } from "bun:test"
import { Effect } from "effect"
import fs from "fs"
import os from "os"
import path from "path"
import { FailoverAvailability } from "../../src/failover/availability"
import { httpError, reply, TestLLMServer } from "../lib/llm-server"
import { it, project, run } from "../lib/session-loop"

afterEach(() => FailoverAvailability.clear())

describe("availability store", () => {
  test("cooldowns survive a restart", () => {
    const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "avail-")), "availability.json")
    FailoverAvailability.useFile(file)
    FailoverAvailability.mark("anthropic/claude-opus-5-5", {
      action: "switch-until",
      until: Date.now() + 3_600_000,
      reason: "usage limit reached",
    })
    FailoverAvailability.useFile(file) // a new process reading the same file
    expect(FailoverAvailability.available("anthropic/claude-opus-5-5")).toBe(false)
    expect(FailoverAvailability.get("anthropic/claude-opus-5-5")?.reason).toBe("usage limit reached")
    FailoverAvailability.recovered("anthropic/claude-opus-5-5")
    FailoverAvailability.useFile(file)
    expect(FailoverAvailability.available("anthropic/claude-opus-5-5")).toBe(true)
  })

  test("backoff grows with repeated failures and starts over after a success", () => {
    const now = Date.now()
    const verdict = { action: "switch" as const, reason: "overloaded" }
    expect(FailoverAvailability.mark("m", verdict, now).until).toBe(now + 60_000)
    expect(FailoverAvailability.mark("m", verdict, now).until).toBe(now + 5 * 60_000)
    expect(FailoverAvailability.mark("m", verdict, now).until).toBe(now + 15 * 60_000)
    FailoverAvailability.recovered("m")
    expect(FailoverAvailability.mark("m", verdict, now).until).toBe(now + 60_000)
  })

  test("a disabled model is tried again after 6 hours, but a chain of disabled models doesn't wait", () => {
    const now = Date.now()
    FailoverAvailability.mark("a", { action: "disable", reason: "authentication failed" }, now)
    FailoverAvailability.mark("b", { action: "disable", reason: "credit exhausted" }, now)
    expect(FailoverAvailability.available("a", now + 5 * 3_600_000)).toBe(false)
    expect(FailoverAvailability.available("a", now + 6 * 3_600_000 + 1)).toBe(true)
    expect(FailoverAvailability.earliest(["a", "b"])).toBeUndefined()
    FailoverAvailability.mark("c", { action: "switch-until", until: now + 120_000, reason: "rate limited" }, now)
    expect(FailoverAvailability.earliest(["a", "b", "c"])).toBe(now + 120_000)
  })
})

const on = (m: string) => (hit: { body: Record<string, unknown> }) => hit.body.model === m
type Body = { model?: string }
const models = (bodies: unknown[]) => (bodies as Body[]).map((b) => b.model)
const later = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

it.instance(
  "once the preferred model's cooldown ends, the session goes back to it at a safe point",
  () =>
    Effect.gen(function* () {
      yield* project(undefined, undefined, { failover: { models: ["test/cheap-model"] } })
      FailoverAvailability.mark("test/test-model", {
        action: "switch-until",
        until: Date.now() + 1_500,
        reason: "rate limited",
      })
      const { bodies } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          // the fallback's step takes longer than the cooldown
          yield* llm.pushMatch(
            on("cheap-model"),
            reply()
              .tool("todowrite", { todos: [{ content: "look", status: "pending", priority: "high", id: "1" }] })
              .wait(later(2_500))
              .item(),
          )
          yield* llm.pushMatch(on("test-model"), reply().text("done, back on the preferred model").stop().item())
        }),
      )
      expect(models(bodies)).toEqual(["cheap-model", "test-model"])
    }),
  30_000,
)

it.instance(
  "no fail-back while an edit waits for its test run",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* project(undefined, undefined, { failover: { models: ["test/cheap-model"] } })
      yield* Effect.promise(() => Bun.write(path.join(directory, "a.py"), "x = 1\n"))
      FailoverAvailability.mark("test/test-model", {
        action: "switch-until",
        until: Date.now() + 1_500,
        reason: "rate limited",
      })
      const { bodies } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          const edit = { filePath: path.join(directory, "a.py"), oldString: "x = 1", newString: "x = 2" }
          yield* llm.pushMatch(on("cheap-model"), reply().tool("edit", edit).wait(later(2_500)).item())
          yield* llm.pushMatch(on("cheap-model"), reply().text("edited; tests next").stop().item())
        }),
      )
      // the step after the edit stays on the fallback although the preferred model is back
      expect(models(bodies)).toEqual(["cheap-model", "cheap-model"])
    }),
  30_000,
)

it.instance(
  "with wait_for_reset, a preferred model returning soon is waited for instead of stepping down",
  () =>
    Effect.gen(function* () {
      yield* project(undefined, undefined, { failover: { models: ["test/cheap-model"], wait_for_reset: 1 } })
      FailoverAvailability.mark("test/test-model", {
        action: "switch-until",
        until: Date.now() + 2_000,
        reason: "rate limited",
      })
      const started = Date.now()
      const { bodies } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(on("test-model"), reply().text("done").stop().item())
        }),
      )
      expect(models(bodies)).toEqual(["test-model"])
      expect(Date.now() - started).toBeGreaterThanOrEqual(1_900)
    }),
  30_000,
)

it.instance(
  "when every model is cooling for longer than max_wait, the run ends instead of waiting",
  () =>
    Effect.gen(function* () {
      yield* project(undefined, undefined, { failover: { models: ["test/cheap-model"], max_wait: 1 } })
      FailoverAvailability.mark("test/test-model", {
        action: "switch-until",
        until: Date.now() + 3_600_000,
        reason: "usage limit reached",
      })
      FailoverAvailability.mark("test/cheap-model", {
        action: "switch-until",
        until: Date.now() + 3_600_000,
        reason: "usage limit reached",
      })
      const started = Date.now()
      const { bodies } = yield* run(Effect.void)
      expect(bodies).toHaveLength(0)
      expect(Date.now() - started).toBeLessThan(10_000)
    }),
  30_000,
)

it.instance(
  "a stream that goes quiet counts as stalled: one retry, then the next model takes over",
  () =>
    Effect.gen(function* () {
      yield* project(undefined, undefined, { failover: { models: ["test/cheap-model"], stall_timeout: 1 } })
      const { bodies, assistants } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(on("test-model"), reply().hang().item())
          yield* llm.pushMatch(on("test-model"), reply().hang().item())
          yield* llm.pushMatch(on("cheap-model"), reply().text("done").stop().item())
        }),
      )
      expect(models(bodies)).toEqual(["test-model", "test-model", "cheap-model"])
      expect(assistants.at(-1)!.finish).toBe("stop")
    }),
  60_000,
)

it.instance(
  "a long outage leaves the model cooling, on disk, for later runs",
  () =>
    Effect.gen(function* () {
      yield* project(undefined, undefined, { failover: { models: ["test/cheap-model"] } })
      const first = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(
            on("test-model"),
            httpError(429, { error: { message: "rate limited" } }, { "retry-after": "3600" }),
          )
          yield* llm.pushMatch(on("cheap-model"), reply().text("done").stop().item())
        }),
      )
      expect(models(first.bodies)).toEqual(["test-model", "cheap-model"])
      expect(FailoverAvailability.available("test/test-model")).toBe(false)
    }),
  30_000,
)
