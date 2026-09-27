/**
 * Failover in the session loop (olaya-provider-failover, gate F3): a step whose provider fails in a
 * way retrying won't fix continues on the next model in the chain, with the whole history, and a
 * session with no chain behaves exactly as before.
 */
import { afterEach, expect } from "bun:test"
import { Effect } from "effect"
import { FailoverAvailability } from "../../src/failover/availability"
import { Todo } from "../../src/session/todo"
import { httpError, reply, TestLLMServer } from "../lib/llm-server"
import { it, project, run } from "../lib/session-loop"

// the availability store is process-wide: each test starts with every model healthy
afterEach(() => FailoverAvailability.clear())

const chain = { failover: { models: ["test/cheap-model"] } }
const textItem = (value: string) => reply().text(value).stop().item()
const toolItem = (name: string, input: unknown) => reply().tool(name, input).item()
const httpErrorItem = httpError
const on = (model: string) => (hit: { body: Record<string, unknown> }) => hit.body.model === model
const quota = { error: { code: "insufficient_quota", message: "You exceeded your current quota" } }
type Body = { model?: string; messages?: unknown[] }

it.instance(
  "without a chain, a failing step stops the session as before",
  () =>
    Effect.gen(function* () {
      yield* project()
      const { assistants, bodies } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.error(401, { error: { message: "invalid api key" } })
        }),
      )
      expect(bodies).toHaveLength(1)
      expect(assistants).toHaveLength(1)
      expect(assistants[0]!.error).toBeDefined()
    }),
  30_000,
)

it.instance(
  "when one model's quota runs out, the task continues on the next with the whole history",
  () =>
    Effect.gen(function* () {
      yield* project(undefined, undefined, chain)
      const { assistants, bodies } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(on("test-model"), httpErrorItem(429, quota))
          yield* llm.pushMatch(on("cheap-model"), textItem("done on the fallback"))
        }),
      )
      const sent = bodies as unknown as Body[]
      expect(sent.map((b) => b.model)).toEqual(["test-model", "cheap-model"])
      expect(JSON.stringify(sent[1]!.messages)).toContain("fix the failing test") // the task went with it
      expect(assistants.map((a) => String(a.modelID))).toEqual(["test-model", "cheap-model"])
      expect(assistants[1]!.error).toBeUndefined()
      expect(assistants[1]!.finish).toBe("stop")
      expect(FailoverAvailability.get("test/test-model")?.state).toBe("disabled")
    }),
  30_000,
)

it.instance(
  "a long retry-after moves on at once instead of waiting",
  () =>
    Effect.gen(function* () {
      yield* project(undefined, undefined, chain)
      const started = Date.now()
      const { bodies } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(
            on("test-model"),
            httpErrorItem(429, { error: { message: "rate limited" } }, { "retry-after": "3600" }),
          )
          yield* llm.pushMatch(on("cheap-model"), textItem("done"))
        }),
      )
      expect((bodies as unknown as Body[]).map((b) => b.model)).toEqual(["test-model", "cheap-model"])
      expect(Date.now() - started).toBeLessThan(10_000)
      const entry = FailoverAvailability.get("test/test-model")!
      expect(entry.state).toBe("cooling")
      expect(entry.until).toBeGreaterThan(Date.now() + 3_500_000)
    }),
  30_000,
)

it.instance(
  "an overloaded model gets two quick retries, then the next model takes over",
  () =>
    Effect.gen(function* () {
      yield* project(undefined, undefined, chain)
      const overloaded = { type: "error", error: { type: "overloaded_error", message: "Overloaded" } }
      const { bodies, assistants } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          for (let i = 0; i < 3; i++) yield* llm.pushMatch(on("test-model"), httpErrorItem(529, overloaded))
          yield* llm.pushMatch(on("cheap-model"), textItem("done"))
        }),
      )
      expect((bodies as unknown as Body[]).map((b) => b.model)).toEqual([
        "test-model",
        "test-model",
        "test-model",
        "cheap-model",
      ])
      expect(assistants.at(-1)!.finish).toBe("stop")
    }),
  60_000,
)

it.instance(
  "tool results from before the failure carry over, and no tool runs twice",
  () =>
    Effect.gen(function* () {
      yield* project(undefined, undefined, chain)
      const { bodies, messages } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(
            on("test-model"),
            toolItem("todowrite", {
              todos: [{ content: "reproduce the bug", status: "completed", priority: "high", id: "1" }],
            }),
          )
          yield* llm.pushMatch(on("test-model"), httpErrorItem(429, quota))
          yield* llm.pushMatch(on("cheap-model"), textItem("done"))
        }),
      )
      const sent = bodies as unknown as Body[]
      expect(sent.map((b) => b.model)).toEqual(["test-model", "test-model", "cheap-model"])
      expect(JSON.stringify(sent[2]!.messages)).toContain("reproduce the bug") // the fallback sees the tool result
      const todo = yield* Todo.Service
      expect(yield* todo.get(messages[0]!.info.sessionID)).toHaveLength(1)
    }),
  30_000,
)

it.instance(
  "when every model in the chain is out for good, the run ends with the reason",
  () =>
    Effect.gen(function* () {
      yield* project(undefined, undefined, chain)
      const { bodies, assistants } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(on("test-model"), httpErrorItem(401, { error: { message: "invalid api key" } }))
          yield* llm.pushMatch(on("cheap-model"), httpErrorItem(401, { error: { message: "invalid api key" } }))
        }),
      )
      expect((bodies as unknown as Body[]).map((b) => b.model)).toEqual(["test-model", "cheap-model"])
      expect(assistants.every((a) => a.error)).toBe(true)
    }),
  30_000,
)

it.instance(
  "a model still cooling from an earlier failure is skipped from the first step",
  () =>
    Effect.gen(function* () {
      yield* project(undefined, undefined, chain)
      FailoverAvailability.mark("test/test-model", {
        action: "switch-until",
        until: Date.now() + 3_600_000,
        reason: "usage limit reached",
      })
      const { bodies } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(on("cheap-model"), textItem("done"))
        }),
      )
      expect((bodies as unknown as Body[]).map((b) => b.model)).toEqual(["cheap-model"])
    }),
  30_000,
)

it.instance(
  "with failover on and no fallbacks, a usage limit is waited out on the same model",
  () =>
    Effect.gen(function* () {
      yield* project(undefined, undefined, { failover: {} })
      const limit = {
        error: {
          type: "usage_limit_reached",
          message: "You've hit your usage limit",
          resets_at: Math.ceil(Date.now() / 1000) + 2,
        },
      }
      const started = Date.now()
      const { bodies, assistants } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(on("test-model"), httpErrorItem(429, limit))
          yield* llm.pushMatch(on("test-model"), textItem("done after the reset"))
        }),
      )
      expect((bodies as unknown as Body[]).map((b) => b.model)).toEqual(["test-model", "test-model"])
      expect(Date.now() - started).toBeGreaterThan(1_000) // it waited for the reset
      expect(assistants.at(-1)!.finish).toBe("stop")
      expect(assistants.at(-1)!.error).toBeUndefined()
      expect(FailoverAvailability.get("test/test-model")).toBeUndefined() // the success cleared it
    }),
  30_000,
)

it.instance(
  "a tool that finished inside the failing step is kept, and the next model doesn't run it again",
  () =>
    Effect.gen(function* () {
      yield* project(undefined, undefined, chain)
      const { bodies, messages } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          // the stream breaks after the tool call, in the same step; the retry then hits the quota
          yield* llm.pushMatch(
            on("test-model"),
            reply()
              .tool("todowrite", {
                todos: [{ content: "reproduce the bug", status: "completed", priority: "high", id: "1" }],
              })
              .streamError("connection reset")
              .item(),
          )
          yield* llm.pushMatch(on("test-model"), httpErrorItem(429, quota))
          yield* llm.pushMatch(on("cheap-model"), textItem("done"))
        }),
      )
      const sent = bodies as unknown as Body[]
      expect(sent.map((b) => b.model)).toEqual(["test-model", "test-model", "cheap-model"])
      expect(JSON.stringify(sent.at(-1)!.messages)).toContain("reproduce the bug") // the fallback sees the result
      const todo = yield* Todo.Service
      expect(yield* todo.get(messages[0]!.info.sessionID)).toHaveLength(1)
      const calls = messages.flatMap((m) => m.parts.filter((p) => p.type === "tool" && p.tool === "todowrite"))
      expect(calls).toHaveLength(1)
    }),
  60_000,
)
