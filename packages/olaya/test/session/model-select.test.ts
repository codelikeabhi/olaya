/**
 * The model-selection seam (experimental.model.select) and per-step usage records
 * (experimental.step.usage), driven through the real session loop, a scripted LLM server and a
 * real plugin file. Gate G2 of the Laya routing work.
 */
import { expect } from "bun:test"
import { Effect } from "effect"
import fs from "fs/promises"
import path from "path"
import { TestInstance } from "../fixture/fixture"
import { TestLLMServer } from "../lib/llm-server"
import { it, project, run } from "../lib/session-loop"

/** The model calls a tool, then answers. */
const twoSteps = Effect.gen(function* () {
  const llm = yield* TestLLMServer
  yield* llm.tool("todowrite", { todos: [{ content: "reproduce", status: "in_progress", priority: "high", id: "1" }] })
  yield* llm.text("done", { usage: { input: 321, output: 12 } })
})

it.instance(
  "without a plugin every step runs on the harness's model",
  () =>
    Effect.gen(function* () {
      yield* project()
      const { assistants, bodies } = yield* run(twoSteps)
      expect(assistants.map((a) => String(a.modelID))).toEqual(["test-model", "test-model"])
      expect(bodies.map((b) => b.model)).toEqual(["test-model", "test-model"])
    }),
  30_000,
)

it.instance(
  "a plugin that leaves the choice unset changes nothing",
  () =>
    Effect.gen(function* () {
      yield* project(
        ["export default async () => ({", '  "experimental.model.select": async () => {},', "})", ""].join("\n"),
      )
      const { assistants, bodies } = yield* run(twoSteps)
      expect(assistants.map((a) => String(a.modelID))).toEqual(["test-model", "test-model"])
      expect(bodies.map((b) => b.model)).toEqual(["test-model", "test-model"])
    }),
  30_000,
)

it.instance(
  "a plugin can put the first step on another model",
  () =>
    Effect.gen(function* () {
      yield* project(
        [
          "export default async () => ({",
          '  "experimental.model.select": async (input, output) => {',
          '    if (input.point === "start") output.model = { providerID: "test", modelID: "cheap-model" }',
          "  },",
          "})",
          "",
        ].join("\n"),
      )
      const { assistants, bodies } = yield* run(twoSteps)
      expect(assistants.map((a) => String(a.modelID))).toEqual(["cheap-model", "test-model"])
      expect(bodies.map((b) => b.model)).toEqual(["cheap-model", "test-model"])
    }),
  30_000,
)

it.instance(
  "an unknown model from a plugin is ignored, not an error",
  () =>
    Effect.gen(function* () {
      yield* project(
        [
          "export default async () => ({",
          '  "experimental.model.select": async (_input, output) => {',
          '    output.model = { providerID: "test", modelID: "no-such-model" }',
          "  },",
          "})",
          "",
        ].join("\n"),
      )
      const { assistants, bodies } = yield* run(twoSteps)
      expect(assistants.map((a) => String(a.modelID))).toEqual(["test-model", "test-model"])
      expect(assistants.every((a) => !a.error)).toBe(true)
      expect(bodies.map((b) => b.model)).toEqual(["test-model", "test-model"])
    }),
  30_000,
)

it.instance(
  "every step is reported once, with its message's usage, and selection sees the previous step",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* TestInstance
      const log = path.join(directory, "records.jsonl")
      yield* project(
        [
          'import { appendFileSync } from "fs"',
          "export default async () => ({",
          '  "experimental.step.usage": async (input) => {',
          `    appendFileSync(${JSON.stringify(log)}, JSON.stringify({ kind: "usage", ...input }) + "\\n")`,
          "  },",
          '  "experimental.model.select": async (input) => {',
          `    appendFileSync(${JSON.stringify(log)}, JSON.stringify({ kind: "select", step: input.step, point: input.point, previous: input.usage?.step ?? null }) + "\\n")`,
          "  },",
          "})",
          "",
        ].join("\n"),
      )
      const { assistants } = yield* run(twoSteps)
      const lines = (yield* Effect.promise(() => fs.readFile(log, "utf8")))
        .trim()
        .split("\n")
        .map((l) => JSON.parse(l))
      const usage = lines.filter((l) => l.kind === "usage")
      const select = lines.filter((l) => l.kind === "select")

      expect(usage.map((u) => u.step)).toEqual([1, 2])
      expect(usage.map((u) => u.messageID)).toEqual(assistants.map((a) => a.id))
      usage.forEach((u, i) => {
        expect(u.modelID).toBe(assistants[i]!.modelID)
        expect(u.tokens.input).toBe(assistants[i]!.tokens.input)
        expect(u.tokens.output).toBe(assistants[i]!.tokens.output)
        expect(u.tokens.cacheRead).toBe(assistants[i]!.tokens.cache.read)
        expect("cost" in u).toBe(false) // the test models report no cost: absent, not zero
      })
      expect(usage[1].tokens.input).toBe(321)
      expect(select).toEqual([
        { kind: "select", step: 1, point: "start", previous: null },
        { kind: "select", step: 2, point: "step", previous: 1 },
      ])
    }),
  30_000,
)

const cheapFirst = [
  "export default async () => ({",
  '  "experimental.model.select": async (input, output) => {',
  '    if (input.point === "start") output.model = { providerID: "test", modelID: "cheap-model" }',
  "  },",
  "})",
  "",
].join("\n")

it.instance(
  "a model outside the user's routing pool is never chosen",
  () =>
    Effect.gen(function* () {
      yield* project(cheapFirst, { models: ["test/test-model"] })
      const { assistants, bodies } = yield* run(twoSteps)
      expect(assistants.map((a) => String(a.modelID))).toEqual(["test-model", "test-model"])
      expect(bodies.map((b) => b.model)).toEqual(["test-model", "test-model"])
    }),
  30_000,
)

it.instance(
  "a model inside the routing pool can be chosen",
  () =>
    Effect.gen(function* () {
      yield* project(cheapFirst, { models: ["test/test-model", "test/cheap-model"] })
      const { assistants } = yield* run(twoSteps)
      expect(assistants.map((a) => String(a.modelID))).toEqual(["cheap-model", "test-model"])
    }),
  30_000,
)
