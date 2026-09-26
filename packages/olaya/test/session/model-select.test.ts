/**
 * The model-selection seam (experimental.model.select) and per-step usage records
 * (experimental.step.usage), driven through the real session loop, a scripted LLM server and a
 * real plugin file. Gate G2 of the Laya routing work.
 */
import { SessionV1 } from "@olaya/core/v1/session"
import { Database } from "@olaya/core/database/database"
import { LayerNode } from "@olaya/core/effect/layer-node"
import { SessionProjector } from "@olaya/core/session/projector"
import { EventV2Bridge } from "@/event-v2-bridge"
import { expect } from "bun:test"
import { Effect, Layer } from "effect"
import fs from "fs/promises"
import path from "path"
import { pathToFileURL } from "url"
import { Agent as AgentSvc } from "../../src/agent/agent"
import { BackgroundJob } from "@/background/job"
import { Command } from "../../src/command"
import { Config } from "@/config/config"
import { LSP } from "@/lsp/lsp"
import { MCP } from "../../src/mcp"
import { Permission } from "../../src/permission"
import { Plugin } from "../../src/plugin"
import { Provider as ProviderSvc } from "@/provider/provider"
import { Env } from "../../src/env"
import { Git } from "../../src/git"
import { Image } from "../../src/image/image"
import { Question } from "../../src/question"
import { Todo } from "../../src/session/todo"
import { Session } from "@/session/session"
import { LLM } from "../../src/session/llm"
import { MessageV2 } from "../../src/session/message-v2"
import { FSUtil } from "@olaya/core/fs-util"
import { SessionCompaction } from "../../src/session/compaction"
import { SessionSummary } from "../../src/session/summary"
import { Instruction } from "../../src/session/instruction"
import { SessionProcessor } from "../../src/session/processor"
import { SessionPrompt } from "../../src/session/prompt"
import { SessionRevert } from "../../src/session/revert"
import { SessionRunState } from "../../src/session/run-state"
import { SessionStatus } from "../../src/session/status"
import { Skill } from "../../src/skill"
import { SystemPrompt } from "../../src/session/system"
import { Snapshot } from "../../src/snapshot"
import { ToolRegistry } from "@/tool/registry"
import { Truncate } from "@/tool/truncate"
import { CrossSpawnSpawner } from "@olaya/core/cross-spawn-spawner"
import { Ripgrep } from "@olaya/core/ripgrep"
import { Format } from "../../src/format"
import { TestInstance } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { TestLLMServer } from "../lib/llm-server"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { ProviderV2 } from "@olaya/core/provider"
import { ModelV2 } from "@olaya/core/model"

const summary = Layer.succeed(
  SessionSummary.Service,
  SessionSummary.Service.of({
    summarize: () => Effect.void,
    diff: () => Effect.succeed([]),
    computeDiff: () => Effect.succeed([]),
  }),
)
const mcp = Layer.succeed(
  MCP.Service,
  MCP.Service.of({
    status: () => Effect.succeed({}),
    clients: () => Effect.succeed({}),
    instructions: () => Effect.succeed([]),
    tools: () => Effect.succeed({}),
    prompts: () => Effect.succeed({}),
    resources: () => Effect.succeed({}),
    resourceTemplates: () => Effect.succeed({}),
    add: () => Effect.succeed({ status: { status: "disabled" as const } }),
    connect: () => Effect.void,
    disconnect: () => Effect.void,
    getPrompt: () => Effect.succeed(undefined),
    readResource: () => Effect.succeed(undefined),
    startAuth: () => Effect.die("unexpected"),
    authenticate: () => Effect.die("unexpected"),
    finishAuth: () => Effect.die("unexpected"),
    removeAuth: () => Effect.void,
    supportsOAuth: () => Effect.succeed(false),
    hasStoredTokens: () => Effect.succeed(false),
    getAuthStatus: () => Effect.succeed("not_authenticated" as const),
  }),
)
const lsp = Layer.succeed(
  LSP.Service,
  LSP.Service.of({
    init: () => Effect.void,
    status: () => Effect.succeed([]),
    hasClients: () => Effect.succeed(false),
    touchFile: () => Effect.void,
    diagnostics: () => Effect.succeed({}),
    hover: () => Effect.succeed(undefined),
    definition: () => Effect.succeed([]),
    references: () => Effect.succeed([]),
    implementation: () => Effect.succeed([]),
    documentSymbol: () => Effect.succeed([]),
    workspaceSymbol: () => Effect.succeed([]),
    prepareCallHierarchy: () => Effect.succeed([]),
    incomingCalls: () => Effect.succeed([]),
    outgoingCalls: () => Effect.succeed([]),
  }),
)

const root = LayerNode.group([
  LayerNode.group([
    SessionPrompt.node,
    Session.node,
    SessionProjector.node,
    MessageV2.node,
    Snapshot.node,
    LLM.node,
    Env.node,
    AgentSvc.node,
    Command.node,
    Permission.node,
    Plugin.node,
    Config.node,
    ProviderSvc.node,
    LSP.node,
    MCP.node,
    FSUtil.node,
    BackgroundJob.node,
    SessionStatus.node,
    SessionRunState.node,
    Database.node,
    EventV2Bridge.node,
    Question.node,
    Todo.node,
    ToolRegistry.node,
    Skill.node,
    Git.node,
    Ripgrep.node,
    Format.node,
    Truncate.node,
    SessionProcessor.node,
    Image.node,
    SessionCompaction.node,
    SessionRevert.node,
    Instruction.node,
    SystemPrompt.node,
    CrossSpawnSpawner.node,
    RuntimeFlags.node,
  ]),
  LayerNode.make({ service: TestLLMServer, layer: TestLLMServer.layer, deps: [] }),
])
const it = testEffect(
  LayerNode.compile(root, [
    [SessionSummary.node, summary],
    [LSP.node, lsp],
    [MCP.node, mcp],
    [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true })],
  ]),
)

const model = (id: string) => ({
  id,
  name: id,
  attachment: false,
  reasoning: false,
  temperature: false,
  tool_call: true,
  release_date: "2025-01-01",
  limit: { context: 100000, output: 10000 },
  cost: { input: 0, output: 0 },
  options: {},
})

/** A project whose "test" provider has two models, and optionally a plugin written from `source`. */
const project = Effect.fn("test.project")(function* (source?: string, routing?: { models: string[] }) {
  const { directory } = yield* TestInstance
  const llm = yield* TestLLMServer
  const plugin = path.join(directory, "router.ts")
  if (source) yield* Effect.promise(() => Bun.write(plugin, source))
  const config = {
    $schema: "https://opencode.ai/config.json",
    ...(source ? { plugin: [pathToFileURL(plugin).href] } : {}),
    ...(routing ? { routing } : {}),
    provider: {
      test: {
        name: "Test",
        id: "test",
        env: [],
        npm: "@ai-sdk/openai-compatible",
        models: { "test-model": model("test-model"), "cheap-model": model("cheap-model") },
        options: { apiKey: "test-key", baseURL: llm.url },
      },
    },
  }
  yield* Effect.promise(() => Bun.write(path.join(directory, "olaya.json"), JSON.stringify(config)))
  return { directory, llm }
})

/** One session: the user asks, `script` queues the model's replies, the loop runs to the end. */
const run = Effect.fn("test.run")(function* (script: Effect.Effect<void, never, TestLLMServer>) {
  const llm = yield* TestLLMServer
  const prompt = yield* SessionPrompt.Service
  const sessions = yield* Session.Service
  const chat = yield* sessions.create({
    title: "Route",
    permission: [{ permission: "*", pattern: "*", action: "allow" }],
  })
  yield* prompt.prompt({
    sessionID: chat.id,
    agent: "build",
    model: { providerID: ProviderV2.ID.make("test"), modelID: ModelV2.ID.make("test-model") },
    noReply: true,
    parts: [{ type: "text", text: "fix the failing test" }],
  })
  yield* script
  yield* prompt.loop({ sessionID: chat.id })
  const msgs = yield* sessions.messages({ sessionID: chat.id })
  const assistants = msgs.filter((m) => m.info.role === "assistant").map((m) => m.info as SessionV1.Assistant)
  const bodies = (yield* llm.inputs) as { model?: string }[]
  return { assistants, bodies }
})

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
