/**
 * Continuity across providers (olaya-provider-failover, gate F4): history made valid for the next
 * provider, and the context fitted to a smaller window without calling the model that failed.
 */
import { afterEach, describe, expect, test } from "bun:test"
import { Effect } from "effect"
import { FailoverAvailability } from "../../src/failover/availability"
import { ProviderTransform } from "../../src/provider/transform"
import { httpError, reply, TestLLMServer } from "../lib/llm-server"
import { it, model, project, run } from "../lib/session-loop"
import { SessionPrompt } from "../../src/session/prompt"
import { Session } from "../../src/session/session"
import { SessionV1 as V1 } from "@olaya/core/v1/session"
import { ProviderV2 } from "@olaya/core/provider"
import { ModelV2 } from "@olaya/core/model"

afterEach(() => FailoverAvailability.clear())

const target = (providerID: string, id: string) =>
  ({
    id: `${providerID}/${id}`,
    providerID,
    api: { id, url: "", npm: "@ai-sdk/openai-compatible" },
    capabilities: { input: { text: true, image: false } },
  }) as any

const history = (ids: string[]) =>
  ids.flatMap((toolCallId) => [
    { role: "assistant", content: [{ type: "tool-call", toolCallId, toolName: "read", input: {} }] },
    {
      role: "tool",
      content: [{ type: "tool-result", toolCallId, toolName: "read", output: { type: "text", value: "x" } }],
    },
  ]) as any

const idsOf = (msgs: any[]) =>
  msgs.flatMap((msg) =>
    (Array.isArray(msg.content) ? msg.content : [])
      .filter((p: any) => p.type === "tool-call" || p.type === "tool-result")
      .map((p: any) => p.toolCallId),
  )

describe("history made valid for the next provider", () => {
  test("Kimi gets its own functions.<name>:<n> IDs, and each call still pairs with its result", () => {
    const out = ProviderTransform.message(history(["toolu_01AAA", "call_xyz"]), target("moonshotai", "kimi-k2"), {})
    expect(idsOf(out)).toEqual(["functions.read:0", "functions.read:0", "functions.read:1", "functions.read:1"])
  })

  test("Kimi's own IDs are left alone", () => {
    const out = ProviderTransform.message(history(["functions.read:0"]), target("moonshotai", "kimi-k2"), {})
    expect(idsOf(out)).toEqual(["functions.read:0", "functions.read:0"])
  })

  test("an ID too long for other providers is shortened, the same way for call and result", () => {
    const long = `function-call-${"9".repeat(40)}`
    const out = ProviderTransform.message(history([long, "call_short"]), target("openai", "gpt-5"), {})
    const ids = idsOf(out)
    expect(ids[0]).toMatch(/^call_[a-zA-Z0-9]{24}$/)
    expect(ids[1]).toBe(ids[0])
    expect(ids.slice(2)).toEqual(["call_short", "call_short"])
  })

  test("an image a text-only fallback can't read is noted, and only the latest message asks for the user", () => {
    const image = { type: "image", image: "data:image/png;base64,iVBORw0KGgo=" }
    const out = ProviderTransform.message(
      [
        { role: "user", content: [{ type: "text", text: "here is the screenshot" }, image] },
        { role: "assistant", content: [{ type: "text", text: "I see the error dialog." }] },
        { role: "user", content: [{ type: "text", text: "and this one" }, image] },
      ] as any,
      target("ollama", "qwen3-8b"),
      {},
    ) as any[]
    expect(JSON.stringify(out[0].content)).toContain("[image omitted: this model does not support image input]")
    expect(JSON.stringify(out[0].content)).not.toContain("Inform the user")
    expect(JSON.stringify(out[2].content)).toContain("Inform the user")
  })
})

const on = (m: string) => (hit: { body: Record<string, unknown> }) => hit.body.model === m
const quota = { error: { code: "insufficient_quota", message: "You exceeded your current quota" } }
type Body = { model?: string; messages?: unknown[] }

it.instance(
  "moving to a smaller window compacts on the new model, never calling the one that failed",
  () =>
    Effect.gen(function* () {
      // cheap-model's usable window is 2,500 tokens; the first step used 5,000
      yield* project(
        undefined,
        undefined,
        { failover: { models: ["test/cheap-model"] } },
        {
          "cheap-model": { ...model("cheap-model"), limit: { context: 3000, output: 500 } },
        },
      )
      const { bodies, assistants } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(
            on("test-model"),
            reply()
              .tool("todowrite", {
                todos: [{ content: "note the constraint", status: "pending", priority: "high", id: "1" }],
              })
              .usage({ input: 5000, output: 10 })
              .item(),
          )
          yield* llm.pushMatch(on("test-model"), httpError(429, quota))
          yield* llm.pushMatch(
            on("cheap-model"),
            reply().text("Summary: fixing the failing test; todo noted.").stop().item(),
          )
          yield* llm.pushMatch(on("cheap-model"), reply().text("done").stop().item())
        }),
      )
      const sent = bodies as unknown as Body[]
      expect(sent.map((b) => b.model)).toEqual(["test-model", "test-model", "cheap-model", "cheap-model"])
      expect(JSON.stringify(sent[2]!.messages)).toContain("summary") // the compaction request, on the new model
      expect(assistants.some((a) => a.summary && String(a.modelID) === "cheap-model")).toBe(true)
      expect(assistants.at(-1)!.finish).toBe("stop")
    }),
  30_000,
)

it.instance(
  "a fallback with no declared window is skipped for one that has one",
  () =>
    Effect.gen(function* () {
      yield* project(
        undefined,
        undefined,
        { failover: { models: ["test/windowless", "test/cheap-model"] } },
        {
          windowless: { ...model("windowless"), limit: { context: 0, output: 0 } },
        },
      )
      const { bodies } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(on("test-model"), httpError(429, quota))
          yield* llm.pushMatch(on("cheap-model"), reply().text("done").stop().item())
        }),
      )
      expect((bodies as unknown as Body[]).map((b) => b.model)).toEqual(["test-model", "cheap-model"])
      expect(FailoverAvailability.get("test/windowless")?.reason).toContain("no context window")
    }),
  30_000,
)

it.instance(
  "a fallback that can't call tools is skipped",
  () =>
    Effect.gen(function* () {
      yield* project(
        undefined,
        undefined,
        { failover: { models: ["test/chat-only", "test/cheap-model"] } },
        {
          "chat-only": { ...model("chat-only"), tool_call: false },
        },
      )
      const { bodies } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(on("test-model"), httpError(429, quota))
          yield* llm.pushMatch(on("cheap-model"), reply().text("done").stop().item())
        }),
      )
      expect((bodies as unknown as Body[]).map((b) => b.model)).toEqual(["test-model", "cheap-model"])
      expect(FailoverAvailability.get("test/chat-only")?.reason).toBe("does not support tool calls")
    }),
  30_000,
)

it.instance(
  "when the compaction agent's own model fails, compaction runs on the session's model, which stays",
  () =>
    Effect.gen(function* () {
      // test-model's usable window is 2,500 tokens; the first step used 5,000
      yield* project(
        undefined,
        undefined,
        { failover: { models: ["test/cheap-model"] }, agent: { compaction: { model: "test/summarizer" } } },
        {
          "test-model": { ...model("test-model"), limit: { context: 3000, output: 500 } },
          summarizer: model("summarizer"),
        },
      )
      const { bodies, assistants } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(
            on("test-model"),
            reply()
              .tool("todowrite", {
                todos: [{ content: "note the constraint", status: "pending", priority: "high", id: "1" }],
              })
              .usage({ input: 5000, output: 10 })
              .item(),
          )
          yield* llm.pushMatch(on("summarizer"), httpError(429, quota))
          yield* llm.pushMatch(
            on("test-model"),
            reply().text("Summary: fixing the failing test; todo noted.").stop().item(),
          )
          yield* llm.pushMatch(on("test-model"), reply().text("done").stop().item())
        }),
      )
      const sent = bodies as unknown as Body[]
      expect(sent.map((b) => b.model)).toEqual(["test-model", "summarizer", "test-model", "test-model"])
      expect(FailoverAvailability.get("test/summarizer")?.state).toBe("disabled")
      expect(FailoverAvailability.get("test/test-model")).toBeUndefined() // the session's model was never blamed
      expect(assistants.at(-1)!.finish).toBe("stop")
    }),
  30_000,
)

it.instance(
  "when compaction's shared-prefix model fails, compaction moves to the session's model instead of retrying it",
  () =>
    Effect.gen(function* () {
      // routing runs the first step on cheap-model; its window overflows, and compaction shares
      // that step's prefix, so it goes to cheap-model too, which is out of quota
      yield* project(
        [
          "export default async () => ({",
          '  "experimental.model.select": async (input, output) => {',
          '    if (input.point === "start") output.model = { providerID: "test", modelID: "cheap-model" }',
          "  },",
          "})",
          "",
        ].join("\n"),
        undefined,
        { failover: {} },
        {
          "test-model": { ...model("test-model"), limit: { context: 3000, output: 500 } },
          "cheap-model": { ...model("cheap-model"), limit: { context: 3000, output: 500 } },
        },
      )
      const { bodies, assistants } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.pushMatch(
            on("cheap-model"),
            reply()
              .tool("todowrite", { todos: [{ content: "x", status: "pending", priority: "high", id: "1" }] })
              .usage({ input: 5000, output: 10 })
              .item(),
          )
          for (let i = 0; i < 6; i++) yield* llm.pushMatch(on("cheap-model"), httpError(429, quota))
          yield* llm.pushMatch(on("test-model"), reply().text("Summary: todo noted.").stop().item())
          yield* llm.pushMatch(on("test-model"), reply().text("done").stop().item())
        }),
      )
      const models = (bodies as unknown as Body[])
        .filter((b) => !JSON.stringify(b).includes("Generate a title"))
        .map((b) => b.model)
      expect(models).toEqual(["cheap-model", "cheap-model", "test-model", "test-model"])
      expect(FailoverAvailability.get("test/cheap-model")?.state).toBe("disabled")
      expect(assistants.at(-1)!.finish).toBe("stop")
    }),
  30_000,
)

it.instance(
  "after an automatic compaction the run keeps its structured output and system prompt",
  () =>
    Effect.gen(function* () {
      yield* project(
        undefined,
        undefined,
        {},
        { "test-model": { ...model("test-model"), limit: { context: 3000, output: 500 } } },
      )
      const llm = yield* TestLLMServer
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Format",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      yield* prompt.prompt({
        sessionID: chat.id,
        agent: "build",
        model: { providerID: ProviderV2.ID.make("test"), modelID: ModelV2.ID.make("test-model") },
        noReply: true,
        system: "SYSTEM-MARKER-XYZ",
        format: new V1.OutputFormatJsonSchema({ type: "json_schema", schema: { type: "object" }, retryCount: 2 }),
        parts: [{ type: "text", text: "fix the failing test" }],
      })
      yield* llm.push(
        reply()
          .tool("todowrite", { todos: [{ content: "x", status: "pending", priority: "high", id: "1" }] })
          .usage({ input: 5000, output: 10 })
          .item(),
      )
      yield* llm.push(reply().text("Summary: todo noted.").stop().item())
      yield* llm.push(reply().tool("StructuredOutput", {}).item())
      yield* prompt.loop({ sessionID: chat.id })
      const bodies = (
        (yield* llm.inputs) as { messages?: unknown[]; tools?: { function?: { name?: string } }[] }[]
      ).filter((b) => !JSON.stringify(b).includes("Generate a title"))
      const after = bodies[2]! // the step after the summary
      expect(JSON.stringify(after.messages)).toContain("SYSTEM-MARKER-XYZ")
      expect((after.tools ?? []).map((t) => t.function?.name)).toContain("StructuredOutput")
    }),
  30_000,
)
