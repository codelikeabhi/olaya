/**
 * The loop guard: a model repeating the same call, or the same failure, across steps gets one
 * reminder to step back, instead of looping until the step limit.
 */
import { describe, expect, test } from "bun:test"
import { Effect } from "effect"
import type { SessionV1 } from "@olaya/core/v1/session"
import { LoopGuard } from "../../src/session/loop-guard"
import { reply, TestLLMServer } from "../lib/llm-server"
import { it, project, run } from "../lib/session-loop"
import { SessionPrompt } from "../../src/session/prompt"
import { Session } from "../../src/session/session"
import { SessionV1 as V1 } from "@olaya/core/v1/session"
import { ProviderV2 } from "@olaya/core/provider"
import { ModelV2 } from "@olaya/core/model"

const tool = (name: string, input: unknown, error?: string, output = "ok") => ({
  type: "tool",
  tool: name,
  state: error ? { status: "error", input, error } : { status: "completed", input, output },
})
const history = (...parts: unknown[]) =>
  [
    { info: { role: "user" }, parts: [] },
    ...parts.map((part) => ({ info: { role: "assistant" }, parts: [part] })),
  ] as unknown as SessionV1.WithParts[]
const notFound = "Could not find oldString in the file. It must match exactly"

describe("loop guard", () => {
  test("three identical calls in a row get a reminder", () => {
    const read = tool("read", { filePath: "/a.py" })
    expect(LoopGuard.loopNudge(history(read, read, read))).toContain("same `read` call 3 times")
    expect(LoopGuard.loopNudge(history(read, read))).toBeUndefined()
  })

  test("three different edits failing the same way get a reminder that quotes the error", () => {
    const edits = [1, 2, 3].map((n) => tool("edit", { filePath: "/a.py", oldString: `v${n}` }, notFound))
    expect(LoopGuard.loopNudge(history(...edits))).toContain(`failed with the same error: "${notFound}"`)
  })

  test("progress is left alone: different calls that succeed, or different errors", () => {
    expect(
      LoopGuard.loopNudge(history(tool("read", { f: 1 }), tool("edit", { f: 1 }), tool("bash", { c: "pytest" }))),
    ).toBeUndefined()
    expect(
      LoopGuard.loopNudge(
        history(tool("edit", { n: 1 }, "a"), tool("edit", { n: 2 }, "b"), tool("edit", { n: 3 }, "a")),
      ),
    ).toBeUndefined()
  })

  test("polling until the output changes is not a loop", () => {
    const poll = (output: string) => tool("bash", { command: "gh run view 42" }, undefined, output)
    expect(LoopGuard.loopNudge(history(poll("queued"), poll("in_progress"), poll("in_progress")))).toBeUndefined()
    expect(LoopGuard.loopNudge(history(poll("queued"), poll("queued"), poll("queued")))).toBeDefined()
  })

  test("calls a provider failure cut off are not counted as the model repeating itself", () => {
    const cut = {
      type: "tool",
      tool: "bash",
      state: { status: "error", input: {}, error: "Tool execution aborted", metadata: { interrupted: true } },
    }
    expect(LoopGuard.loopNudge(history(cut, cut, cut))).toBeUndefined()
  })

  test("calls before the latest user message don't count, so a reminder starts the count again", () => {
    const read = tool("read", { filePath: "/a.py" })
    const msgs = history(read, read, read)
    msgs.push({ info: { role: "user" }, parts: [] } as unknown as SessionV1.WithParts)
    expect(LoopGuard.loopNudge(msgs)).toBeUndefined()
  })
})

it.instance(
  "in a session, the step after a repeated failing call sees the reminder, once",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* project()
      const missing = `${directory}/missing.py`
      const { bodies, messages } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          for (let i = 0; i < 3; i++) yield* llm.push(reply().tool("read", { filePath: missing }).item())
          yield* llm.push(reply().text("I'll look at the directory instead").stop().item())
        }),
      )
      expect(bodies).toHaveLength(4)
      const sent = (bodies as { messages?: unknown[] }[]).map((b) => JSON.stringify(b.messages))
      expect(sent[2]).not.toContain("times in a row")
      expect(sent[3]).toContain("same `read` call 3 times in a row")
      const reminders = messages.filter((m) =>
        m.parts.some((part) => part.type === "text" && part.metadata?.loop_guard_nudge),
      )
      expect(reminders).toHaveLength(1)
    }),
  30_000,
)

it.instance(
  "the reminder keeps the run's system prompt and structured output",
  () =>
    Effect.gen(function* () {
      const { directory } = yield* project()
      const llm = yield* TestLLMServer
      const prompt = yield* SessionPrompt.Service
      const sessions = yield* Session.Service
      const chat = yield* sessions.create({
        title: "Loop",
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
      })
      yield* prompt.prompt({
        sessionID: chat.id,
        agent: "build",
        model: { providerID: ProviderV2.ID.make("test"), modelID: ModelV2.ID.make("test-model") },
        noReply: true,
        system: "SYSTEM-MARKER-XYZ",
        format: new V1.OutputFormatJsonSchema({
          type: "json_schema",
          schema: { type: "object", properties: { a: { type: "string" } } },
          retryCount: 2,
        }),
        parts: [{ type: "text", text: "fix the failing test" }],
      })
      for (let i = 0; i < 3; i++)
        yield* llm.push(
          reply()
            .tool("read", { filePath: `${directory}/missing.py` })
            .item(),
        )
      yield* llm.push(reply().tool("StructuredOutput", { a: "done" }).item())
      yield* prompt.loop({ sessionID: chat.id })
      const bodies = (
        (yield* llm.inputs) as { messages?: unknown[]; tools?: { function?: { name?: string } }[] }[]
      ).filter((b) => !JSON.stringify(b).includes("Generate a title"))
      const after = bodies[3]!
      expect(JSON.stringify(after.messages)).toContain("times in a row") // the reminder is there
      expect(JSON.stringify(after.messages)).toContain("SYSTEM-MARKER-XYZ")
      expect((after.tools ?? []).map((t) => t.function?.name)).toContain("StructuredOutput")
    }),
  30_000,
)
