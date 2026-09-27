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

const tool = (name: string, input: unknown, error?: string) => ({
  type: "tool",
  tool: name,
  state: error ? { status: "error", input, error } : { status: "completed", input, output: "ok" },
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
