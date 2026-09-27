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

  test("re-reading between the same failing edits is still a loop", () => {
    const read = tool("read", { filePath: "/a.py" })
    const edit = (n: number) => tool("edit", { filePath: "/a.py", oldString: `v${n}` }, notFound)
    expect(LoopGuard.loopNudge(history(read, edit(1), read, edit(2), read, edit(3)))).toContain(
      "`edit` calls all failed",
    )
    expect(LoopGuard.loopNudge(history(read, edit(1), read, edit(2), read))).toBeUndefined()
  })

  test("a command rerun unchanged is a loop even when its timing and saved-output name differ", () => {
    const run = (secs: string, id: string) =>
      tool("bash", { command: "bun test" }, undefined, `3 fail in ${secs}s\nFull output saved to: /tmp/tool_${id}`)
    expect(
      LoopGuard.loopNudge(history(run("1.02", "0e04a1b2c3"), run("0.98", "0e04d4e5f6"), run("1.10", "0e04a7b8c9"))),
    ).toContain("same `bash` call")
  })

  test("re-delegating the same task is a loop even though each run gets a new session ID", () => {
    // Track E, 2026-09-27: 207 identical `task` calls, each subagent answering "I'll start by..."
    const delegate = (id: string) =>
      tool(
        "task",
        { subagent_type: "explore", prompt: "Refactor markdown.py" },
        undefined,
        `<task id="${id}" state="completed"> <task_result> I'll start by analyzing it. </task_result>`,
      )
    expect(
      LoopGuard.loopNudge(
        history(
          delegate("ses_f1f38ba79ffexcbAIx6Kqn4aPq"),
          delegate("ses_f1f3862c0ffeeRFSAZu13rIOuC"),
          delegate("ses_f1f384dbeffegiyAcrMXrG58wV"),
        ),
      ),
    ).toContain("same `task` call")
  })

  test("the same failure on different files is not a repeat", () => {
    const edit = (file: string) => tool("edit", { filePath: file, oldString: "x" }, notFound)
    const write = (file: string) => tool("write", { filePath: file })
    expect(
      LoopGuard.loopNudge(history(edit("/a.py"), write("/a.py"), edit("/b.py"), write("/b.py"), edit("/c.py"))),
    ).toBeUndefined()
  })

  test("a metric that keeps improving is progress, not a loop", () => {
    const poll = (loss: string) => tool("bash", { command: "tail -1 train.log" }, undefined, `loss ${loss} acc 81.2%`)
    expect(LoopGuard.loopNudge(history(poll("0.4121"), poll("0.3977"), poll("0.3810")))).toBeUndefined()
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
          yield* llm.push(reply().text("The file does not exist, so there is nothing to fix.").stop().item())
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

describe("ending on an announced step", () => {
  test("endings from the benchmark runs that announced a step and stopped", () => {
    for (const text of [
      "I'll need to read the current state of the file to proceed accurately. Let me check the file.",
      "Next, I'll work on fixing the `reversed` method to properly reverse the list. Let me make that change now.",
      "I'll try a different approach. Let's first read the current content of luhn.py.",
      "I will proceed to run the tests to verify the implementation. Let me execute the tests using `python -m pytest -q`.",
    ])
      expect(LoopGuard.announcedAction(text)).toBe(true)
  })

  test("a sign-off, a question or a plain summary is left alone", () => {
    for (const text of [
      "These changes should fix the errors. Let me know if you need further adjustments.",
      "The tests pass. Shall I also update the README?",
      "Fixed the off-by-one in mean(); all 12 tests pass.",
      "",
      // from the third review: answers and hand-backs that are not announced steps
      "Now I have the full picture. Findings:\n\n- `src/auth.ts:42`: token not refreshed\n- `src/db.ts:10`: pool never closed",
      "Now I've fixed the bug and all 12 tests pass.",
      "I'll leave running it against production to you.",
      "I'll wait for your go-ahead before changing anything.",
      "Let's summarize: the cache key was wrong.",
      "I will not touch the lockfile without approval.",
      "Next, I recommend adding a regression test.",
      "The migration is ready. **Shall I proceed?**",
    ])
      expect(LoopGuard.announcedAction(text)).toBe(false)
  })
})

it.instance(
  "a run that ends on an announced step is asked to take it, once",
  () =>
    Effect.gen(function* () {
      yield* project()
      const { bodies, messages } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.push(reply().text("I need to see the code first. Let me check the file.").stop().item())
          yield* llm.push(reply().text("Done: the fix is in and the tests pass.").stop().item())
        }),
      )
      const sent = (bodies as { messages?: unknown[] }[]).filter((b) => !JSON.stringify(b).includes("Generate a title"))
      expect(sent).toHaveLength(2)
      expect(JSON.stringify(sent[1]!.messages)).toContain("you called no tool")
      expect(messages.filter((m) => m.parts.some((p) => p.type === "text" && p.metadata?.action_nudge))).toHaveLength(1)
    }),
  30_000,
)

it.instance(
  "a run that ends with a sign-off is not nudged",
  () =>
    Effect.gen(function* () {
      yield* project()
      const { bodies } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.push(reply().text("Fixed it. Let me know if you need anything else.").stop().item())
        }),
      )
      expect(
        (bodies as { messages?: unknown[] }[]).filter((b) => !JSON.stringify(b).includes("Generate a title")),
      ).toHaveLength(1)
    }),
  30_000,
)
