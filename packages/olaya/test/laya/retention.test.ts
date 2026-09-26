/**
 * Compaction fixes that come before Laya retention (laya-context-retention tasks 1.1–1.3), driven
 * through the real session loop: a step whose usage passes the overflow limit triggers an automatic
 * compaction, and the summary request is read back from the scripted LLM server.
 */
import { expect } from "bun:test"
import { Effect } from "effect"
import { Todo } from "../../src/session/todo"
import { TestLLMServer } from "../lib/llm-server"
import { it, project, run } from "../lib/session-loop"

type Body = { messages: { role: string; content: unknown }[]; tools?: unknown }

// The test models have a 100k context and 10k output, so 95k input overflows the usable window.
const overflowing = { usage: { input: 95_000, output: 10 } }

it.instance(
  "compaction sees the tail of a long tool output, where the error is",
  () =>
    Effect.gen(function* () {
      yield* project()
      const error = "FAILED tests/test_bank.py::test_withdraw - AssertionError"
      const { bodies } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.tool("todowrite", {
            todos: [{ content: `${"x".repeat(3_000)} ${error}`, status: "in_progress", priority: "high", id: "1" }],
          })
          yield* llm.text("working", overflowing)
          yield* llm.text("The withdraw test fails.")
          yield* llm.text("done")
        }),
      )
      const summary = JSON.stringify(bodies[2])
      expect(summary).toContain("characters truncated")
      expect(summary).toContain(error)
    }),
  30_000,
)

it.instance(
  "the summary request repeats the session's system prompt and tools",
  () =>
    Effect.gen(function* () {
      yield* project()
      const { bodies } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.text("working", overflowing)
          yield* llm.text("The work so far.")
          yield* llm.text("done")
        }),
      )
      const [step, summary] = bodies as unknown as Body[]
      const system = (body: Body) => body.messages.filter((m) => m.role === "system")
      expect(JSON.stringify(summary.messages)).toContain("Do not call tools")
      expect(step.tools).toBeDefined()
      expect(JSON.stringify(summary.tools)).toBe(JSON.stringify(step.tools))
      expect(JSON.stringify(system(summary))).toBe(JSON.stringify(system(step)))
    }),
  30_000,
)

it.instance(
  "a tool call during compaction stops it without running the tool",
  () =>
    Effect.gen(function* () {
      yield* project()
      const { messages } = yield* run(
        Effect.gen(function* () {
          const llm = yield* TestLLMServer
          yield* llm.text("working", overflowing)
          yield* llm.tool("todowrite", { todos: [{ content: "sneaky", status: "pending", priority: "high", id: "9" }] })
          yield* llm.text("done")
        }),
      )
      const compaction = messages.find((m) => m.info.role === "assistant" && m.info.summary)
      expect(JSON.stringify(compaction?.info)).toContain("Tool call not allowed while generating summary")
      const todo = yield* Todo.Service
      expect(yield* todo.get(messages[0]!.info.sessionID)).toEqual([])
    }),
  30_000,
)
