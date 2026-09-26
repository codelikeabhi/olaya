import { describe, expect, test } from "bun:test"
import type { SessionV1 } from "@olaya/core/v1/session"
import { stepUsage, testCounts } from "../../src/session/step-usage"

let seq = 0
const tool = (name: string, input: Record<string, unknown>, state: { output?: string; error?: string } = {}) =>
  ({
    id: `prt_${++seq}`,
    type: "tool",
    tool: name,
    callID: `call_${seq}`,
    state: state.error
      ? { status: "error", input, error: state.error, time: { start: 0, end: 1 } }
      : { status: "completed", input, output: state.output ?? "", title: "", metadata: {}, time: { start: 0, end: 1 } },
  }) as unknown as SessionV1.ToolPart

const assistant = (parts: SessionV1.Part[], usage: { cost?: number; input?: number } = {}) =>
  ({
    info: {
      id: `msg_${++seq}`,
      role: "assistant",
      sessionID: "ses_1",
      providerID: "anthropic",
      modelID: "claude-sonnet-5",
      cost: usage.cost ?? 0,
      tokens: { input: usage.input ?? 10, output: 5, reasoning: 1, cache: { read: 900, write: 40 } },
    },
    parts,
  }) as unknown as SessionV1.WithParts

const bash = (command: string, output = "") => tool("bash", { command }, { output })
const edit = (filePath: string) => tool("edit", { filePath, oldString: "a", newString: "b" })

describe("stepUsage", () => {
  test("carries the message's model and four token counts", () => {
    const msg = assistant([], { input: 123 })
    const u = stepUsage([msg], 4, msg)
    expect(u).toMatchObject({ step: 4, providerID: "anthropic", modelID: "claude-sonnet-5", sessionID: "ses_1" })
    expect(u.tokens).toEqual({ input: 123, output: 5, reasoning: 1, cacheRead: 900, cacheWrite: 40 })
  })

  test("cost is absent when the provider reports none, present when it does", () => {
    const free = assistant([])
    expect("cost" in stepUsage([free], 1, free)).toBe(false)
    const paid = assistant([], { cost: 0.0123 })
    expect(stepUsage([paid], 1, paid).cost).toBe(0.0123)
  })

  test("counts errored and malformed tool calls separately", () => {
    const msg = assistant([
      tool("invalid", { tool: "edit" }),
      tool("edit", {}, { error: "The edit tool was called with invalid arguments: filePath missing." }),
      tool("bash", { command: "ls" }, { error: "exit 1" }),
      bash("ls", "a b"),
    ])
    const s = stepUsage([msg], 1, msg).signals
    expect(s.toolErrors).toBe(2)
    expect(s.malformed).toBe(2)
  })

  test("the identical-call run spans earlier steps", () => {
    const first = assistant([bash("npm run build"), bash("npm run build")])
    const second = assistant([bash("npm run build")])
    expect(stepUsage([first, second], 2, second).signals.identicalRun).toBe(3)
    const changed = assistant([bash("npm run lint")])
    expect(stepUsage([first, changed], 2, changed).signals.identicalRun).toBe(1)
  })

  test("same-file edits count since the last test run", () => {
    const before = assistant([edit("/a.ts"), edit("/a.ts"), edit("/a.ts"), bash("bun test", " 3 pass\n 0 fail")])
    const after = assistant([edit("/a.ts"), edit("/b.ts"), edit("/a.ts")])
    expect(stepUsage([before, after], 2, after).signals.sameFileEdits).toBe(2)
    const noTest = assistant([edit("/a.ts"), edit("/a.ts")])
    expect(stepUsage([noTest], 1, noTest).signals.sameFileEdits).toBe(2)
  })

  test("test counts appear only when a recognised test command ran", () => {
    const withTests = assistant([bash("pytest -q", "1 failed, 3 passed in 0.20s")])
    expect(stepUsage([withTests], 1, withTests).signals.tests).toEqual({ passed: 3, failed: 1 })
    const notATest = assistant([bash("echo '2 passed'", "2 passed")])
    expect(stepUsage([notATest], 1, notATest).signals.tests).toBeUndefined()
  })
})

describe("testCounts", () => {
  test.each([
    ["bun test", " 12 pass\n 2 fail\n Ran 14 tests", { passed: 12, failed: 2 }],
    ["npx jest", "Tests:       1 failed, 7 passed, 8 total", { passed: 7, failed: 1 }],
    ["go test ./...", "ok  \tpkg/a\t0.1s\nFAIL\tpkg/b\t0.2s\nFAIL", { passed: 1, failed: 1 }],
    ["cargo test", "test result: FAILED. 4 passed; 1 failed; 0 ignored", { passed: 4, failed: 1 }],
  ])("%s", (command, output, expected) => {
    expect(testCounts(bash(command, output))).toEqual(expected)
  })
})
