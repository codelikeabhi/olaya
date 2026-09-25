import { describe, test, expect } from "bun:test"
import { compact, looksEnglish, summariseDiff, QUESTIONS, AUTO_APPROVE } from "../../src/laya/state"

const shell = (command: string, patterns: string[] = ["bun install"]) => ({
  permission: "shell",
  patterns,
  metadata: { command },
})

describe("compact", () => {
  test("the action and its subject come first", () => {
    const result = compact(shell("bun install"), { task: "reinstall deps", cwd: "." }, 400)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    // Key order is what survives head-truncation, so assert the order itself.
    expect(Object.keys(result.state).slice(0, 2)).toEqual(["action", "command"])
    expect(Object.keys(result.state).indexOf("task")).toBeGreaterThan(Object.keys(result.state).indexOf("command"))
  })

  test("field order is stable across identical requests", () => {
    const args = [shell("bun install"), { task: "t", cwd: ".", recent: ["a", "b"] }, 400] as const
    const a = compact(...args)
    const b = compact(...args)
    expect(a.ok && b.ok).toBe(true)
    if (a.ok && b.ok) expect(JSON.stringify(a.state)).toBe(JSON.stringify(b.state))
  })

  test("an oversized state drops history, never the action", () => {
    const recent = Array.from({ length: 200 }, (_, i) => `read packages/olaya/src/tool/file${i}.ts`)
    const result = compact(shell("rm -rf ./node_modules && bun install"), { task: "deps are broken", recent, cwd: "." }, 60)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.state["action"]).toBe("shell")
    expect(result.state["command"]).toContain("rm -rf")
    expect(result.state["recent"]).toBeUndefined()
    expect(result.dropped).toContain("recent")
    expect(result.estimatedTokens).toBeLessThanOrEqual(60)
  })

  test("history is dropped before intent", () => {
    const recent = Array.from({ length: 60 }, (_, i) => `step ${i}`)
    const result = compact(shell("git push"), { task: "ship it", recent }, 40)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    // `recent` must go first; `task` only if there is still no room.
    expect(result.dropped[0]).toBe("recent")
  })

  test("a command too large even alone is shrunk rather than silently cut", () => {
    const huge = "echo " + "x".repeat(20_000)
    const result = compact(shell(huge, []), {}, 50)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect((result.state["command"] as string).length).toBeLessThan(huge.length)
    expect(result.state["action"]).toBe("shell")
  })

  test("missing intent is omitted, not faked", () => {
    const result = compact(shell("ls"), {}, 400)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect("task" in result.state).toBe(false)
  })

  test("parsed shell patterns are carried through", () => {
    const result = compact(shell("rm -rf build", ["rm -rf *", "build"]), {}, 400)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.state["patterns"]).toEqual(["rm -rf *", "build"])
  })

  test("an edit is summarised, not inlined", () => {
    const diff = ["--- a/x.ts", "+++ b/x.ts", "+added one", "+added two", "-removed one"].join("\n")
    const result = compact(
      { permission: "edit", patterns: ["x.ts"], metadata: { filepath: "x.ts", diff } },
      {},
      400,
    )
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.state["files"]).toEqual(["x.ts"])
    expect(result.state["changes"]).toEqual({ added: 2, removed: 1 })
    expect(JSON.stringify(result.state)).not.toContain("added one")
  })

  test("a webfetch carries its url", () => {
    const result = compact({ permission: "webfetch", patterns: ["https://x.dev"], metadata: { url: "https://x.dev" } }, {}, 400)
    expect(result.ok && result.state["url"]).toBe("https://x.dev")
  })

  test("a non-English state is refused outright", () => {
    const result = compact(shell("echo hi"), { task: "តើនេះមានសុវត្ថិភាពទេ ខ្ញុំចង់ដំឡើងកញ្ចប់ទាំងអស់" }, 400)
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.reason).toBe("non-english")
  })
})

describe("looksEnglish", () => {
  test("ordinary commands and prose pass", () => {
    expect(looksEnglish(JSON.stringify({ action: "shell", command: "rm -rf ./node_modules && bun install" }))).toBe(true)
    expect(looksEnglish("please reinstall the dependencies for me")).toBe(true)
  })

  test("non-Latin script fails", () => {
    expect(looksEnglish("これは安全ですか、パッケージを全部入れ直したいのですが")).toBe(false)
    expect(looksEnglish("តើនេះមានសុវត្ថិភាពទេ ខ្ញុំចង់ដំឡើងកញ្ចប់")).toBe(false)
  })

  test("a few accents do not trip the gate", () => {
    expect(looksEnglish("please rename the café module and update naïve callers")).toBe(true)
  })

  test("too little text to judge is allowed through", () => {
    expect(looksEnglish("ls")).toBe(true)
  })
})

describe("summariseDiff", () => {
  test("counts changes and ignores file headers", () => {
    expect(summariseDiff(["--- a/f", "+++ b/f", "+one", "+two", "-three", " ctx"].join("\n"))).toEqual({
      added: 2,
      removed: 1,
    })
  })
})

describe("QUESTIONS", () => {
  test("v1 asks exactly one binary question", () => {
    expect(Object.keys(QUESTIONS)).toEqual([AUTO_APPROVE])
    expect(QUESTIONS[AUTO_APPROVE]!.type).toBe("noul")
  })
})
