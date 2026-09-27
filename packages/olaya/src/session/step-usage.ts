import type { SessionV1 } from "@olaya/core/v1/session"
import type { StepUsage } from "@olaya/plugin"

/**
 * One completed model step, as plugins see it: what it cost and cheap signals of trouble. Every
 * signal is counted from tool parts, never inferred by a model, so routing and retention can act
 * on it without a second opinion.
 */
export function stepUsage(history: SessionV1.WithParts[], step: number, message: SessionV1.WithParts): StepUsage {
  const info = message.info as SessionV1.Assistant
  const tools = message.parts.filter((part): part is SessionV1.ToolPart => part.type === "tool")
  const earlier = history
    .slice(0, history.findIndex((m) => m.info.id === info.id))
    .flatMap((m) => m.parts.filter((part): part is SessionV1.ToolPart => part.type === "tool"))
  const tests = tools.flatMap((part) => testCounts(part) ?? [])
  return {
    sessionID: info.sessionID,
    messageID: info.id,
    step,
    providerID: info.providerID,
    modelID: info.modelID,
    tokens: {
      input: info.tokens.input,
      output: info.tokens.output,
      reasoning: info.tokens.reasoning,
      cacheRead: info.tokens.cache.read,
      cacheWrite: info.tokens.cache.write,
    },
    // A provider that reports no cost leaves it absent, never zero: benchmarks price it themselves.
    ...(info.cost ? { cost: info.cost } : {}),
    signals: {
      toolErrors: tools.filter((part) => part.state.status === "error").length,
      malformed: tools.filter(malformed).length,
      identicalRun: identicalRun([...earlier, ...tools]),
      sameFileEdits: sameFileEdits([...earlier, ...tools]),
      ...(tests.length
        ? { tests: { passed: tests.reduce((n, t) => n + t.passed, 0), failed: tests.reduce((n, t) => n + t.failed, 0) } }
        : {}),
    },
  }
}

const EDIT_TOOLS = new Set(["edit", "write", "multiedit"])
const TEST_COMMAND = /\b(bun test|npm (run )?test|pnpm (run )?test|yarn test|pytest|go test|cargo test|jest|vitest|mocha|rspec|phpunit|dotnet test|mvn test|gradle test)\b/

function malformed(part: SessionV1.ToolPart) {
  if (part.tool === "invalid") return true
  return part.state.status === "error" && /called with invalid arguments/.test(part.state.error)
}

/** Length of the trailing run of identical calls (same tool, same input); 1 means no repeat. */
function identicalRun(tools: SessionV1.ToolPart[]) {
  const key = (part: SessionV1.ToolPart) => part.tool + "\u0000" + JSON.stringify(part.state.input ?? null)
  const last = tools.at(-1)
  if (!last) return 0
  const target = key(last)
  const run = tools.toReversed().findIndex((part) => key(part) !== target)
  return run === -1 ? tools.length : run
}

/**
 * The files one tool call edits: `filePath` for the edit tools, and each file an `apply_patch` updates
 * or adds (the edit tool GPT models get; without it their edits went uncounted).
 */
export function editedFiles(part: SessionV1.ToolPart): string[] {
  const input = (part.state.input ?? {}) as { filePath?: unknown; patchText?: unknown }
  if (EDIT_TOOLS.has(part.tool)) return typeof input.filePath === "string" ? [input.filePath] : []
  if (part.tool !== "apply_patch" || typeof input.patchText !== "string") return []
  return [...new Set([...input.patchText.matchAll(/^\*\*\* (?:Update|Add) File: (.+)$/gm)].map((m) => m[1]!.trim()))]
}

/** The most edits to any one file since the last test run. */
function sameFileEdits(tools: SessionV1.ToolPart[]) {
  const since = tools.findLastIndex((part) => testCounts(part) !== undefined)
  const counts = new Map<string, number>()
  for (const part of tools.slice(since + 1))
    for (const file of editedFiles(part)) counts.set(file, (counts.get(file) ?? 0) + 1)
  return Math.max(0, ...counts.values())
}

/** Pass/fail counts when a recognised test command ran; undefined when none did. */
export function testCounts(part: SessionV1.ToolPart): { passed: number; failed: number } | undefined {
  if (part.tool !== "bash" || part.state.status !== "completed") return undefined
  const command = String((part.state.input as { command?: unknown } | undefined)?.command ?? "")
  if (!TEST_COMMAND.test(command)) return undefined
  const output = part.state.output
  const count = (re: RegExp) => [...output.matchAll(re)].reduce((n, m) => n + Number(m[1]), 0)
  return {
    // "12 passed" / "12 pass" (pytest, jest, vitest, bun); "ok  pkg" lines (go test)
    passed: count(/\b(\d+) pass(?:ed|ing)?\b/gi) + (output.match(/^ok\s+\S+/gm)?.length ?? 0),
    // "3 failed" / "3 fail" / "3 failing"; "FAIL pkg" lines (go test)
    failed: count(/\b(\d+) fail(?:ed|ing|ures?)?\b/gi) + (output.match(/^FAIL\s+\S+/gm)?.length ?? 0),
  }
}
