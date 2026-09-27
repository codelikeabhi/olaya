import type { SessionV1 } from "@olaya/core/v1/session"

/**
 * Loops across steps: the processor's doom-loop check only sees identical calls within one step,
 * and headless runs approve it anyway. An unattended model can repeat a failing edit for as long
 * as it is allowed to (Track E, 2026-09-27: the same failed edit 8 times running). A reminder at
 * the third repeat is usually enough for it to re-read and change course.
 */
export const REPEATS = 3
/** Reminders per run; past this, the loop is left to the step limit or the user. */
export const MAX_NUDGES = 3

/**
 * The reminder to give, or undefined. Only tool calls since the latest user message count, so a
 * reminder (itself a user message) starts the count again.
 */
export function loopNudge(msgs: SessionV1.WithParts[]) {
  const since = msgs.findLastIndex((m) => m.info.role === "user")
  const tools = msgs
    .slice(since + 1)
    .flatMap((m) => m.parts.filter((part): part is SessionV1.ToolPart => part.type === "tool"))
  const last = tools.slice(-REPEATS)
  if (last.length < REPEATS) return undefined
  const call = (part: SessionV1.ToolPart) => part.tool + "\u0000" + JSON.stringify(part.state.input ?? null)
  const error = (part: SessionV1.ToolPart) =>
    part.state.status === "error" ? part.tool + "\u0000" + part.state.error.split("\n")[0] : undefined
  const tool = last[0]!.tool
  if (last.every((part) => call(part) === call(last[0]!)))
    return `You have made the same \`${tool}\` call ${REPEATS} times in a row. Repeating it will not change the outcome. Step back: check the current state (re-read the file, or look at the latest output), then take a different approach.`
  const failure = error(last[0]!)
  if (failure && last.every((part) => error(part) === failure))
    return `Your last ${REPEATS} \`${tool}\` calls all failed with the same error: "${failure.split("\u0000")[1]}". Repeating it will not change the outcome. Step back: check the current state (re-read the file before editing it again, or look at the latest output), then take a different approach.`
  return undefined
}

export * as LoopGuard from "./loop-guard"
