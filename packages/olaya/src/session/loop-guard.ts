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
    // calls a provider failure cut off are the harness's leftovers, not the model repeating itself
    .filter((part) => !(part.state.status === "error" && part.state.metadata?.interrupted === true))
  const last = tools.slice(-REPEATS)
  if (last.length < REPEATS) return undefined
  // the same call with the same result: polling a job or a log until it changes is not a loop
  const call = (part: SessionV1.ToolPart) =>
    [
      part.tool,
      JSON.stringify(part.state.input ?? null),
      part.state.status === "completed" ? part.state.output : part.state.status === "error" ? part.state.error : "",
    ].join("\u0000")
  const error = (part: SessionV1.ToolPart) =>
    part.state.status === "error" ? part.tool + "\u0000" + part.state.error.split("\n")[0] : undefined
  if (last.every((part) => call(part) === call(last[0]!)))
    return `You have made the same \`${last[0]!.tool}\` call ${REPEATS} times in a row. Repeating it will not change the outcome. Step back: check the current state (re-read the file, or look at the latest output), then take a different approach.`
  // The same failure from one tool, with other calls in between: re-reading the file and sending
  // the same failing edit again is a loop too (Track C, 2026-09-27: read, failed edit, 77 times).
  const failed = tools.findLast((part) => part.state.status === "error")
  if (!failed || tools.indexOf(failed) < tools.length - 2) return undefined
  const same = tools.filter((part) => part.tool === failed.tool).slice(-REPEATS)
  const failure = error(failed)
  if (same.length === REPEATS && same.every((part) => error(part) === failure))
    return `Your last ${REPEATS} \`${failed.tool}\` calls all failed with the same error: "${failure!.split("\u0000")[1]}". Repeating it will not change the outcome. Step back: check the current state (re-read the file before editing it again, or look at the latest output), then take a different approach.`
  return undefined
}

export * as LoopGuard from "./loop-guard"
