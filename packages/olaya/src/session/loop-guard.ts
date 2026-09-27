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
  // The same call with the same result: polling a job or a log until it changes is not a loop. Test
  // timings and saved-output file names differ on every run, so durations and long IDs with digits
  // don't count as a change; other numbers (a progress count, a falling loss) do.
  const outcome = (part: SessionV1.ToolPart) =>
    (part.state.status === "completed" ? part.state.output : part.state.status === "error" ? part.state.error : "")
      .replace(/\d+(?:\.\d+)?\s?(?:ms|s|sec|seconds|min)\b/g, "#")
      .replace(/\b(?=[\w-]*\d)[\w-]{8,}\b/g, "#")
  const call = (part: SessionV1.ToolPart) =>
    [part.tool, JSON.stringify(part.state.input ?? null), outcome(part)].join("\u0000")
  const error = (part: SessionV1.ToolPart) =>
    part.state.status === "error" ? part.tool + "\u0000" + part.state.error.split("\n")[0] : undefined
  // what a call acts on: the same failure on different files or URLs is not a repeat
  const target = (part: SessionV1.ToolPart) => {
    const input = (part.state.input ?? {}) as Record<string, unknown>
    return JSON.stringify(input.filePath ?? input.path ?? input.url ?? input.command ?? null)
  }
  if (last.every((part) => call(part) === call(last[0]!)))
    return `You have made the same \`${last[0]!.tool}\` call ${REPEATS} times in a row. Repeating it will not change the outcome. Step back: check the current state (re-read the file, or look at the latest output), then take a different approach.`
  // The same failure from one tool, with other calls in between: re-reading the file and sending
  // the same failing edit again is a loop too (Track C, 2026-09-27: read, failed edit, 77 times).
  const failed = tools.findLast((part) => part.state.status === "error")
  if (!failed || tools.indexOf(failed) < tools.length - 2) return undefined
  const same = tools.filter((part) => part.tool === failed.tool).slice(-REPEATS)
  const failure = error(failed)
  if (same.length === REPEATS && same.every((part) => error(part) === failure && target(part) === target(failed)))
    return `Your last ${REPEATS} \`${failed.tool}\` calls all failed with the same error: "${failure!.split("\u0000")[1]}". Repeating it will not change the outcome. Step back: check the current state (re-read the file before editing it again, or look at the latest output), then take a different approach.`
  return undefined
}

/**
 * A run about to end on a sentence announcing the model's own next step ("Let me check the file."),
 * with no tool call made. Half the failed runs of the local benchmarks ended this way (11 of 21,
 * 2026-09-27); the work stopped because nothing asked for the next step.
 *
 * Only the last line or sentence counts: a report that ends in a list or a code block is an answer.
 * Handing back to the user ("I'll wait for your go-ahead"), declining ("I will not…") and questions
 * are theirs to answer, not announced steps.
 */
export function announcedAction(text: string) {
  const last = text
    .trim()
    .replace(/[*_`"')\]\s]+$/, "")
    .split(/\n+|(?<=[.!?])\s+/)
    .map((segment) => segment.trim())
    .filter(Boolean)
    .at(-1)
  if (!last || last.length > 300 || last.endsWith("?")) return false
  if (/\b(?:you|your|not|never)\b|n't\b/i.test(last)) return false
  return /\b(?:let me(?! know)|let's(?! (?:summari[sz]e|recap))|i'll|i will|i'm going to|i am going to|next,? i(?:'ll| will)|now,? i(?:'ll| will))\b/i.test(
    last,
  )
}

export const ACTION_NUDGE =
  "You ended by saying what you would do next, but you called no tool, so nothing happened. If there is more to do, do it now with a tool call. If the task is already complete, give your final answer again in full: only your last message is returned."

export * as LoopGuard from "./loop-guard"
