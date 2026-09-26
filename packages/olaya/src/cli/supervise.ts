/**
 * `olaya run --supervise`: keeps an unattended task going when the Olaya process itself dies.
 *
 * The task runs in a child `olaya run`. If the child crashes or exits with an error, the
 * supervisor starts it again on the same session with a message to check the current state and
 * carry on; the history is in the session store, so nothing already done is lost. Restarts back
 * off (15 s, 1 min, 5 min, then 15 min) and stop after 6 within an hour, so a permanently broken
 * setup ends instead of looping.
 */

export const CONTINUE_MESSAGE =
  "The previous run of this task stopped unexpectedly. Check the current state of the work (files, tests, anything half-done), then continue the task from where it left off."

const BACKOFF_MS = [15_000, 60_000, 5 * 60_000, 15 * 60_000]
const MAX_PER_HOUR = 6
/** A run failing this soon after starting, twice in a row, is failing for a reason a restart won't fix. */
const QUICK_FAILURE_MS = 60_000

export type Deps = {
  /** Run one child `olaya run` with these arguments; resolves with its exit code (null when killed). */
  spawn: (args: string[], env: Record<string, string>) => Promise<number | null>
  /** The session ID the child recorded, if it got that far. */
  session: () => string | undefined
  sleep: (ms: number) => Promise<void>
  log: (line: string) => void
  now?: () => number
}

/**
 * Supervise a run. `first` are the run's own arguments (without `--supervise`); `resume` builds the
 * arguments for continuing a session. Resolves with the exit code to leave with.
 */
export async function supervise(first: string[], resume: (session: string) => string[], deps: Deps) {
  const now = deps.now ?? Date.now
  const restarts: number[] = []
  let args = first
  let quick = 0
  while (true) {
    const started = now()
    const code = await deps.spawn(args, {})
    if (code === 0) return 0
    quick = now() - started < QUICK_FAILURE_MS ? quick + 1 : 0
    const session = deps.session()
    const recent = restarts.filter((t) => now() - t < 3_600_000)
    if (!session) {
      deps.log(`olaya: the run failed before it had a session (exit ${code}); not restarting`)
      return code ?? 1
    }
    if (quick >= 2) {
      deps.log(
        `olaya: the run failed within a minute of starting, twice; a restart won't fix that (check credentials and config). Resume with: olaya run --session ${session}`,
      )
      return code ?? 1
    }
    if (recent.length >= MAX_PER_HOUR) {
      deps.log(
        `olaya: ${MAX_PER_HOUR} restarts in the last hour; stopping. Resume with: olaya run --session ${session}`,
      )
      return code ?? 1
    }
    const wait = BACKOFF_MS[Math.min(recent.length, BACKOFF_MS.length - 1)]!
    deps.log(`olaya: the run exited (${code ?? "killed"}); resuming session ${session} in ${Math.round(wait / 1000)}s`)
    await deps.sleep(wait)
    restarts.push(now())
    args = resume(session)
  }
}

/**
 * The arguments to continue a session: the original options that shape how it runs, without the
 * first message, attachments or session selection.
 */
export function resumeArgs(original: string[], session: string) {
  const out: string[] = ["run"]
  const keep = new Set([
    "--model",
    "-m",
    "--agent",
    "--format",
    "--dir",
    "--variant",
    "--port",
    "--attach",
    "--password",
    "-p",
    "--username",
    "-u",
  ])
  const flags = new Set(["--auto", "--yolo", "--dangerously-skip-permissions", "--thinking"])
  const start = original.indexOf("run") + 1
  for (let i = start; i < original.length; i++) {
    const arg = original[i]!
    if (arg === "--") break
    const [name] = arg.split("=")
    if (keep.has(name!)) {
      out.push(arg)
      if (!arg.includes("=") && i + 1 < original.length) out.push(original[++i]!)
    } else if (flags.has(name!)) out.push(arg)
  }
  return [...out, "--session", session, "--", CONTINUE_MESSAGE]
}

export * as Supervise from "./supervise"
