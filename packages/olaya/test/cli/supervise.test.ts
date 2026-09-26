/**
 * `olaya run --supervise`: an unattended task survives its own process dying.
 */
import { describe, expect, test } from "bun:test"
import { CONTINUE_MESSAGE, resumeArgs, supervise } from "../../src/cli/supervise"

function harness(codes: (number | null)[], session: string | null = "ses_1", runMs = 10 * 60_000) {
  const calls: string[][] = []
  const waits: number[] = []
  const logs: string[] = []
  let clock = 0
  return {
    calls,
    waits,
    logs,
    deps: {
      spawn: async (args: string[]) => {
        calls.push(args)
        clock += runMs
        return codes.length ? codes.shift()! : 0 // null (killed) must stay null
      },
      session: () => session ?? undefined,
      sleep: async (ms: number) => {
        waits.push(ms)
        clock += ms
      },
      log: (line: string) => void logs.push(line),
      now: () => clock,
    },
  }
}

const first = ["run", "--model", "anthropic/claude-sonnet-5", "--dangerously-skip-permissions", "--", "fix the tests"]

describe("supervised runs", () => {
  test("a run that finishes is not restarted", async () => {
    const h = harness([0])
    expect(await supervise(first, (s) => resumeArgs(first, s), h.deps)).toBe(0)
    expect(h.calls).toHaveLength(1)
  })

  test("a crashed run resumes on its session with the same model and permissions, then finishes", async () => {
    const h = harness([null, 1, 0])
    expect(await supervise(first, (s) => resumeArgs(first, s), h.deps)).toBe(0)
    expect(h.calls).toHaveLength(3)
    expect(h.calls[1]).toEqual([
      "run",
      "--model",
      "anthropic/claude-sonnet-5",
      "--dangerously-skip-permissions",
      "--session",
      "ses_1",
      "--",
      CONTINUE_MESSAGE,
    ])
    expect(h.waits).toEqual([15_000, 60_000]) // backing off
  })

  test("six restarts within an hour is the limit", async () => {
    const h = harness(Array(20).fill(1), "ses_1", 61_000) // not quick failures, and six fit in an hour
    expect(await supervise(first, (s) => resumeArgs(first, s), h.deps)).toBe(1)
    expect(h.calls).toHaveLength(7) // the first run and six restarts
    expect(h.logs.at(-1)).toContain("olaya run --session ses_1")
  })

  test("a run that failed before creating a session is not restarted", async () => {
    const h = harness([1], null)
    expect(await supervise(first, (s) => resumeArgs(first, s), h.deps)).toBe(1)
    expect(h.calls).toHaveLength(1)
  })

  test("resuming keeps how the run was configured and drops its first message, files and session choice", () => {
    expect(
      resumeArgs(
        [
          "run",
          "--model=openai/gpt-6",
          "-f",
          "a.png",
          "--continue",
          "--agent",
          "build",
          "--format",
          "json",
          "hello",
          "--",
          "rest",
        ],
        "ses_9",
      ),
    ).toEqual([
      "run",
      "--model=openai/gpt-6",
      "--agent",
      "build",
      "--format",
      "json",
      "--session",
      "ses_9",
      "--",
      CONTINUE_MESSAGE,
    ])
  })
  test("two quick failures in a row stop the restarts: a restart won't fix a bad key", async () => {
    const h = harness([1, 1, 1], "ses_1", 5_000)
    expect(await supervise(first, (s) => resumeArgs(first, s), h.deps)).toBe(1)
    expect(h.calls).toHaveLength(2)
    expect(h.logs.at(-1)).toContain("within a minute of starting, twice")
  })
})
