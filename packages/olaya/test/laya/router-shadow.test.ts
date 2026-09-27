/**
 * Laya's deterministic router (laya-model-routing, gate G5): start, hard triggers, guardrails,
 * shadow vs live, and fail-safe. The routing head (G4) is not part of this yet.
 */
import { describe, expect, test } from "bun:test"
import fs from "fs/promises"
import os from "os"
import path from "path"
import type { PoolModel, StepUsage } from "@olaya/plugin"
import {
  FAILED_DONE_PROMPT,
  failedDone,
  handler,
  restartPoint,
  RESTART_TOKENS,
  type RoutingState,
} from "../../src/laya/routing"
import { ShadowLog } from "../../src/laya/shadow"

const model = (modelID: string, output: number): PoolModel => ({
  providerID: "anthropic",
  modelID,
  cost: { input: output / 5, output, cacheRead: output / 50 },
})
const pool = [model("claude-opus-5-5", 20), model("claude-haiku-4-5", 5), model("claude-sonnet-5", 10)]
const sonnet = { providerID: "anthropic", modelID: "claude-sonnet-5" }

const usage = (step: number, signals: Partial<StepUsage["signals"]> = {}): StepUsage => ({
  sessionID: "s",
  messageID: `m${step}`,
  step,
  providerID: "anthropic",
  modelID: "claude-sonnet-5",
  tokens: { input: 1000, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 0 },
  signals: { toolErrors: 0, malformed: 0, identicalRun: 1, sameFileEdits: 0, ...signals },
})

/** Runs steps through one handler and returns what each step chose. */
async function steps(
  options: Parameters<typeof handler>[0],
  script: {
    step: number
    point?: "start" | "step" | "compaction"
    signals?: Partial<StepUsage["signals"]>
    enabled?: boolean
    model?: typeof sonnet
  }[],
) {
  const route = handler(options)
  const out: { model?: { modelID: string }; reason?: string }[] = []
  for (const s of script) {
    const output: { model?: { providerID: string; modelID: string }; reason?: string } = {}
    await route(
      {
        sessionID: "s",
        agent: "build",
        step: s.step,
        point: s.point ?? (s.step === 1 ? "start" : "step"),
        model: s.model ?? sonnet,
        ...(s.step > 1 && { usage: usage(s.step - 1, s.signals) }),
        routing: { enabled: s.enabled ?? true, pool },
      },
      output,
    )
    out.push(output)
  }
  return out
}

const live = { mode: "live" as const, start: "default" as const }
const quiet = (n: number, from = 2) => Array.from({ length: n }, (_, i) => ({ step: from + i }))

describe("laya router", () => {
  test("starts on the prompt's model by default, or the cheapest when asked", async () => {
    expect((await steps(live, [{ step: 1 }]))[0]!.model).toBeUndefined()
    expect((await steps({ ...live, start: "cheapest" }, [{ step: 1 }]))[0]!.model?.modelID).toBe("claude-haiku-4-5")
  })

  test("a doom loop escalates one step up the pool by price, once the switch spacing allows", async () => {
    const out = await steps(live, [{ step: 1 }, ...quiet(9), { step: 11, signals: { identicalRun: 3 } }])
    expect(out.slice(0, 10).every((o) => !o.model)).toBe(true)
    expect(out[10]!.model?.modelID).toBe("claude-opus-5-5")
    expect(out[10]!.reason).toBe("escalate: doom-loop")
    const after = await steps(live, [
      { step: 1 },
      ...quiet(9),
      { step: 11, signals: { identicalRun: 3 } },
      { step: 12 },
    ])
    expect(after.at(-1)!.model?.modelID).toBe("claude-opus-5-5") // and stays there
  })

  test("a trigger too soon after a switch waits, then fires without a second trigger", async () => {
    // the prompt asked for Opus, so both the start on Haiku and the step up to Sonnet are overrides
    const opus = { providerID: "anthropic", modelID: "claude-opus-5-5" }
    const script = [
      { step: 1 },
      { step: 3, signals: { malformed: 1 } },
      { step: 4, signals: { malformed: 1 } },
      ...quiet(6, 5),
      { step: 11 },
    ]
    const out = await steps(
      { ...live, start: "cheapest" },
      script.map((s) => ({ ...s, model: opus })),
    )
    expect(out.slice(0, 9).every((o) => o.model?.modelID === "claude-haiku-4-5")).toBe(true)
    expect(out.at(-1)!.model?.modelID).toBe("claude-sonnet-5")
    expect(out.at(-1)!.reason).toBe("escalate: malformed-calls")
  })

  test("starting on the prompt's model is not a switch, so an early trigger escalates at once", async () => {
    const out = await steps(live, [
      { step: 1 },
      { step: 2, signals: { malformed: 1 } },
      { step: 3, signals: { malformed: 1 } },
    ])
    expect(out[2]!.model?.modelID).toBe("claude-opus-5-5")
  })

  test("no test progress over three runs escalates; progress does not", async () => {
    const stuck = await steps(live, [
      { step: 1 },
      ...quiet(8),
      ...[3, 3, 3].map((failed, i) => ({ step: 10 + i, signals: { tests: { passed: 5, failed } } })),
    ])
    expect(stuck.at(-1)!.reason).toBe("escalate: no-test-progress")
    const improving = await steps(live, [
      { step: 1 },
      ...quiet(8),
      ...[3, 2, 1].map((failed, i) => ({ step: 10 + i, signals: { tests: { passed: 5, failed } } })),
    ])
    expect(improving.every((o) => !o.model)).toBe(true)
  })

  test("an edit waiting for its test run holds the switch until the tests run", async () => {
    const out = await steps(live, [
      { step: 1 },
      ...quiet(9),
      { step: 11, signals: { identicalRun: 3, sameFileEdits: 1 } },
      { step: 12, signals: { sameFileEdits: 2 } },
      { step: 13, signals: { tests: { passed: 1, failed: 1 } } },
    ])
    expect(out[10]!.model).toBeUndefined()
    expect(out[11]!.model).toBeUndefined()
    expect(out[12]!.model?.modelID).toBe("claude-opus-5-5")
  })

  test("never more than two escalations, and never a step down", async () => {
    const out = await steps({ ...live, start: "cheapest" }, [
      { step: 1 },
      ...quiet(9),
      { step: 11, signals: { identicalRun: 3 } },
      ...quiet(9, 12),
      { step: 21, signals: { identicalRun: 3 } },
      ...quiet(9, 22),
      { step: 31, signals: { identicalRun: 3 } },
      { step: 32, point: "compaction" },
    ])
    const effective = out.map((o) => o.model?.modelID ?? sonnet.modelID)
    expect(effective.filter((m, i) => m !== effective[i - 1])).toEqual([
      "claude-haiku-4-5",
      "claude-sonnet-5",
      "claude-opus-5-5",
    ])
    expect(effective.slice(0, 10).every((m) => m === "claude-haiku-4-5")).toBe(true) // held, not just set once
    expect(out[30]!.reason).toContain("escalation cap reached")
    expect(effective.at(-1)).toBe("claude-opus-5-5") // compaction does not step down
  })

  test("shadow mode and a switched-off toggle record decisions but change nothing", async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "route-"))
    const shadow = new ShadowLog(dir)
    const script = [{ step: 1 }, ...quiet(9), { step: 11, signals: { identicalRun: 3 } }]
    expect((await steps({ mode: "shadow", start: "default", shadow }, script)).every((o) => !o.model)).toBe(true)
    expect(
      (
        await steps(
          { ...live, shadow },
          script.map((s) => ({ ...s, enabled: false })),
        )
      ).every((o) => !o.model),
    ).toBe(true)
    const [file] = await fs.readdir(dir)
    const records = (await fs.readFile(path.join(dir, file!), "utf8"))
      .trim()
      .split("\n")
      .map((l) => JSON.parse(l))
    expect(records).toHaveLength(22)
    expect(records.every((r) => r.kind === "route" && r.applied === false)).toBe(true)
    expect(records[10]).toMatchObject({
      step: 11,
      choice: "anthropic/claude-opus-5-5",
      reason: "escalate: doom-loop",
      mode: "shadow",
    })
    await fs.rm(dir, { recursive: true, force: true })
  })

  test("a pool of one leaves routing idle, and a failure leaves the harness's model", async () => {
    const one = handler(live)
    const output: { model?: { providerID: string; modelID: string } } = {}
    await one(
      {
        sessionID: "s",
        agent: "build",
        step: 1,
        point: "start",
        model: sonnet,
        routing: { enabled: true, pool: [pool[0]!] },
      },
      output,
    )
    expect(output.model).toBeUndefined()
    await one({ sessionID: "t", agent: "build", step: 1, point: "start", model: sonnet } as never, output)
    expect(output.model).toBeUndefined()
  })
})

describe("restart-smart escalation", () => {
  // a doom loop after 11 quiet steps on Sonnet, with the given context size
  const run = async (context: number, mode: "live" | "shadow", after: ("compaction" | "step")[]) => {
    const sessions = new Map<string, RoutingState>()
    const route = handler({ mode, start: "default" }, sessions)
    const point = restartPoint(sessions)
    const call = async (
      step: number,
      kind: "start" | "step" | "compaction",
      signals: Partial<StepUsage["signals"]> = {},
    ) => {
      const output: { model?: { providerID: string; modelID: string }; reason?: string } = {}
      const used = usage(step - 1, signals)
      used.tokens.input = context
      await route(
        {
          sessionID: "s",
          agent: "build",
          step,
          point: kind,
          model: sonnet,
          ...(step > 1 && { usage: used }),
          routing: { enabled: true, pool },
        },
        output,
      )
      const compact = { compact: false }
      await point({ sessionID: "s", tokens: context, window: 200_000, idleMs: 0, items: [] }, compact)
      return { model: output.model?.modelID, reason: output.reason, compact: compact.compact }
    }
    await call(1, "start")
    for (let step = 2; step <= 12; step++) await call(step, "step")
    const trigger = await call(13, "step", { identicalRun: 3 })
    const next = []
    for (const [i, kind] of after.entries()) next.push(await call(14 + i, kind))
    return { trigger, next }
  }

  test("with a small context the escalation is immediate", async () => {
    const r = await run(10_000, "live", [])
    expect(r.trigger.model).toBe("claude-opus-5-5")
    expect(r.trigger.compact).toBe(false)
  })

  test("with a large context it asks for a compaction and escalates after it", async () => {
    const r = await run(RESTART_TOKENS, "live", ["compaction"])
    expect(r.trigger.model).toBeUndefined() // held: the history is not sent to the stronger model
    expect(r.trigger.compact).toBe(true)
    expect(r.next[0]!.model).toBe("claude-opus-5-5")
    expect(r.next[0]!.compact).toBe(false)
  })

  test("if no compaction comes within two steps, it escalates anyway", async () => {
    const r = await run(RESTART_TOKENS, "live", ["step", "step"])
    expect(r.next[0]!.model).toBeUndefined()
    expect(r.next[1]!.model).toBe("claude-opus-5-5")
    expect(r.next[1]!.reason).toContain("no compaction came")
  })

  test("shadow mode never asks for a compaction", async () => {
    const r = await run(RESTART_TOKENS, "shadow", [])
    expect(r.trigger.compact).toBe(false)
  })
})

describe("a failed done", () => {
  const opus = { providerID: "anthropic", modelID: "claude-opus-5-5" }
  // the prompt's model is Opus and routing starts on the cheapest (Haiku, a switch at step 1); tests
  // run at step 3, an edit at step 4, then the turn ends
  const run = async (opts: { mode?: "live" | "shadow"; failed?: number; tested?: boolean; on?: typeof sonnet; already?: boolean }) => {
    const sessions = new Map<string, RoutingState>()
    const route = handler({ mode: opts.mode ?? "live", start: opts.on ? "default" : "cheapest" }, sessions)
    const exit = failedDone(sessions)
    const call = async (step: number, signals: Partial<StepUsage["signals"]> = {}) => {
      const output: { model?: { providerID: string; modelID: string }; reason?: string } = {}
      await route(
        {
          sessionID: "s",
          agent: "build",
          step,
          point: step === 1 ? "start" : "step",
          model: opts.on ?? opus,
          ...(step > 1 && { usage: usage(step - 1, signals) }),
          routing: { enabled: true, pool },
        },
        output,
      )
      return output
    }
    await call(1)
    await call(2)
    await call(3, opts.tested === false ? {} : { tests: { passed: 13, failed: opts.failed ?? 15 } })
    await call(4, { sameFileEdits: 1 })
    const ended = { continue: opts.already ?? false, prompt: opts.already ? "other" : undefined } as {
      continue: boolean
      prompt?: string
    }
    await exit({ sessionID: "s", agent: "build", step: 4, text: "The tests still fail; I could not finish." }, ended)
    const next = await call(5, { sameFileEdits: 1 })
    return { ended, next }
  }

  test("ending with failing tests goes on, one model up, even with an edit untested and soon after the start", async () => {
    const r = await run({})
    expect(r.ended).toEqual({ continue: true, prompt: FAILED_DONE_PROMPT })
    expect(r.next.model?.modelID).toBe("claude-sonnet-5")
    expect(r.next.reason).toBe("escalate: failed-done")
  })

  test("passing tests, no test run, the strongest model, shadow mode or another nudge: the turn ends as it would", async () => {
    for (const r of [
      await run({ failed: 0 }),
      await run({ tested: false }),
      await run({ on: opus }),
      await run({ mode: "shadow" }),
    ]) {
      expect(r.ended.continue).toBe(false)
      expect(r.next.reason).not.toBe("escalate: failed-done")
      expect(r.next.model?.modelID).not.toBe("claude-sonnet-5")
    }
    const other = await run({ already: true })
    expect(other.ended.prompt).toBe("other")
  })
})
