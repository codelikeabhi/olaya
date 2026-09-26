/**
 * Laya's deterministic router (laya-model-routing, gate G5): start, hard triggers, guardrails,
 * shadow vs live, and fail-safe. The routing head (G4) is not part of this yet.
 */
import { describe, expect, test } from "bun:test"
import fs from "fs/promises"
import os from "os"
import path from "path"
import type { PoolModel, StepUsage } from "@olaya/plugin"
import { handler } from "../../src/laya/routing"
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
