import { describe, test, expect, beforeEach, afterEach } from "bun:test"
import fs from "fs/promises"
import os from "os"
import path from "path"
import { ShadowLog, redact } from "../../src/laya/shadow"
import { QUESTIONS } from "../../src/laya/state"

let dir: string
let log: ShadowLog

beforeEach(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), "olaya-shadow-"))
  log = new ShadowLog(dir)
})
afterEach(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

async function records(): Promise<any[]> {
  const names = await fs.readdir(dir)
  const out: any[] = []
  for (const name of names) {
    const text = await fs.readFile(path.join(dir, name), "utf8")
    for (const line of text.split("\n")) if (line.trim()) out.push(JSON.parse(line))
  }
  return out
}

const prediction = (id = "per_1") => ({
  id,
  ts: "2026-09-22T00:00:00.000Z",
  state: { action: "shell", command: "bun install" },
  questions: QUESTIONS,
  probability: 0.81,
  checkpoint: "convaiinnovations/laya",
  latencyMs: 86,
  estimatedTokens: 40,
  inputTokens: 94,
})

describe("ShadowLog", () => {
  test("a prediction alone writes nothing", async () => {
    log.predicted(prediction())
    expect(await records()).toHaveLength(0)
  })

  test("a reply produces a labelled row in the training schema", async () => {
    log.predicted(prediction())
    expect(await log.replied("per_1", "once")).toBe(true)

    const [row] = await records()
    // state/questions/gold are JSON strings, exactly as the fine-tuning notebook reads them.
    expect(typeof row.state).toBe("string")
    expect(typeof row.questions).toBe("string")
    expect(typeof row.gold).toBe("string")
    expect(JSON.parse(row.state).command).toBe("bun install")
    expect(JSON.parse(row.gold).auto_approve.probabilities).toEqual({ true: 1, false: 0 })
    expect(row.probability).toBe(0.81)
    expect(row.checkpoint).toBe("convaiinnovations/laya")
    expect(row.latency_ms).toBe(86)
  })

  test("a rejection is the opposite one-hot label", async () => {
    log.predicted(prediction())
    await log.replied("per_1", "reject")
    const [row] = await records()
    expect(JSON.parse(row.gold).auto_approve.probabilities).toEqual({ true: 0, false: 1 })
  })

  test("always is preserved distinctly from once", async () => {
    log.predicted(prediction("a"))
    log.predicted(prediction("b"))
    await log.replied("a", "once")
    await log.replied("b", "always")
    const replies = (await records()).map((r) => r.reply)
    expect(replies).toContain("once")
    expect(replies).toContain("always")
  })

  test("replying to an unknown id is a no-op", async () => {
    expect(await log.replied("nope", "once")).toBe(false)
    expect(await records()).toHaveLength(0)
  })

  test("unanswered predictions flush as unlabelled and are excluded from training", async () => {
    log.predicted(prediction("a"))
    log.predicted(prediction("b"))
    expect(await log.flushUnlabelled()).toBe(2)
    const rows = await records()
    expect(rows).toHaveLength(2)
    for (const row of rows) {
      expect(row.gold).toBeNull() // the marker a training split filters on
      expect(row.reply).toBeNull()
    }
  })

  test("a refusal records the reason and action but never the state", async () => {
    await log.refused({ id: "per_9", action: "shell", reason: "denylisted" })
    const [row] = await records()
    expect(row.kind).toBe("refusal")
    expect(row.reason).toBe("denylisted")
    expect(row.action).toBe("shell")
    expect("state" in row).toBe(false)
  })

  test("secrets are redacted before reaching disk", async () => {
    log.predicted({
      ...prediction(),
      state: {
        action: "shell",
        command: 'curl -H "Authorization: Bearer sk-abcdefghijklmnopqrst" https://api.example.com',
        task: "export AWS_KEY=AKIAIOSFODNN7EXAMPLE and set password=hunter2correct",
      },
    })
    await log.replied("per_1", "once")

    const raw = await fs.readFile(path.join(dir, (await fs.readdir(dir))[0]!), "utf8")
    expect(raw).not.toContain("sk-abcdefghijklmnopqrst")
    expect(raw).not.toContain("AKIAIOSFODNN7EXAMPLE")
    expect(raw).not.toContain("hunter2correct")
    expect(raw).toContain("«redacted")
    // The surrounding command is still legible, which is the point of redacting rather than dropping.
    expect(raw).toContain("curl")
  })

  test("purge removes every record and reports them", async () => {
    log.predicted(prediction())
    await log.replied("per_1", "once")
    expect(await records()).toHaveLength(1)

    const removed = await log.purge()
    expect(removed.length).toBe(1)
    expect(await records()).toHaveLength(0)
  })

  test("purge on an empty or missing directory is harmless", async () => {
    expect(await new ShadowLog(path.join(dir, "does-not-exist")).purge()).toEqual([])
  })
})

describe("redact", () => {
  test("leaves ordinary content untouched", () => {
    const value = { command: "bun install && bun test", files: ["src/index.ts"] }
    expect(redact(value)).toEqual(value)
  })

  test("removes private keys wholesale", () => {
    const value = { blob: "-----BEGIN RSA PRIVATE KEY-----\nabc123\n-----END RSA PRIVATE KEY-----" }
    expect(JSON.stringify(redact(value))).not.toContain("abc123")
  })

  test("removes github and slack tokens", () => {
    const out = JSON.stringify(redact({ a: "ghp_0123456789abcdefghij", b: "xoxb-123456789-abcdefghij" }))
    expect(out).not.toContain("ghp_0123456789abcdefghij")
    expect(out).not.toContain("xoxb-123456789-abcdefghij")
  })
})
