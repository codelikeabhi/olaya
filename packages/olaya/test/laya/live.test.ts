import { describe, test, expect, beforeEach, afterEach } from "bun:test"
import fs from "fs/promises"
import os from "os"
import path from "path"
import { liveHandler } from "../../src/laya/judgment"
import { ShadowLog } from "../../src/laya/shadow"
import type { Gate, LayaClient } from "../../src/laya/client"

const request = (command: string) => ({ id: "per_1", sessionID: "ses_1", permission: "bash", patterns: [], metadata: { command } })

function client(p: number | "fail", gate: Gate | undefined) {
  return {
    stateBudget: async () => 471,
    gate: async () => gate,
    decide: async () =>
      p === "fail"
        ? { ok: false, reason: "timeout" }
        : { ok: true, probabilities: { auto_approve: p }, latencyMs: 20, checkpoint: "ckpt-a" },
  } as unknown as LayaClient
}

const passed: Gate = { passed: true, threshold: 0.9, alpha: 0.01, delta: 0.05 }

let dir: string
let log: ShadowLog
beforeEach(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), "olaya-live-"))
  log = new ShadowLog(dir)
})
afterEach(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

async function run(p: number | "fail", gate: Gate | undefined, command = "npm test", status: "ask" | "allow" | "deny" = "ask") {
  const output = { status }
  await liveHandler({ client: () => client(p, gate), shadow: log })(request(command), output)
  return output.status
}

async function audit() {
  const files = (await fs.readdir(dir)).filter((f) => f.endsWith(".jsonl"))
  const lines = (await Promise.all(files.map((f) => fs.readFile(path.join(dir, f), "utf8")))).join("").trim()
  return lines ? lines.split("\n").map((l) => JSON.parse(l)) : []
}

describe("live mode", () => {
  test("allows at or above the certified threshold, and audits the grant", async () => {
    expect(await run(0.95, passed)).toBe("allow")
    const records = (await audit()).filter((r) => r.kind === "auto-approved")
    expect(records).toHaveLength(1)
    expect(records[0]).toMatchObject({ id: "per_1", action: "bash", probability: 0.95, threshold: 0.9, checkpoint: "ckpt-a" })
  })

  test("below the threshold the user is asked", async () => {
    expect(await run(0.89, passed)).toBe("ask")
  })

  test("an uncertified checkpoint never grants, however confident", async () => {
    expect(await run(0.999, undefined)).toBe("ask")
    expect(await run(0.999, { passed: false, threshold: 0.5 })).toBe("ask")
  })

  test("denylisted actions are never granted", async () => {
    expect(await run(0.999, passed, "git push --force origin main")).toBe("ask")
  })

  test("a failed judgment leaves the ask in place", async () => {
    expect(await run("fail", passed)).toBe("ask")
  })

  test("it never denies and never overrides a decision already made", async () => {
    expect(await run(0.0, passed)).toBe("ask")
    expect(await run(0.999, passed, "npm test", "deny")).toBe("deny")
  })

  test("nothing is written to the audit log unless a permission was granted", async () => {
    await run(0.5, passed)
    expect((await audit()).filter((r) => r.kind === "auto-approved")).toHaveLength(0)
  })
})

describe("mode configuration", () => {
  const { resolve } = require("../../src/laya/config") as typeof import("../../src/laya/config")
  test("shadow is the default; only the exact value 'live' selects live mode", () => {
    expect(resolve({}, { OLAYA_LAYA_ENABLED: "1" }).mode).toBe("shadow")
    expect(resolve({}, { OLAYA_LAYA_ENABLED: "1", OLAYA_LAYA_MODE: "live" }).mode).toBe("live")
    expect(resolve({}, { OLAYA_LAYA_ENABLED: "1", OLAYA_LAYA_MODE: "LIVE!" }).mode).toBe("shadow")
  })
})

describe("who replied", () => {
  test("an auto-approved reply is recorded as such, so it never becomes a training label", async () => {
    log.predicted({ id: "per_9", ts: "t", state: { action: "bash" }, questions: {}, probability: 0.4, checkpoint: "c", latencyMs: 1 })
    expect(await log.replied("per_9", "once", "auto")).toBe(true)
    const rows = (await audit()).filter((r) => r.kind === "decision")
    expect(rows[0].replier).toBe("auto")
  })
})
