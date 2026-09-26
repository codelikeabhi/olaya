/**
 * End-to-end check against the real checkpoint.
 *
 * Skipped unless OLAYA_LAYA_INTEGRATION=1, because it loads an 848 MB model and takes
 * ~25 s. Point OLAYA_LAYA_PYTHON at an interpreter that has `laya` installed.
 */
import { describe, test, expect, beforeAll, afterAll } from "bun:test"
import fs from "fs/promises"
import os from "os"
import path from "path"
import { Sidecar } from "../../src/laya/sidecar"
import { resolve } from "../../src/laya/config"
import { evaluate } from "../../src/laya/judgment"
import { ShadowLog } from "../../src/laya/shadow"
import { compact } from "../../src/laya/state"

const ENABLED = process.env["OLAYA_LAYA_INTEGRATION"] === "1"
const suite = ENABLED ? describe : describe.skip

let sidecar: Sidecar
let shadow: ShadowLog
let dir: string

suite("laya end to end", () => {
  beforeAll(async () => {
    process.env["OLAYA_LAYA_DIR"] = path.resolve(import.meta.dir, "../../../../laya")
    dir = await fs.mkdtemp(path.join(os.tmpdir(), "olaya-e2e-"))
    shadow = new ShadowLog(dir)
    sidecar = new Sidecar(resolve({ enabled: true, shadow: true, timeoutMs: 5000 }))
    await sidecar.start()

    const deadline = Date.now() + 120_000
    let last = "no health response"
    while (Date.now() < deadline) {
      // Check the process first: a failed spawn leaves no client, so polling health alone
      // would loop silently until the deadline and report nothing useful.
      if (sidecar.status === "given-up") throw new Error("sidecar gave up: " + (sidecar.lastError ?? "unknown"))
      const client = sidecar.current()
      if (!client) {
        last = `no client (status=${sidecar.status})`
        await Bun.sleep(500)
        continue
      }
      const health = await client.health(2000)
      if (health?.ready) return
      if (health?.error) throw new Error("sidecar failed to load: " + health.error)
      last = health ? `ready=false (status=${sidecar.status})` : `unreachable (status=${sidecar.status})`
      await Bun.sleep(500)
    }
    throw new Error(`sidecar never became ready: ${last}`)
  }, 180_000)

  afterAll(async () => {
    await sidecar?.stop()
    await fs.rm(dir, { recursive: true, force: true })
  })

  test("reports a real state budget from the checkpoint", async () => {
    const budget = await sidecar.current()!.stateBudget()
    expect(budget).toBe(316) // 512 max_len - 192 head_max_len - 4
  })

  test("judges a real permission request and records it", async () => {
    // Warm up first: the very first inference pays MPS kernel setup (~380 ms observed), which
    // is a startup cost, not the per-decision cost the permission path actually pays.
    await evaluate(
      { id: "per_warm", sessionID: "ses_e2e", permission: "shell", patterns: ["ls"], metadata: { command: "ls" } },
      { client: () => sidecar.current(), context: async () => ({ task: "warm up" }) },
    )

    const started = Date.now()
    const result = await evaluate(
      {
        id: "per_e2e",
        sessionID: "ses_e2e",
        permission: "shell",
        patterns: ["rm -rf *", "bun install"],
        metadata: { command: "rm -rf ./node_modules && bun install" },
      },
      { client: () => sidecar.current(), shadow, context: async () => ({ task: "deps are broken, reinstall them" }) },
    )
    const elapsed = Date.now() - started

    expect(result.evaluated).toBe(true)
    if (result.evaluated) {
      expect(result.probability).toBeGreaterThanOrEqual(0)
      expect(result.probability).toBeLessThanOrEqual(1)
      console.log(`  p(auto_approve)=${result.probability}  end-to-end=${elapsed}ms`)
    }
    // Steady-state cost on the permission path.
    const samples: number[] = []
    for (let i = 0; i < 5; i++) {
      const t = Date.now()
      await evaluate(
        { id: `per_m${i}`, sessionID: "ses_e2e", permission: "shell", patterns: ["ls"], metadata: { command: "ls -la" } },
        { client: () => sidecar.current(), context: async () => ({ task: "look around" }) },
      )
      samples.push(Date.now() - t)
    }
    samples.sort((a, b) => a - b)
    console.log(`  steady-state end-to-end: p50=${samples[2]}ms min=${samples[0]}ms max=${samples[4]}ms`)

    // Comfortably inside the 86 ms full-context model baseline plus HTTP overhead.
    expect(samples[2]!).toBeLessThan(200)
    expect(elapsed).toBeLessThan(500)
  }, 60_000)

  test("the token estimate is conservative, never optimistic", async () => {
    const client = sidecar.current()!
    const budget = (await client.stateBudget())!
    const recent = Array.from({ length: 40 }, (_, i) => `read packages/olaya/src/tool/file${i}.ts`)
    const compacted = compact(
      { permission: "shell", patterns: ["rm -rf *"], metadata: { command: "rm -rf ./node_modules && bun install" } },
      { task: "deps are broken, reinstall them", recent, cwd: "." },
      budget,
    )
    expect(compacted.ok).toBe(true)
    if (!compacted.ok) return

    const judgment = await client.decide(compacted.state, {
      auto_approve: { type: "noul", instructions: "Is it safe?" },
    })
    expect(judgment.ok).toBe(true)
    if (!judgment.ok) return

    // input_tokens covers the whole sequence (head + options + state), so it exceeds the
    // state alone. What matters is that our estimate did not let the state overflow: the
    // real sequence must still fit inside max_len.
    console.log(`  estimated=${compacted.estimatedTokens} state tokens, real sequence=${judgment.inputTokens} tokens`)
    expect(judgment.inputTokens).toBeLessThanOrEqual(512)
  }, 30_000)

  test("labels accumulate across once, always and reject", async () => {
    for (const [id, reply] of [
      ["per_a", "once"],
      ["per_b", "always"],
      ["per_c", "reject"],
    ] as const) {
      const result = await evaluate(
        { id, sessionID: "ses_e2e", permission: "shell", patterns: ["ls"], metadata: { command: "ls -la" } },
        { client: () => sidecar.current(), shadow, context: async () => ({ task: "look around" }) },
      )
      expect(result.evaluated).toBe(true)
      expect(await shadow.replied(id, reply)).toBe(true)
    }

    const files = await fs.readdir(dir)
    const rows = (await fs.readFile(path.join(dir, files[0]!), "utf8"))
      .split("\n")
      .filter(Boolean)
      .map((l) => JSON.parse(l))
      .filter((r) => r.kind === "decision")

    const byReply = Object.fromEntries(rows.map((r) => [r.reply, JSON.parse(r.gold).auto_approve.probabilities]))
    expect(byReply["once"]).toEqual({ true: 1, false: 0 })
    expect(byReply["always"]).toEqual({ true: 1, false: 0 })
    expect(byReply["reject"]).toEqual({ true: 0, false: 1 })

    const approvals = rows.filter((r) => r.reply && r.reply !== "reject").length
    console.log(`  class balance so far: ${approvals} approve / ${rows.length - approvals} reject`)
  }, 60_000)

  test("a stopped sidecar degrades to no judgment, not an approval", async () => {
    await sidecar.stop()
    const result = await evaluate(
      { id: "per_down", sessionID: "ses_e2e", permission: "shell", patterns: ["ls"], metadata: { command: "ls" } },
      { client: () => sidecar.current(), shadow },
    )
    expect(result.evaluated).toBe(false)
    if (!result.evaluated) expect(result.reason).toBe("unavailable")
  }, 30_000)
})
