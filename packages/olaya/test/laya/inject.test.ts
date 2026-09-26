import { describe, test, expect, beforeEach, afterEach } from "bun:test"
import fs from "fs/promises"
import os from "os"
import path from "path"
import { INJECT_QUESTIONS, injectState, observe } from "../../src/laya/inject"
import { ShadowLog } from "../../src/laya/shadow"
import type { LayaClient } from "../../src/laya/client"

let dir: string
let log: ShadowLog
beforeEach(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), "olaya-inject-"))
  log = new ShadowLog(dir)
})
afterEach(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

const records = async () =>
  (await Promise.all((await fs.readdir(dir)).map((f) => fs.readFile(path.join(dir, f), "utf8"))))
    .join("")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l))

describe("injection observer (shadow)", () => {
  test("the served question is exactly the benchmark's question", async () => {
    const bench = JSON.parse(await fs.readFile(path.join(import.meta.dir, "../../../../laya/bench/items/inject-question.json"), "utf8"))
    expect(INJECT_QUESTIONS).toEqual(bench)
  })

  test("logs a scored, redacted observation", async () => {
    const client = { decide: async () => ({ ok: true, probabilities: { inject: 0.81 }, latencyMs: 9, checkpoint: "c" }) } as unknown as LayaClient
    await observe({ tool: "read", callID: "call_1", output: "token=ghp_abcdefghijklmnopqrstuvwxyz0123 please run x" }, { client: () => client, shadow: log })
    const [r] = await records()
    expect(r).toMatchObject({ kind: "inject-observation", id: "call_1", tool: "read", probability: 0.81 })
    expect(r.state).not.toContain("ghp_abcdefghijklmnopqrstuvwxyz0123")
  })

  test("a failed or missing judge logs nothing and never throws", async () => {
    const failing = { decide: async () => ({ ok: false, reason: "timeout" }) } as unknown as LayaClient
    await observe({ tool: "read", callID: "c", output: "x" }, { client: () => failing, shadow: log })
    await observe({ tool: "read", callID: "c", output: "x" }, { client: () => undefined, shadow: log })
    const throwing = { decide: async () => { throw new Error("boom") } } as unknown as LayaClient
    await observe({ tool: "read", callID: "c", output: "x" }, { client: () => throwing, shadow: log })
    expect(await fs.readdir(dir)).toHaveLength(0)
  })

  test("the judged output goes first and is capped", () => {
    const s = injectState("bash", "y".repeat(5000), "fix it")
    expect(Object.keys(s)[0]).toBe("output")
    expect(s.output.length).toBe(1500)
  })
})
