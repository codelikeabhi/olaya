import { describe, test, expect } from "bun:test"
import { LayaClient, type Question } from "../../src/laya/client"
import { resolve } from "../../src/laya/config"

const QUESTIONS: Record<string, Question> = {
  auto_approve: { type: "noul", instructions: "Is it safe to run this without asking?" },
}

/**
 * Runs `body` against a throwaway loopback server and always tears it down. Connections are not
 * kept alive: under load the next test's server can get the same port, and a pooled connection
 * would reach the old handler.
 */
async function withServer(
  handler: (req: Request) => Response | Promise<Response>,
  body: (url: string) => Promise<void>,
) {
  const server = Bun.serve({
    port: 0,
    fetch: async (req) => {
      const res = await handler(req)
      res.headers.set("Connection", "close")
      return res
    },
  })
  try {
    await body(`http://127.0.0.1:${server.port}`)
  } finally {
    server.stop(true)
  }
}

const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } })

const ok = (probability: number) =>
  json({
    answers: { auto_approve: { type: "noul", probability } },
    latency_ms: 12,
    checkpoint: "test",
    usage: { input_tokens: 94 },
  })

describe("LayaClient", () => {
  test("a good response yields a probability", async () => {
    await withServer(
      () => ok(0.87),
      async (url) => {
        const result = await new LayaClient(url, 5000).decide({ action: "shell" }, QUESTIONS)
        expect(result.ok).toBe(true)
        if (result.ok) {
          expect(result.probabilities.auto_approve).toBe(0.87)
          expect(result.checkpoint).toBe("test")
          expect(result.inputTokens).toBe(94)
        }
      },
    )
  })

  // The core safety property: no failure mode may ever present as an approval.
  test.each([
    ["not-ready", () => json({ error: "model not ready" }, 503), "not-ready"],
    ["server error", () => json({ error: "boom" }, 500), "http"],
    ["invalid JSON", () => new Response("{not json", { status: 200 }), "malformed"],
    ["missing answers", () => json({ latency_ms: 1 }), "malformed"],
    ["answer for another question", () => json({ answers: { something_else: { probability: 0.9 } } }), "malformed"],
    ["non-numeric probability", () => json({ answers: { auto_approve: { probability: "yes" } } }), "malformed"],
    ["probability above 1", () => json({ answers: { auto_approve: { probability: 1.4 } } }), "out-of-range"],
    ["probability below 0", () => json({ answers: { auto_approve: { probability: -0.2 } } }), "out-of-range"],
  ])("%s yields no judgment", async (_label, handler, reason) => {
    await withServer(handler as () => Response, async (url) => {
      const result = await new LayaClient(url, 5000).decide({}, QUESTIONS)
      expect(result.ok).toBe(false)
      if (!result.ok) expect(result.reason).toBe(reason as never)
    })
  })

  test("a slow sidecar times out rather than stalling the caller", async () => {
    await withServer(
      async () => {
        await Bun.sleep(300)
        return ok(0.99)
      },
      async (url) => {
        const started = Date.now()
        const result = await new LayaClient(url, 60).decide({}, QUESTIONS)
        expect(result.ok).toBe(false)
        if (!result.ok) expect(result.reason).toBe("timeout")
        expect(Date.now() - started).toBeLessThan(250)
      },
    )
  })

  test("an unreachable sidecar yields no judgment", async () => {
    // Bind then immediately release, so the port is almost certainly closed.
    const probe = Bun.serve({ port: 0, fetch: () => new Response("") })
    const url = `http://127.0.0.1:${probe.port}`
    probe.stop(true)
    const result = await new LayaClient(url, 200).decide({}, QUESTIONS)
    expect(result.ok).toBe(false)
  })

  test("state budget is read from health and only cached once ready", async () => {
    let ready = false
    await withServer(
      (req) =>
        new URL(req.url).pathname === "/health"
          ? json(ready ? { ready: true, checkpoint: "c", state_budget: 316 } : { ready: false, checkpoint: "c" })
          : ok(0.5),
      async (url) => {
        const client = new LayaClient(url, 5000)
        expect(await client.stateBudget()).toBeUndefined()
        ready = true
        expect(await client.stateBudget()).toBe(316)
      },
    )
  })
})

describe("config", () => {
  test("defaults are off", () => {
    const cfg = resolve({}, {})
    expect(cfg.enabled).toBe(false)
    expect(cfg.shadow).toBe(false)
  })

  test("shadow cannot be on while the layer is off", () => {
    expect(resolve({}, { OLAYA_LAYA_SHADOW: "1" }).shadow).toBe(false)
    expect(resolve({}, { OLAYA_LAYA_SHADOW: "1", OLAYA_LAYA_ENABLED: "1" }).shadow).toBe(true)
  })

  test("plugin options beat the environment", () => {
    const cfg = resolve({ timeoutMs: 900 }, { OLAYA_LAYA_ENABLED: "1", OLAYA_LAYA_TIMEOUT_MS: "100" })
    expect(cfg.timeoutMs).toBe(900)
    expect(cfg.enabled).toBe(true)
  })

  test("a nonsense timeout falls back to the default", () => {
    expect(resolve({}, { OLAYA_LAYA_TIMEOUT_MS: "-5" }).timeoutMs).toBe(1500)
  })
})
