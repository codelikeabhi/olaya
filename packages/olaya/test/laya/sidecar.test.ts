import { describe, test, expect, afterEach } from "bun:test"
import path from "path"
import { Sidecar } from "../../src/laya/sidecar"
import { resolve } from "../../src/laya/config"

const LAYA_DIR = path.resolve(import.meta.dir, "../../../../laya")

let started: Sidecar[] = []
afterEach(async () => {
  await Promise.all(started.map((s) => s.stop()))
  started = []
})

function track(s: Sidecar) {
  started.push(s)
  return s
}

describe("Sidecar", () => {
  test("disabled config starts nothing", async () => {
    const s = track(new Sidecar(resolve({ enabled: false })))
    await s.start()
    expect(s.status).toBe("stopped")
    expect(s.current()).toBeUndefined()
  })

  test("an external url is attached to, not spawned", async () => {
    const s = track(new Sidecar(resolve({ enabled: true, url: "http://127.0.0.1:9" })))
    await s.start()
    expect(s.status).toBe("external")
    expect(s.current()).toBeDefined()
  })

  test("spawns the service, learns its port, and reports not-ready while the model loads", async () => {
    // The system python has no `laya` installed, which is exactly the shape we need: the
    // service binds and announces its port, then the background load fails. That is the
    // same sequence as a real cold start, minus the 21 s wait.
    process.env["OLAYA_LAYA_DIR"] = LAYA_DIR
    const s = track(new Sidecar(resolve({ enabled: true, python: "python3", timeoutMs: 2000 })))
    await s.start()

    expect(s.status).toBe("running")
    const client = s.current()
    expect(client).toBeDefined()

    const health = await client!.health(2000)
    expect(health).toBeDefined()
    expect(health!.ready).toBe(false)

    // A not-ready sidecar must never answer with a probability.
    const judgment = await client!.decide({ action: "shell" }, { q: { type: "noul", instructions: "safe?" } })
    expect(judgment.ok).toBe(false)
    if (!judgment.ok) expect(judgment.reason).toBe("not-ready")
  })

  test("stop reaps the process and leaves no client behind", async () => {
    process.env["OLAYA_LAYA_DIR"] = LAYA_DIR
    const s = new Sidecar(resolve({ enabled: true, python: "python3", timeoutMs: 2000 }))
    await s.start()
    expect(s.status).toBe("running")
    const url = (s.current() as unknown as { baseUrl: string }).baseUrl

    await s.stop()
    expect(s.status).toBe("stopped")
    expect(s.current()).toBeUndefined()

    // The port is closed, so nothing is still listening.
    await expect(fetch(new URL("/health", url), { signal: AbortSignal.timeout(500) })).rejects.toThrow()
  })

  test("a missing interpreter gives up instead of looping", async () => {
    process.env["OLAYA_LAYA_DIR"] = LAYA_DIR
    const s = track(new Sidecar(resolve({ enabled: true, python: "definitely-not-a-python-abc123" })))
    await s.start()
    expect(s.status).toBe("given-up")
    expect(s.current()).toBeUndefined()
  })
})
