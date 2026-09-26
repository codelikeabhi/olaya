/**
 * Lifecycle for the Laya decision sidecar.
 *
 * Loading a checkpoint takes ~21 s, so the process is started once at launch and kept.
 * Olaya never blocks on it: until the model reports ready, every judgment request simply
 * returns no judgment and the permission flows to the user.
 */

import { launch, sourceDir } from "./runtime"
import os from "os"
import path from "path"
import type { LayaConfig } from "./config"
import { LayaClient } from "./client"

/**
 * After this many crashes the layer stays down for the session. A reproducible crash would
 * otherwise become a restart loop paying a 21 s model load each time, and shadow mode has no
 * user-visible symptom that would make the loop obvious.
 */
const MAX_RESTARTS = 3
const BACKOFF_MS = [1_000, 4_000, 10_000]
const LISTENING_PREFIX = "OLAYA_SIDECAR_LISTENING "

export type SidecarStatus = "stopped" | "starting" | "running" | "external" | "given-up"

export class Sidecar {
  private proc?: ReturnType<typeof Bun.spawn>
  private client?: LayaClient
  private restarts = 0
  private stopping = false
  status: SidecarStatus = "stopped"
  lastError?: string

  constructor(private readonly config: LayaConfig) {}

  /** Client for the running sidecar, or undefined if there is not one yet. */
  current(): LayaClient | undefined {
    return this.client
  }

  async start(): Promise<void> {
    if (!this.config.enabled) return
    // An externally managed sidecar: attach, never supervise. A developer running it by
    // hand owns its lifetime, and killing their process on our exit would be rude.
    if (this.config.url) {
      this.client = new LayaClient(this.config.url, this.config.timeoutMs)
      this.status = "external"
      return
    }
    await this.spawn()
  }

  private async spawn(): Promise<void> {
    this.status = "starting"
    const env: Record<string, string> = { ...(process.env as Record<string, string>), OLAYA_LAYA_PORT: "0" }
    if (this.config.checkpoint) env["OLAYA_LAYA_MODEL"] = this.config.checkpoint
    // Pin the model cache instead of inheriting it. huggingface_hub derives its cache from
    // XDG_CACHE_HOME, so any caller that remaps XDG - olaya's own test preload does -
    // would point the sidecar at an empty directory and silently re-download 848 MB. The
    // checkpoint belongs in one place per machine, independent of ambient XDG state.
    env["HF_HOME"] = process.env["OLAYA_LAYA_HF_HOME"] ?? process.env["HF_HOME"] ?? path.join(os.homedir(), ".cache", "huggingface")

    let proc: ReturnType<typeof Bun.spawn>
    try {
      const { argv, cwd } = launch(this.config.python, process.env, sourceDir())
      proc = Bun.spawn(argv, {
        cwd,
        env,
        stdout: "pipe",
        // Not inherited: this runs under a TUI, and a Python traceback written straight to the
        // terminal would corrupt the display. Load failures are reported by /health instead.
        // OLAYA_LAYA_STDERR routes it to a file when you need the detail health cannot carry.
        stderr: process.env["OLAYA_LAYA_STDERR"] ? Bun.file(process.env["OLAYA_LAYA_STDERR"]) : "ignore",
        onExit: () => this.onExit(),
      })
    } catch (error) {
      this.lastError = error instanceof Error ? error.message : String(error)
      this.status = "given-up"
      return
    }
    this.proc = proc

    const port = await this.readPort(proc)
    if (port === undefined) {
      this.lastError = "sidecar did not announce a port"
      // onExit drives the restart decision; if it is still alive but silent, stop it so the
      // process cannot linger unreferenced.
      proc.kill()
      return
    }
    this.client = new LayaClient(`http://127.0.0.1:${port}`, this.config.timeoutMs)
    this.status = "running"
  }

  /**
   * Wait for the port announcement. Binding happens before the model loads, so this
   * resolves in milliseconds and does not wait out the ~21 s load.
   */
  private async readPort(proc: ReturnType<typeof Bun.spawn>): Promise<number | undefined> {
    const stdout = proc.stdout
    if (!stdout || typeof stdout === "number") return undefined
    const reader = (stdout as ReadableStream<Uint8Array>).getReader()
    // Raced against the read rather than checked per chunk: a sidecar that binds nothing and
    // writes nothing would otherwise never wake this loop, and we would wait forever.
    let timer: ReturnType<typeof setTimeout> | undefined
    const expired = new Promise<"expired">((resolve) => {
      timer = setTimeout(() => resolve("expired"), 10_000)
    })
    const decoder = new TextDecoder()
    let buffered = ""
    try {
      while (true) {
        const next = await Promise.race([reader.read(), expired])
        if (next === "expired") return undefined
        const { done, value } = next
        if (done) return undefined
        buffered += decoder.decode(value, { stream: true })
        const lines = buffered.split("\n")
        buffered = lines.pop() ?? ""
        for (const line of lines) {
          if (!line.startsWith(LISTENING_PREFIX)) continue
          try {
            const port = JSON.parse(line.slice(LISTENING_PREFIX.length)).port
            if (typeof port === "number") return port
          } catch {
            // a malformed announcement is not fatal; keep reading
          }
        }
      }
    } finally {
      if (timer) clearTimeout(timer)
      reader.releaseLock()
    }
  }

  private onExit(): void {
    this.client = undefined
    this.proc = undefined
    if (this.stopping || this.status === "external") {
      this.status = "stopped"
      return
    }
    if (this.restarts >= MAX_RESTARTS) {
      this.status = "given-up"
      this.lastError = `sidecar exited ${this.restarts} times; decision layer is down for this session`
      return
    }
    const delay = BACKOFF_MS[Math.min(this.restarts, BACKOFF_MS.length - 1)]!
    this.restarts += 1
    this.status = "starting"
    setTimeout(() => {
      if (!this.stopping) void this.spawn()
    }, delay).unref?.()
  }

  async stop(): Promise<void> {
    this.stopping = true
    this.status = "stopped"
    this.client = undefined
    const proc = this.proc
    this.proc = undefined
    if (!proc) return
    proc.kill()
    // Reap it, so exiting Olaya never leaves an orphan holding the checkpoint in memory.
    await proc.exited
  }
}
