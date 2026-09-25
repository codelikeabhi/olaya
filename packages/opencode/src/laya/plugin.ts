/**
 * Olaya's decision-layer plugin.
 *
 * Disabled by default. With `OLAYA_LAYA_ENABLED` unset this returns no hooks at all, so
 * opencode behaves exactly as upstream and nothing is spawned, called, or written.
 */

import path from "path"
import { Global } from "@opencode-ai/core/global"
import type { Plugin } from "@opencode-ai/plugin"
import { resolve } from "./config"
import { Sidecar } from "./sidecar"
import { ShadowLog, type Reply } from "./shadow"
import { shadowHandler } from "./judgment"
import type { SessionContext } from "./state"

export const SHADOW_DIR = path.join(Global.Path.data, "laya-shadow")

/**
 * Remembers each session's originating request, fed by the `chat.message` hook.
 *
 * This deliberately does NOT call back into opencode's API. The permission.ask handler runs
 * *inside* `Permission.ask()`, before the request is published and while the tool call is
 * blocked on it. Making a server round trip from there re-enters opencode from within a path
 * it is itself awaiting - a stall there stalls the permission, and no timeout makes that
 * shape safe. Reading a value populated earlier by another hook has no such hazard.
 */
class TaskMemory {
  private tasks = new Map<string, string>()

  /** Records the first user message of a session; later messages do not overwrite it. */
  remember(sessionID: string, parts: unknown): void {
    if (!sessionID || this.tasks.has(sessionID)) return
    const text = (Array.isArray(parts) ? parts : [])
      .filter((p: any) => p?.type === "text" && typeof p.text === "string")
      .map((p: any) => p.text)
      .join(" ")
      .trim()
    // ponytail: unbounded in principle, one short string per session in practice. Cap it if a
    // long-lived server ever accumulates enough sessions for it to matter.
    if (text) this.tasks.set(sessionID, text)
  }

  get(sessionID: string): string | undefined {
    return this.tasks.get(sessionID)
  }
}

export const LayaPlugin: Plugin = async (input, options) => {
  const config = resolve((options ?? {}) as Record<string, unknown>)
  if (!config.enabled) return {}

  const sidecar = new Sidecar(config)
  await sidecar.start()

  const shadow = config.shadow ? new ShadowLog(process.env["OLAYA_LAYA_SHADOW_DIR"] ?? SHADOW_DIR) : undefined
  const memory = new TaskMemory()
  const handler = shadowHandler({
    client: () => sidecar.current(),
    shadow,
    // Synchronous lookup, no I/O: see TaskMemory.
    context: async ({ sessionID }) => {
      const task = memory.get(sessionID)
      return { cwd: input.directory, ...(task ? { task } : {}) }
    },
  })

  return {
    "permission.ask": handler as never,

    // Capture the user's request as it arrives, so the permission path never has to ask for it.
    "chat.message": async (msg, out) => {
      memory.remember(msg.sessionID, (out as { parts?: unknown }).parts)
    },

    // The label join. A reply is the ground truth for the prediction logged moments earlier,
    // which is what makes shadow mode a training-data pipeline and not just telemetry.
    event: async ({ event }) => {
      if (!shadow) return
      if (event.type !== "permission.replied") return
      const props = (event as { properties?: { requestID?: string; reply?: Reply } }).properties
      if (props?.requestID && props.reply) await shadow.replied(props.requestID, props.reply)
    },

    dispose: async () => {
      // Predictions the user never answered are still worth keeping, marked unlabelled.
      await shadow?.flushUnlabelled()
      await sidecar.stop()
    },
  }
}
