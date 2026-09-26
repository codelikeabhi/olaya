/**
 * Olaya's decision-layer plugin.
 *
 * Disabled by default. With `OLAYA_LAYA_ENABLED` unset this returns no hooks at all, so
 * olaya behaves exactly as upstream and nothing is spawned, called, or written.
 */

import path from "path"
import { Global } from "@olaya/core/global"
import type { Plugin } from "@olaya/plugin"
import { resolve } from "./config"
import { Sidecar } from "./sidecar"
import { ShadowLog, type Reply } from "./shadow"
import { liveHandler, shadowHandler } from "./judgment"
import { observe } from "./inject"
import type { SessionContext } from "./state"
import { handler as retentionHandler, recallTool } from "./retention"

export const SHADOW_DIR = path.join(Global.Path.data, "laya-shadow")
/** Full text of what live retention shortened or dropped, one append-only file per session. */
export const RECALL_DIR = path.join(Global.Path.data, "laya-recall")

/**
 * Whether permission replies in this process come from auto-approval rather than a person.
 * Benchmark harnesses run with these flags, so every reply they log is the tool answering itself.
 */
const AUTO_REPLIES = ["--auto", "--yolo", "--dangerously-skip-permissions"].some((flag) => process.argv.includes(flag))

/**
 * Remembers each session's originating request, fed by the `chat.message` hook.
 *
 * This deliberately does NOT call back into olaya's API. The permission.ask handler runs
 * *inside* `Permission.ask()`, before the request is published and while the tool call is
 * blocked on it. Making a server round trip from there re-enters olaya from within a path
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

// ponytail: a fixed ceiling on what retention keeps; tune from the G9 A/B.
const RETENTION_CAP_TOKENS = 24_000

export const LayaPlugin: Plugin = async (input, options) => {
  const config = resolve((options ?? {}) as Record<string, unknown>)
  const shadowDir = process.env["OLAYA_LAYA_SHADOW_DIR"] ?? SHADOW_DIR
  const retention =
    config.retention === "off"
      ? {}
      : {
          "experimental.session.retention": retentionHandler({
            mode: config.retention,
            budget: config.retentionBudget,
            cap: RETENTION_CAP_TOKENS,
            shadow: new ShadowLog(shadowDir),
            recallDir: RECALL_DIR,
          }),
          ...(config.retention === "live" && { tool: { recall: recallTool(RECALL_DIR) } }),
        }
  if (!config.enabled) return retention

  const sidecar = new Sidecar(config)
  await sidecar.start()

  // Live mode always keeps the local log: every permission it grants must leave an audit record.
  const live = config.mode === "live"
  const shadow = config.shadow || live ? new ShadowLog(shadowDir) : undefined
  const memory = new TaskMemory()
  const deps = {
    client: () => sidecar.current(),
    shadow,
    // Synchronous lookup, no I/O: see TaskMemory.
    context: async ({ sessionID }: { sessionID: string }) => {
      const task = memory.get(sessionID)
      return { cwd: input.directory, ...(task ? { task } : {}) }
    },
  }
  // Shadow mode's handler cannot change a permission by construction (it never sees the
  // output). Live mode's can, within the limits enforced in liveHandler.
  const handler = live ? liveHandler(deps) : shadowHandler(deps)

  return {
    ...retention,
    "permission.ask": handler as never,

    // Tier 2 (shadow only): score tool outputs for injection. Scheduled, not awaited, so the
    // agent's next step never waits on it; it cannot modify the output.
    "tool.execute.after": async (call, result) => {
      if (!config.shadow || process.env["OLAYA_LAYA_OBSERVE_OUTPUTS"] === "0") return
      void observe(
        { tool: call.tool, callID: call.callID, output: (result as { output?: unknown })?.output ?? result, task: memory.get(call.sessionID) },
        { client: () => sidecar.current(), shadow },
      )
    },

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
      if (props?.requestID && props.reply) await shadow.replied(props.requestID, props.reply, AUTO_REPLIES ? "auto" : "user")
    },

    dispose: async () => {
      // Predictions the user never answered are still worth keeping, marked unlabelled.
      await shadow?.flushUnlabelled()
      await sidecar.stop()
    },
  }
}
