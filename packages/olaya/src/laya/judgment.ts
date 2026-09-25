/**
 * The permission.ask handler.
 *
 * In this change it only ever observes. The shadow handler's signature does not bind the
 * hook's `output` parameter at all, so there is no expression anywhere in this file that
 * could change a permission outcome - a stronger guarantee than a flag defaulting to off.
 */

import type { LayaClient } from "./client"
import type { ShadowLog, RefusalReason } from "./shadow"
import { compact, QUESTIONS, AUTO_APPROVE, type PermissionRequest, type SessionContext } from "./state"

/** Conservative fallback when the sidecar has not reported a checkpoint budget yet. */
const FALLBACK_BUDGET = 300

/**
 * Actions that always go to the user, whatever the model thinks.
 *
 * The decision layer is a convenience, not the only safeguard. These are the cases where
 * being wrong is unrecoverable, so they are never delegated to a probability.
 */
export const DENYLIST: { name: string; pattern: RegExp }[] = [
  { name: "recursive-delete-of-root-or-home", pattern: /\brm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rR][a-zA-Z]*f?\s+(\/|~|\$HOME)(\s|$)/ },
  { name: "force-push", pattern: /\bgit\s+push\b[^\n]*(--force(?!-with-lease)|\s-f\b)/ },
  { name: "history-rewrite", pattern: /\bgit\s+(filter-branch|filter-repo)\b/ },
  { name: "piped-installer", pattern: /\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba|z|k|)sh\b/ },
  { name: "credential-path", pattern: /(^|[\s"'\/])(\.env(\.[\w-]+)?|\.npmrc|\.netrc|id_rsa|id_ed25519|credentials)(\s|$|["'])/ },
  { name: "ssh-or-cloud-secrets", pattern: /(^|[\s"'])(~|\$HOME)?\/?\.(ssh|aws|gnupg)\// },
  { name: "raw-device-write", pattern: /\b(dd\s+[^\n]*of=\/dev\/|mkfs(\.\w+)?\s)/ },
  { name: "fork-bomb", pattern: /:\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:/ },
]

/** Text a denylist rule should be matched against. */
function surface(req: PermissionRequest): string {
  const meta = req.metadata ?? {}
  const parts = [req.permission, ...req.patterns]
  for (const key of ["command", "url", "filepath"]) {
    const value = meta[key]
    if (typeof value === "string") parts.push(value)
  }
  const directories = meta["directories"]
  if (Array.isArray(directories)) parts.push(...directories.filter((d): d is string => typeof d === "string"))
  return parts.join("\n")
}

export function denylisted(req: PermissionRequest): string | undefined {
  const text = surface(req)
  for (const rule of DENYLIST) if (rule.pattern.test(text)) return rule.name
  return undefined
}

export interface JudgmentDeps {
  client: () => LayaClient | undefined
  shadow?: ShadowLog
  /** Resolves the user's originating request for a session. Returns {} when unavailable. */
  context?: (input: { sessionID: string }) => Promise<SessionContext>
}

/** What a judgment attempt produced. Returned for tests and diagnostics; never acted upon. */
export type Outcome =
  | { evaluated: true; probability: number }
  | { evaluated: false; reason: RefusalReason }

export async function evaluate(input: any, deps: JudgmentDeps): Promise<Outcome> {
  const req: PermissionRequest = {
    permission: input?.permission ?? "unknown",
    patterns: Array.isArray(input?.patterns) ? input.patterns : [],
    metadata: input?.metadata ?? {},
  }
  const id = String(input?.id ?? "")

  // Denylist first: these never reach the compactor, let alone the model.
  const blocked = denylisted(req)
  if (blocked) {
    await deps.shadow?.refused({ id, action: req.permission, reason: "denylisted" })
    return { evaluated: false, reason: "denylisted" }
  }

  const client = deps.client()
  if (!client) {
    await deps.shadow?.refused({ id, action: req.permission, reason: "unavailable" })
    return { evaluated: false, reason: "unavailable" }
  }

  const context = (await deps.context?.({ sessionID: String(input?.sessionID ?? "") })) ?? {}
  const budget = (await client.stateBudget()) ?? FALLBACK_BUDGET
  const compacted = compact(req, context, budget)
  if (!compacted.ok) {
    await deps.shadow?.refused({ id, action: req.permission, reason: compacted.reason })
    return { evaluated: false, reason: compacted.reason }
  }

  const judgment = await client.decide(compacted.state, QUESTIONS)
  if (!judgment.ok) {
    await deps.shadow?.refused({ id, action: req.permission, reason: judgment.reason as RefusalReason })
    return { evaluated: false, reason: judgment.reason as RefusalReason }
  }

  const probability = judgment.probabilities[AUTO_APPROVE]!
  deps.shadow?.predicted({
    id,
    ts: new Date().toISOString(),
    state: compacted.state,
    questions: QUESTIONS,
    probability,
    checkpoint: judgment.checkpoint,
    latencyMs: judgment.latencyMs,
    estimatedTokens: compacted.estimatedTokens,
    inputTokens: judgment.inputTokens,
  })
  return { evaluated: true, probability }
}

/**
 * The hook handler for shadow mode.
 *
 * Note the single parameter. The hook is typed `(input, output) => Promise<void>`, and a
 * one-parameter function satisfies it - so `output` is not merely left alone, it is not in
 * scope. No future edit inside this closure can reach a permission outcome by accident.
 */
export function shadowHandler(deps: JudgmentDeps): (input: unknown) => Promise<void> {
  return async (input) => {
    try {
      await evaluate(input, deps)
    } catch {
      // A judgment is advisory. It must never surface an error into the permission path.
    }
  }
}
