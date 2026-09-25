/**
 * Turns a permission request into a Laya state.
 *
 * The ordering rule here is not stylistic. `build_sequence` fills the sequence with
 * `state[:room]` - it keeps the head and silently discards the tail - and `system_one` does
 * not expose the `truncate_left` flag that would reverse it. So whatever matters most must
 * be serialised first, and anything droppable must come last.
 */

import type { Question } from "./client"

export const AUTO_APPROVE = "auto_approve"

/**
 * v1 asks exactly one binary question. `noul` is the model's strongest primitive; `score`
 * is its weakest (SST-5 0.372) and is deliberately unused. One question also means one
 * threshold to calibrate, and one forward pass.
 */
export const QUESTIONS: Record<string, Question> = {
  [AUTO_APPROVE]: {
    type: "noul",
    instructions: "Is it safe to perform this action without asking the user for confirmation first?",
  },
}

export interface PermissionRequest {
  permission: string
  patterns: readonly string[]
  metadata?: Record<string, unknown>
}

export interface SessionContext {
  /** What the user actually asked for. Absent when it cannot be resolved. */
  task?: string
  /** Recent agent activity, most recent last. Purely contextual, first to be dropped. */
  recent?: readonly string[]
  cwd?: string
}

export type Compaction =
  | { ok: true; state: Record<string, unknown>; estimatedTokens: number; dropped: readonly string[] }
  | { ok: false; reason: "non-english" }

/**
 * ponytail: characters/3 stands in for the real tokenizer, which lives in Python. It is a
 * deliberate over-estimate of token count - code, paths and punctuation tokenize denser than
 * prose - so we trim early rather than overflow. Upgrade path: the sidecar returns
 * `usage.input_tokens` on every call, so a measured ratio can replace this constant.
 */
const CHARS_PER_TOKEN = 3
const estimate = (state: unknown): number => Math.ceil(JSON.stringify(state).length / CHARS_PER_TOKEN)

/** Dropped in this order when the state does not fit. `action` and its subject are never dropped. */
const DROPPABLE = ["recent", "task", "cwd", "patterns"] as const

const MAX_COMMAND_CHARS = 600
const MAX_PATTERNS = 8
const MAX_FILES = 10
const MAX_RECENT = 6

function truncate(value: string, max: number): string {
  return value.length <= max ? value : value.slice(0, max) + "…"
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined
}

/**
 * Reduce a unified diff to what a judgment actually needs: which files, how much churn.
 * A real diff is far larger than the entire state budget, and inlining one would push the
 * user's request out of the window entirely.
 */
export function summariseDiff(diff: string): { added: number; removed: number } {
  let added = 0
  let removed = 0
  for (const line of diff.split("\n")) {
    if (line.startsWith("+++") || line.startsWith("---")) continue
    if (line.startsWith("+")) added++
    else if (line.startsWith("-")) removed++
  }
  return { added, removed }
}

/**
 * Whether the state is readable by the English checkpoint.
 *
 * This gate exists because the English model does not degrade gracefully outside English -
 * it collapses and stays confident doing so (Khmer: 0.000 accuracy at 0.952 confidence, with
 * mean confidence never below 0.885). No downstream confidence threshold can catch that, so
 * the only safe move is to never make the call.
 */
export function looksEnglish(text: string): boolean {
  const letters = text.match(/\p{L}/gu)
  if (!letters || letters.length < 20) return true // too little signal to judge; let it through
  let foreign = 0
  for (const ch of letters) if (!/[A-Za-z]/.test(ch)) foreign++
  return foreign / letters.length <= 0.15
}

/** The decisive subject of the request, by permission kind. Always first after `action`. */
function subject(req: PermissionRequest): Record<string, unknown> {
  const meta = req.metadata ?? {}
  const command = str(meta["command"])
  if (command) return { command: truncate(command, MAX_COMMAND_CHARS) }

  const url = str(meta["url"])
  if (url) return { url: truncate(url, MAX_COMMAND_CHARS) }

  const diff = str(meta["diff"])
  const filepath = str(meta["filepath"])
  if (diff || filepath) {
    const out: Record<string, unknown> = {}
    if (filepath) out["files"] = filepath.split(", ").slice(0, MAX_FILES)
    if (diff) out["changes"] = summariseDiff(diff)
    return out
  }
  return {}
}

export function compact(
  req: PermissionRequest,
  ctx: SessionContext,
  budgetTokens: number,
): Compaction {
  const state: Record<string, unknown> = { action: req.permission, ...subject(req) }

  if (req.patterns.length) state["patterns"] = req.patterns.slice(0, MAX_PATTERNS)
  if (ctx.cwd) state["cwd"] = ctx.cwd
  // Omitted rather than filled with a placeholder: a fabricated intent is worse than a
  // missing one, because the model cannot tell the difference.
  if (ctx.task) state["task"] = truncate(ctx.task, MAX_COMMAND_CHARS)
  if (ctx.recent?.length) state["recent"] = ctx.recent.slice(-MAX_RECENT)

  if (!looksEnglish(JSON.stringify(state))) return { ok: false, reason: "non-english" }

  const dropped: string[] = []
  for (const key of DROPPABLE) {
    if (estimate(state) <= budgetTokens) break
    if (!(key in state)) continue
    delete state[key]
    dropped.push(key)
  }

  // Everything droppable is gone and it still does not fit: the subject itself is oversized,
  // so shrink it rather than hand the model a silently truncated command.
  if (estimate(state) > budgetTokens && typeof state["command"] === "string") {
    const room = Math.max(40, budgetTokens * CHARS_PER_TOKEN - 120)
    state["command"] = truncate(state["command"] as string, room)
  }

  return { ok: true, state, estimatedTokens: estimate(state), dropped }
}
