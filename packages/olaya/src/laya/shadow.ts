/**
 * Shadow-mode recording: what the model predicted, and what the user actually decided.
 *
 * Records are written in Laya's own training schema - `state`, `questions` and `gold` as
 * JSON strings - so the fine-tuning step needs a split and a filter rather than a converter.
 * Extra keys sit alongside and are ignored by the trainer.
 *
 * Nothing here runs unless shadow mode is explicitly enabled.
 */

import fs from "fs/promises"
import path from "path"
import type { Question } from "./client"

export type Reply = "once" | "always" | "reject"
export type RefusalReason = "non-english" | "denylisted" | "unavailable" | "not-ready" | "timeout" | "transport" | "malformed" | "out-of-range" | "http"

/** A prediction awaiting the user's reply. */
interface Pending {
  id: string
  ts: string
  state: unknown
  questions: Record<string, Question>
  probability: number
  checkpoint: string
  latencyMs: number
  estimatedTokens?: number
  inputTokens?: number
}

/**
 * ponytail: a short regex list, not a secret scanner. It covers the shapes that actually
 * show up in shell commands and diffs. Upgrade path: swap in a real detector (gitleaks
 * rules, detect-secrets) if a leak is ever observed in practice.
 */
const REDACTIONS: [RegExp, string][] = [
  [/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g, "«redacted-private-key»"],
  [/\b(gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,})\b/g, "«redacted-token»"],
  [/\b(sk-[A-Za-z0-9-_]{16,}|xox[baprs]-[A-Za-z0-9-]{10,})\b/g, "«redacted-token»"],
  [/\bAKIA[0-9A-Z]{16}\b/g, "«redacted-aws-key»"],
  [/\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g, "«redacted-jwt»"],
  [/(?<=\b(?:Authorization|Bearer)\s+)\S+/gi, "«redacted»"],
  // key=value and key: value where the key names a secret
  [/((?:password|passwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key)\s*[=:]\s*)(?:"[^"]*"|'[^']*'|\S+)/gi, "$1«redacted»"],
]

export function redact<T>(value: T): T {
  let text = JSON.stringify(value)
  if (text === undefined) return value
  for (const [pattern, replacement] of REDACTIONS) text = text.replace(pattern, replacement)
  return JSON.parse(text) as T
}

/** A reply is one-hot over the same two options a teacher label is soft over. */
function gold(reply: Reply, questionId: string) {
  const approved = reply !== "reject"
  return { [questionId]: { probabilities: { true: approved ? 1 : 0, false: approved ? 0 : 1 } } }
}

export class ShadowLog {
  private pending = new Map<string, Pending>()

  constructor(private readonly dir: string) {}

  private file(): string {
    // One file per day keeps any single file browsable and makes retention a matter of
    // deleting whole files.
    return path.join(this.dir, `shadow-${new Date().toISOString().slice(0, 10)}.jsonl`)
  }

  private async append(record: Record<string, unknown>): Promise<void> {
    // Redact before the record can reach disk - never after, and never conditionally.
    const safe = redact(record)
    await fs.mkdir(this.dir, { recursive: true })
    await fs.appendFile(this.file(), JSON.stringify(safe) + "\n", "utf8")
  }

  /** Hold a prediction until the user replies. Nothing is written yet. */
  predicted(entry: Pending): void {
    this.pending.set(entry.id, entry)
  }

  /**
   * Record a request the model never judged. Deliberately carries the reason and the kind of
   * action only - not the compacted state. Storing content that was intentionally never
   * evaluated would widen the privacy surface for no analytical gain.
   */
  async refused(input: { id: string; action: string; reason: RefusalReason }): Promise<void> {
    await this.append({ kind: "refusal", ts: new Date().toISOString(), ...input })
  }

  /**
   * Audit record for a permission the model granted in live mode: what was allowed, at what
   * probability, above which certified threshold, by which checkpoint.
   */
  async autoApproved(input: { id: string; action: string; probability: number; threshold: number; checkpoint: string }): Promise<void> {
    await this.append({ kind: "auto-approved", ts: new Date().toISOString(), ...input })
  }

  /**
   * A tool output scored for injection. Unlabelled by construction: it becomes training or
   * evaluation data only after a person labels it (train/label.py).
   */
  async observedOutput(input: { id: string; tool: string; probability: number; checkpoint: string; state: unknown }): Promise<void> {
    await this.append({ kind: "inject-observation", ts: new Date().toISOString(), ...input, state: JSON.stringify(input.state) })
  }

  /** Join the user's reply to the prediction, producing a labelled training row. */
  async replied(id: string, reply: Reply, replier: "user" | "auto" = "user"): Promise<boolean> {
    const entry = this.pending.get(id)
    if (!entry) return false
    this.pending.delete(id)
    const questionId = Object.keys(entry.questions)[0]!
    await this.append({
      kind: "decision",
      id: entry.id,
      ts: entry.ts,
      // Laya's training schema: three JSON strings.
      state: JSON.stringify(entry.state),
      questions: JSON.stringify(entry.questions),
      gold: JSON.stringify(gold(reply, questionId)),
      // Kept distinct from `once`: "always" is a stronger approval and may be weighted
      // differently, which collapsing to a boolean would make impossible to recover.
      reply,
      // "auto" when no human answered (e.g. `run --auto`, `--dangerously-skip-permissions`): the
      // reply then says nothing about what a user would decide and must never become a label.
      replier,
      probability: entry.probability,
      checkpoint: entry.checkpoint,
      latency_ms: entry.latencyMs,
      estimated_tokens: entry.estimatedTokens,
      input_tokens: entry.inputTokens,
    })
    return true
  }

  /**
   * Write everything still waiting as unlabelled. `gold` is null, which is the marker the
   * training split filters on: a prediction with no human decision is not a training example.
   */
  /** A retention plan at a compaction point: counts and token sizes only, no text. */
  async retention(record: Record<string, unknown>): Promise<void> {
    await this.append({ kind: "retention", ts: new Date().toISOString(), ...record })
  }

  async flushUnlabelled(): Promise<number> {
    const entries = [...this.pending.values()]
    this.pending.clear()
    for (const entry of entries) {
      await this.append({
        kind: "decision",
        id: entry.id,
        ts: entry.ts,
        state: JSON.stringify(entry.state),
        questions: JSON.stringify(entry.questions),
        gold: null,
        reply: null,
        probability: entry.probability,
        checkpoint: entry.checkpoint,
        latency_ms: entry.latencyMs,
      })
    }
    return entries.length
  }

  /** Delete every record. Returns the files removed. */
  async purge(): Promise<string[]> {
    this.pending.clear()
    let names: string[]
    try {
      names = await fs.readdir(this.dir)
    } catch {
      return []
    }
    const removed: string[] = []
    for (const name of names) {
      if (!name.startsWith("shadow-") || !name.endsWith(".jsonl")) continue
      await fs.rm(path.join(this.dir, name), { force: true })
      removed.push(name)
    }
    return removed
  }

  get location(): string {
    return this.dir
  }
}
