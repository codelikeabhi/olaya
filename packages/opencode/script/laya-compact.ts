#!/usr/bin/env bun
/**
 * Compaction bridge for offline tooling (benchmark, dataset builder).
 *
 * Runs the production `compact()` and denylist over raw permission requests, so nothing
 * outside this package ever reimplements them: a state built offline is byte-identical to
 * the state the plugin would send for the same request.
 *
 *   stdin : JSONL  {"id", "request": {permission, patterns, metadata}, "context": {task, recent, cwd}, "budget"}
 *   stdout: JSONL  {"id", "denylisted"?: rule, "state"?, "estimatedTokens"?, "dropped"?, "refused"?: reason}
 */

import { compact, QUESTIONS } from "../src/laya/state"
import { denylisted } from "../src/laya/judgment"

const FALLBACK_BUDGET = 300

// `--questions` prints the production question set, so offline callers ask exactly what the
// plugin asks.
if (process.argv.includes("--questions")) {
  process.stdout.write(JSON.stringify(QUESTIONS) + "\n")
  process.exit(0)
}

let buffer = ""
for await (const chunk of Bun.stdin.stream()) {
  buffer += new TextDecoder().decode(chunk)
  let nl: number
  while ((nl = buffer.indexOf("\n")) >= 0) {
    const line = buffer.slice(0, nl).trim()
    buffer = buffer.slice(nl + 1)
    if (line) process.stdout.write(JSON.stringify(handle(JSON.parse(line))) + "\n")
  }
}
if (buffer.trim()) process.stdout.write(JSON.stringify(handle(JSON.parse(buffer))) + "\n")

function handle(row: { id: string; request: any; context?: any; budget?: number }) {
  const req = {
    permission: row.request?.permission ?? "unknown",
    patterns: Array.isArray(row.request?.patterns) ? row.request.patterns : [],
    metadata: row.request?.metadata ?? {},
  }
  // Same order as judgment.evaluate: the denylist is checked before compaction.
  const rule = denylisted(req)
  if (rule) return { id: row.id, denylisted: rule }
  const out = compact(req, row.context ?? {}, row.budget ?? FALLBACK_BUDGET)
  if (!out.ok) return { id: row.id, refused: out.reason }
  return { id: row.id, state: out.state, estimatedTokens: out.estimatedTokens, dropped: out.dropped }
}
