import fs from "fs"
import path from "path"
import { Global } from "@olaya/core/global"
import type { Verdict } from "./classify"

/**
 * Which models are unavailable, until when and why (olaya-provider-failover, design D4). A model
 * that failed with `switch` cools with backoff (1, 5, 15, then 60 minutes); `switch-until` cools
 * to the provider's reset time. `disable` (credit, auth, a missing model) cools for 6 hours, so a
 * fixed account is tried again without anyone having to clear it, but a chain where every model
 * is disabled ends the run rather than waiting. A success clears the model's entry.
 *
 * Kept on disk and shared by every session: with `olaya run` each task is a new process, and
 * without this every task would try an exhausted provider first.
 */
export type Entry = { state: "cooling" | "disabled"; until: number; reason: string; failures: number }

const BACKOFF_MINUTES = [1, 5, 15, 60]
const DISABLED_MS = 6 * 3_600_000

// Shared by every Olaya process on the machine: re-read whenever another process has written it,
// and replaced atomically, so parallel runs and a long-lived server see each other's cooldowns.
// ponytail: two processes marking in the same instant can still drop one entry; per-key files or
// a lock if that ever matters.
let file: string | undefined
const where = () => (file ??= path.join(Global.Path.data, "failover", "availability.json"))
/** The file's identity when last read or written by this process; undefined before the first read. */
let seen: string | undefined
const entries = new Map<string, Entry>()

// Every write is a rename, so a new inode: with size and mtime, a change shows even within one
// tick of a coarse filesystem clock.
const identity = (file: string) => {
  try {
    const stat = fs.statSync(file)
    return `${stat.ino}:${stat.size}:${stat.mtimeMs}`
  } catch {
    return "absent"
  }
}

function load() {
  const now = identity(where())
  if (now === seen) return
  seen = now
  const raw = (() => {
    try {
      return JSON.parse(fs.readFileSync(where(), "utf8")) as Record<string, Entry>
    } catch {
      return {}
    }
  })()
  entries.clear()
  for (const [model, entry] of Object.entries(raw)) entries.set(model, entry)
}

function save() {
  try {
    fs.mkdirSync(path.dirname(where()), { recursive: true })
    const tmp = `${where()}.${process.pid}.tmp`
    fs.writeFileSync(tmp, JSON.stringify(Object.fromEntries(entries), null, 1))
    // taken before the rename, so a write another process makes right after is still seen as new
    const written = identity(tmp)
    fs.renameSync(tmp, where())
    seen = written
  } catch {
    // a read-only data directory leaves availability per process, as before it was saved
  }
}

export function mark(model: string, verdict: Verdict, now = Date.now()) {
  load()
  const failures = (entries.get(model)?.failures ?? 0) + 1
  const backoff = BACKOFF_MINUTES[Math.min(failures, BACKOFF_MINUTES.length) - 1]! * 60_000
  const until =
    verdict.action === "disable"
      ? now + DISABLED_MS
      : verdict.action === "switch-until" && verdict.until
        ? verdict.until
        : now + backoff
  const entry: Entry = {
    state: verdict.action === "disable" ? "disabled" : "cooling",
    until,
    reason: verdict.reason,
    failures,
  }
  entries.set(model, entry)
  save()
  return entry
}

export function available(model: string, now = Date.now()) {
  load()
  return (entries.get(model)?.until ?? 0) <= now
}

export function get(model: string) {
  load()
  return entries.get(model)
}

/**
 * A success: the model's backoff starts over. A cooldown still in the future was written by
 * another process while this request was in flight (this one never picks a cooling model), so it
 * stays; a disabled model that answered has had its key fixed.
 */
export function recovered(model: string, now = Date.now()) {
  load()
  const entry = entries.get(model)
  if (!entry || (entry.state === "cooling" && entry.until > now)) return
  entries.delete(model)
  save()
}

export function clear() {
  entries.clear()
  save()
}

/** Point the store at another file (tests), dropping what is in memory. */
export function useFile(next: string) {
  file = next
  seen = undefined
  entries.clear()
}

/** The chain for a session: its own model first, then the configured fallbacks, without repeats. */
export const chain = (preferred: string, fallbacks: readonly string[] = []) => [...new Set([preferred, ...fallbacks])]

/** The first model in the chain that is available now. */
export const pick = (models: readonly string[], now = Date.now()) => models.find((m) => available(m, now))

/** When the first cooling model in the chain comes back; undefined when every one is disabled. */
export function earliest(models: readonly string[]) {
  load()
  const times = models.flatMap((m) => {
    const entry = entries.get(m)
    return !entry ? [0] : entry.state === "disabled" ? [] : [entry.until]
  })
  return times.length ? Math.min(...times) : undefined
}

export * as FailoverAvailability from "./availability"
