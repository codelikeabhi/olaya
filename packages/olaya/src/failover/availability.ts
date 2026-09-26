import type { Verdict } from "./classify"

/**
 * Which models are unavailable, until when and why (olaya-provider-failover, design D4). A model
 * that failed with `switch` cools with backoff (1, 5, 15, then 60 minutes); `switch-until` cools
 * to the provider's reset time; `disable` lasts until a person acts. A success clears it.
 */
export type Entry = { state: "cooling" | "disabled"; until: number; reason: string; failures: number }

const BACKOFF_MINUTES = [1, 5, 15, 60]

// ponytail: one in-memory store per process, shared by every session; written to disk in F5 so
// cooldowns survive a restart.
const entries = new Map<string, Entry>()

export function mark(model: string, verdict: Verdict, now = Date.now()) {
  const failures = (entries.get(model)?.failures ?? 0) + 1
  const backoff = BACKOFF_MINUTES[Math.min(failures, BACKOFF_MINUTES.length) - 1]! * 60_000
  const until =
    verdict.action === "disable"
      ? Infinity
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
  return entry
}

export const available = (model: string, now = Date.now()) => (entries.get(model)?.until ?? 0) <= now
export const get = (model: string) => entries.get(model)
export const recovered = (model: string) => void entries.delete(model)
export const clear = () => entries.clear()

/** The chain for a session: its own model first, then the configured fallbacks, without repeats. */
export const chain = (preferred: string, fallbacks: readonly string[] = []) => [...new Set([preferred, ...fallbacks])]

/** The first model in the chain that is available now. */
export const pick = (models: readonly string[], now = Date.now()) => models.find((m) => available(m, now))

/** When the first cooling model in the chain comes back; undefined if all are disabled. */
export function earliest(models: readonly string[]) {
  const times = models.map((m) => entries.get(m)?.until ?? 0).filter((t) => t !== Infinity)
  return times.length ? Math.min(...times) : undefined
}

export * as FailoverAvailability from "./availability"
