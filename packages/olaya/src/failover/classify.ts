/**
 * What to do about a failed model call (olaya-provider-failover, gate F2).
 *
 * Providers disagree about what a status code means: Anthropic's monthly spend cap is a 429 with
 * no `retry-after`, a user-set Anthropic limit is a 400, Qwen's `insufficient_quota` is a
 * one-minute throttle while OpenAI's means the credit is gone, and Kimi answers 429 for
 * throttling, overload and quota alike. So rules key on provider family, status, error code, body
 * and headers together, first match wins, and anything unrecognised takes the conservative
 * `switch` (docs/research/11c).
 */

export type Action =
  /** Try the same model again after `wait` ms. */
  | "retry"
  /** Try the same model up to `retries` more times, then move on; cool it with backoff. */
  | "switch"
  /** Move on now; the model is unavailable until `until`. */
  | "switch-until"
  /** Move on; the model stays unavailable until a person acts (credit, auth, missing model). */
  | "disable"
  /** The context is too long for this model. */
  | "shrink"
  /** The history is in a shape this provider rejects; fix it and try once more. */
  | "repair"
  /** No other model would do better (content policy, a malformed request). */
  | "stop"

export type Failure = {
  providerID: string
  status?: number
  message: string
  /** The raw response body or stream error chunk, when there is one. */
  body?: string
  headers?: Record<string, string>
  /** What the harness itself already knows about the failure. */
  kind?: "context" | "auth" | "timeout" | "stall" | "network" | "aborted" | "refusal"
}

export type Verdict = { action: Action; reason: string; wait?: number; until?: number; retries?: number }

/** The reason given to a failure no rule recognises. */
export const UNRECOGNISED = "unrecognised failure"

/** Whether waiting and asking the same model again can succeed: not with a bad key, a rejected history or a failure nobody recognised. */
export const clearsWithTime = (verdict: Verdict) =>
  verdict.action !== "disable" && verdict.action !== "repair" && verdict.reason !== UNRECOGNISED

/** A throttle longer than this is not waited out: the model is skipped until it ends. */
export const MAX_WAIT_MS = 60_000

const CONTEXT =
  /context[_ ]length|context window|prompt is too long|input token length too long|exceed(s|ed)? .*(token|context)|maximum (prompt|context) length|range of input length/i

export function classify(failure: Failure, now = Date.now()): Verdict {
  const family = familyOf(failure.providerID)
  const text = `${failure.message}\n${failure.body ?? ""}`
  const lower = text.toLowerCase()
  const status = failure.status
  const has = (...needles: string[]) => needles.some((n) => lower.includes(n.toLowerCase()))
  const resetAt = parseReset(failure.headers ?? {}, text, now)

  if (failure.kind === "aborted") return { action: "stop", reason: "aborted by the user" }

  // content policy: another model would refuse too, or should not be asked
  if (
    failure.kind === "refusal" ||
    has("cyber_policy", "bio_policy", "misalignment_policy_violation", "data_inspection_failed", "content_filter") ||
    has("prohibited_content", "content_policy_violation") ||
    (family === "openai" && has('"invalid_prompt"')) ||
    (family === "google" && status === 400 && has("safety"))
  )
    return { action: "stop", reason: "content policy" }

  // usage windows that reset at a known time
  if (has("usage_limit_reached"))
    return resetAt
      ? { action: "switch-until", until: resetAt, reason: "usage limit reached" }
      : { action: "switch", retries: 0, reason: "usage limit reached" }
  if (has("enforced_spend_limit_reached"))
    return resetAt
      ? { action: "switch-until", until: resetAt, reason: "spend limit reached" }
      : { action: "disable", reason: "spend limit reached" }
  if (family === "google" && status === 429 && (has("quota_exceeded") || /PerDay/.test(text)))
    return { action: "switch-until", until: nextMidnight(now, "America/Los_Angeles"), reason: "daily quota reached" }

  // credit or billing gone: nothing changes until a person acts
  if (
    has(
      "usage_not_included",
      "credit_balance_exhausted",
      "spend_limit_exceeded",
      "organization_usage_limit_exceeded",
    ) ||
    has(
      "exceeded_current_quota_error",
      "arrearage",
      "allocationquota.freetieronly",
      "billing_error",
      "insufficient balance",
    ) ||
    has("you have reached your specified", "used all available credits", "monthly spending limit") ||
    (family !== "alibaba" && has("insufficient_quota")) ||
    (status === 402 && !has("openrouter_in_flight_budget"))
  )
    return { action: "disable", reason: "credit or billing exhausted" }

  // provider-side windows with no exposed reset: skip the model, cool with backoff
  if (has("allocated quota exceeded", "usage limit") && (status === 403 || status === 429))
    return { action: "switch", retries: 0, reason: "usage window reached" }

  // the context is too long: only on a rejected request or a stream error, never on a throttle
  // ("exceeded … tokens per minute" reads like a context error)
  const throttled = status === 429 || has("rate_limit", "rate limit", "per minute", "per min", "too_many_requests")
  if (
    failure.kind === "context" ||
    status === 413 ||
    ((status === 400 || status === undefined) && !throttled && CONTEXT.test(text))
  )
    return { action: "shrink", reason: "context too long" }

  if (failure.kind === "auth" || status === 401) return { action: "disable", reason: "authentication failed" }
  if (status === 403) return { action: "disable", reason: "permission denied" }
  if (status === 404) return { action: "disable", reason: "model not found" }

  // a history shape this provider rejects: tool-call IDs, reasoning fields, signatures, images
  if (
    status === 400 &&
    (has("tool_call_id", "tool_use_id", "tool_use.id", "reasoning_content is missing", "thought_signature") ||
      has("signature", "image_url", "with a length of 9", "range of max_tokens"))
  )
    return { action: "repair", reason: "history rejected by this provider" }

  // overload and outage: two quick tries on the same model, then move on
  if (
    status === 529 ||
    status === 503 ||
    status === 502 ||
    has("overloaded_error", "server_is_overloaded", "engine_overloaded_error", "provider_overloaded", '"unavailable"')
  )
    return { action: "switch", retries: 2, reason: "provider overloaded or unavailable" }

  // throttles: wait them out when short, skip the model when long
  if (throttled || has("resource_exhausted", "slow_down", "openrouter_in_flight_budget")) {
    const wait = resetAt === undefined ? undefined : Math.max(0, resetAt - now)
    if (wait !== undefined && wait > MAX_WAIT_MS)
      return { action: "switch-until", until: resetAt, reason: "rate limited" }
    return { action: "retry", ...(wait !== undefined && { wait }), reason: "rate limited" }
  }

  if (status !== undefined && status >= 500) return { action: "switch", retries: 2, reason: `server error ${status}` }
  if (failure.kind === "stall") return { action: "switch", retries: 1, reason: "stream stalled" }
  if (failure.kind === "timeout" || failure.kind === "network" || status === 408)
    return { action: "switch", retries: 2, reason: "network or timeout" }
  if (status === 400 || status === 422) return { action: "stop", reason: `request rejected (${status})` }
  return { action: "switch", retries: 2, reason: UNRECOGNISED }
}

/** Providers grouped by who defines their error shapes. */
export function familyOf(providerID: string) {
  const id = providerID.toLowerCase()
  if (id.includes("anthropic") || id === "bedrock") return "anthropic"
  if (id === "google" || id.startsWith("google-") || id.includes("vertex") || id.includes("gemini")) return "google"
  if (id.includes("moonshot") || id.includes("kimi")) return "kimi"
  if (id.includes("alibaba") || id.includes("dashscope") || id.includes("qwen")) return "alibaba"
  if (id.includes("deepseek")) return "deepseek"
  if (id.includes("openrouter")) return "openrouter"
  if (id === "openai" || id.includes("azure")) return "openai"
  return id
}

/**
 * When the provider says to come back, from headers and body, in every format providers use:
 * `retry-after-ms`; `retry-after` in seconds or as a date; Anthropic's RFC 3339 reset headers;
 * OpenAI's durations (`6m0s`); epoch seconds (`resets_at`, `x-codex-*-reset-at`); Gemini's
 * `retryDelay`; and "try again in 11.05s" or "regain access on 2026-10-01" in the message.
 */
export function parseReset(headers: Record<string, string>, text: string, now = Date.now()): number | undefined {
  const h = Object.fromEntries(Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]))
  const ms = Number(h["retry-after-ms"])
  if (h["retry-after-ms"] !== undefined && Number.isFinite(ms)) return now + ms
  if (h["retry-after"] !== undefined) {
    const seconds = Number(h["retry-after"])
    if (Number.isFinite(seconds)) return now + seconds * 1000
    const date = Date.parse(h["retry-after"])
    if (!Number.isNaN(date)) return date
  }
  const epoch =
    /"resets?_at"\s*:\s*(\d{9,})/.exec(text)?.[1] ?? h["x-codex-primary-reset-at"] ?? h["x-codex-secondary-reset-at"]
  if (epoch) return Number(epoch) * 1000
  const anthropic = Object.entries(h)
    .filter(([k]) => k.startsWith("anthropic-ratelimit-") && k.endsWith("-reset"))
    .map(([, v]) => Date.parse(v))
    .filter((t) => !Number.isNaN(t))
  if (anthropic.length) return Math.max(...anthropic)
  const openai = [h["x-ratelimit-reset-requests"], h["x-ratelimit-reset-tokens"]].flatMap((v) =>
    v ? [duration(v)] : [],
  )
  if (openai.some((d) => d !== undefined)) return now + Math.max(...openai.filter((d): d is number => d !== undefined))
  const delay =
    /"retryDelay"\s*:\s*"([\d.]+)s"/.exec(text)?.[1] ?? /try again in ([\d.]+)\s*s(?:econds)?\b/i.exec(text)?.[1]
  if (delay) return now + Number(delay) * 1000
  const later = /try again in ((?:\d+h)?(?:\d+m)?(?:[\d.]+s)?)/i.exec(text)?.[1]
  if (later && duration(later) !== undefined) return now + duration(later)!
  const on = /(?:regain access|resets?|available again)\s+(?:on|at)\s+(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}))?/i.exec(
    text,
  )
  if (on) return Date.parse(`${on[1]}T${on[2] ?? "00:00"}:00Z`)
  return undefined
}

/** "1s", "6m0s", "1h2m3.5s", "250ms" → milliseconds. */
export function duration(value: string) {
  const m = /^(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?(?:(\d+)ms)?$/.exec(value.trim())
  if (!m || !value.trim()) return undefined
  const [, hours, minutes, seconds, millis] = m
  return ((Number(hours ?? 0) * 60 + Number(minutes ?? 0)) * 60 + Number(seconds ?? 0)) * 1000 + Number(millis ?? 0)
}

/** The next midnight in a time zone (Gemini's daily quotas reset at midnight Pacific). */
export function nextMidnight(now: number, timeZone: string) {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone,
      hourCycle: "h23",
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
    })
      .formatToParts(new Date(now))
      .map((p) => [p.type, Number(p.value)]),
  )
  const elapsed = ((parts.hour ?? 0) * 3600 + (parts.minute ?? 0) * 60 + (parts.second ?? 0)) * 1000
  return now - (now % 1000) - elapsed + 24 * 3600 * 1000
}

export * as FailoverClassify from "./classify"
