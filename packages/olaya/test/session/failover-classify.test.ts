/**
 * The failure classifier against the provider error shapes in docs/research/11c (gate F2): one
 * fixture per documented shape, reset parsing in every format providers use, and the conservative
 * default for shapes it does not know.
 */
import { FailoverFailure } from "../../src/failover/failure"
import { FailoverClassify } from "../../src/failover/classify"
import { describe, expect, test } from "bun:test"
import { classify, duration, familyOf, parseReset, type Action, type Failure } from "../../src/failover/classify"

const NOW = Date.parse("2026-09-27T10:00:00Z")
const json = (value: unknown) => JSON.stringify(value)

type Case = [
  name: string,
  failure: Failure,
  action: Action,
  extra?: { wait?: number; until?: number; retries?: number },
]

const cases: Case[] = [
  // Anthropic
  [
    "anthropic throttle",
    {
      providerID: "anthropic",
      status: 429,
      message: "rate limited",
      body: json({
        type: "error",
        error: {
          type: "rate_limit_error",
          message: "Number of request tokens has exceeded your per-minute rate limit",
        },
      }),
      headers: { "retry-after": "3" },
    },
    "retry",
    { wait: 3000 },
  ],
  [
    "anthropic throttle, long wait",
    {
      providerID: "anthropic",
      status: 429,
      message: "rate limited",
      body: json({ type: "error", error: { type: "rate_limit_error" } }),
      headers: { "retry-after": "3600" },
    },
    "switch-until",
    { until: NOW + 3_600_000 },
  ],
  [
    "anthropic monthly spend cap",
    {
      providerID: "anthropic",
      status: 429,
      message: "spend limit",
      body: json({
        type: "error",
        error: {
          type: "rate_limit_error",
          message: "You will regain access on 2026-10-01 at 00:00 UTC.",
          details: { error_code: "enforced_spend_limit_reached" },
        },
      }),
    },
    "switch-until",
    { until: Date.parse("2026-10-01T00:00:00Z") },
  ],
  [
    "anthropic user-set limit (a 400)",
    {
      providerID: "anthropic",
      status: 400,
      message: "You have reached your specified workspace API usage limits.",
      body: json({ type: "error", error: { type: "invalid_request_error" } }),
    },
    "disable",
  ],
  [
    "anthropic billing",
    {
      providerID: "anthropic",
      status: 402,
      message: "billing",
      body: json({ type: "error", error: { type: "billing_error" } }),
    },
    "disable",
  ],
  [
    "anthropic overloaded",
    {
      providerID: "anthropic",
      status: 529,
      message: "Overloaded",
      body: json({ type: "error", error: { type: "overloaded_error" } }),
    },
    "switch",
    { retries: 2 },
  ],
  [
    "anthropic overloaded mid-stream",
    {
      providerID: "anthropic",
      message: "stream error",
      body: json({ type: "error", error: { type: "overloaded_error", message: "Overloaded" } }),
    },
    "switch",
    { retries: 2 },
  ],
  [
    "anthropic prompt too long",
    {
      providerID: "anthropic",
      status: 400,
      message: "prompt is too long: 215000 tokens > 200000 maximum",
      body: json({ type: "error", error: { type: "invalid_request_error" } }),
    },
    "shrink",
  ],
  ["anthropic request too large", { providerID: "anthropic", status: 413, message: "request_too_large" }, "shrink"],
  [
    "anthropic signature mismatch",
    {
      providerID: "anthropic",
      status: 400,
      message: "Invalid `signature` in `thinking` block: bound to a different conversation",
    },
    "repair",
  ],
  [
    "anthropic bad key",
    {
      providerID: "anthropic",
      status: 401,
      message: "invalid x-api-key",
      body: json({ type: "error", error: { type: "authentication_error" } }),
    },
    "disable",
  ],
  ["anthropic refusal", { providerID: "anthropic", message: "refused", kind: "refusal" }, "stop"],
  // OpenAI and Codex
  [
    "openai throttle",
    {
      providerID: "openai",
      status: 429,
      message: "Rate limit reached for gpt. Please try again in 11.054s.",
      body: json({ error: { code: "rate_limit_exceeded" } }),
    },
    "retry",
    { wait: 11054 },
  ],
  [
    "openai slow down",
    {
      providerID: "openai",
      status: 429,
      message: "slow down",
      body: json({ error: { type: "rate_limit_error", code: "slow_down" } }),
      headers: { "retry-after": "2" },
    },
    "retry",
    { wait: 2000 },
  ],
  [
    "openai quota gone",
    {
      providerID: "openai",
      status: 429,
      message: "You exceeded your current quota",
      body: json({ error: { code: "insufficient_quota" } }),
    },
    "disable",
  ],
  [
    "openai credit exhausted",
    {
      providerID: "openai",
      status: 429,
      message: "Credit balance exhausted",
      body: json({ error: { code: "credit_balance_exhausted" } }),
    },
    "disable",
  ],
  [
    "openai project spend limit",
    {
      providerID: "openai",
      status: 429,
      message: "limit",
      body: json({ error: { code: "project_spend_limit_exceeded" } }),
    },
    "disable",
  ],
  [
    "openai overloaded",
    {
      providerID: "openai",
      status: 503,
      message: "server overloaded",
      body: json({ error: { code: "server_is_overloaded" } }),
    },
    "switch",
    { retries: 2 },
  ],
  [
    "openai context",
    {
      providerID: "openai",
      status: 400,
      message: "Your input exceeds the context window of this model",
      body: json({ error: { code: "context_length_exceeded" } }),
    },
    "shrink",
  ],
  [
    "openai cyber policy",
    { providerID: "openai", status: 400, message: "blocked", body: json({ error: { code: "cyber_policy" } }) },
    "stop",
  ],
  [
    "codex usage window",
    {
      providerID: "openai",
      status: 429,
      message: "You've hit your usage limit",
      body: json({
        error: { type: "usage_limit_reached", plan_type: "plus", resets_at: 1790500000, limit_window_minutes: 300 },
      }),
    },
    "switch-until",
    { until: 1790500000 * 1000 },
  ],
  [
    "codex plan without codex",
    {
      providerID: "openai",
      status: 403,
      message: "usage not included",
      body: json({ error: { type: "usage_not_included" } }),
    },
    "disable",
  ],
  // Gemini
  [
    "gemini per-minute throttle",
    {
      providerID: "google",
      status: 429,
      message: "RESOURCE_EXHAUSTED",
      body: json({
        error: {
          status: "RESOURCE_EXHAUSTED",
          details: [
            { "@type": "type.googleapis.com/google.rpc.RetryInfo", retryDelay: "34s" },
            { quotaId: "GenerateRequestsPerMinutePerProjectPerModel" },
          ],
        },
      }),
    },
    "retry",
    { wait: 34000 },
  ],
  [
    "gemini daily quota (a short retryDelay does not mean soon)",
    {
      providerID: "google",
      status: 429,
      message: "quota",
      body: json({
        error: {
          status: "RESOURCE_EXHAUSTED",
          details: [{ retryDelay: "20s" }, { quotaId: "GenerateRequestsPerDayPerProjectPerModel" }],
        },
      }),
    },
    "switch-until",
    { until: Date.parse("2026-09-28T07:00:00Z") },
  ],
  ["gemini prepay credit gone", { providerID: "google", status: 402, message: "Prepay credits exhausted" }, "disable"],
  [
    "gemini unavailable",
    { providerID: "google", status: 503, message: "UNAVAILABLE", body: json({ error: { status: "UNAVAILABLE" } }) },
    "switch",
    { retries: 2 },
  ],
  [
    "gemini context",
    {
      providerID: "google",
      status: 400,
      message: "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).",
    },
    "shrink",
  ],
  [
    "gemini thought signature",
    {
      providerID: "google",
      status: 400,
      message: "Function call is missing a thought_signature in functionCall parts.",
    },
    "repair",
  ],
  // Kimi: 429 means three different things
  [
    "kimi throttle",
    {
      providerID: "moonshotai",
      status: 429,
      message: "rate limit",
      body: json({
        error: { type: "rate_limit_reached_error", message: "Your account reached max request per minute" },
      }),
    },
    "retry",
  ],
  [
    "kimi quota",
    {
      providerID: "moonshotai",
      status: 429,
      message: "quota",
      body: json({ error: { type: "exceeded_current_quota_error", message: "Your account balance is insufficient" } }),
    },
    "disable",
  ],
  [
    "kimi overloaded",
    {
      providerID: "moonshotai",
      status: 429,
      message: "overloaded",
      body: json({ error: { type: "engine_overloaded_error" } }),
      headers: { "retry-after": "5" },
    },
    "switch",
    { retries: 2 },
  ],
  [
    "kimi code 5-hour window",
    { providerID: "kimi-for-coding", status: 403, message: "You've reached your 5-hour usage limit." },
    "switch",
    { retries: 0 },
  ],
  [
    "kimi context",
    {
      providerID: "moonshotai",
      status: 400,
      message: "Invalid request: Input token length too long",
      body: json({ error: { type: "invalid_request_error" } }),
    },
    "shrink",
  ],
  [
    "kimi missing reasoning_content",
    { providerID: "moonshotai", status: 400, message: "reasoning_content is missing in assistant tool call message" },
    "repair",
  ],
  [
    "kimi content filter",
    {
      providerID: "moonshotai",
      status: 400,
      message: "The request was rejected because it was considered high risk",
      body: json({ error: { type: "content_filter" } }),
    },
    "stop",
  ],
  // Qwen: insufficient_quota is a throttle here, billing is a 400
  [
    "qwen throttle named insufficient_quota",
    {
      providerID: "alibaba",
      status: 429,
      message: "insufficient_quota",
      body: json({ error: { code: "insufficient_quota" } }),
    },
    "retry",
  ],
  [
    "qwen overdue account (a 400)",
    {
      providerID: "alibaba",
      status: 400,
      message: "Access denied, please make sure your account is in good standing.",
      body: json({ code: "Arrearage" }),
    },
    "disable",
  ],
  [
    "qwen free tier only",
    {
      providerID: "alibaba",
      status: 403,
      message: "free tier only",
      body: json({ code: "AllocationQuota.FreeTierOnly" }),
    },
    "disable",
  ],
  [
    "qwen context",
    {
      providerID: "alibaba",
      status: 400,
      message: "Range of input length should be [1, 129024]",
      body: json({ code: "InvalidParameter" }),
    },
    "shrink",
  ],
  [
    "qwen max_tokens too large",
    { providerID: "alibaba", status: 400, message: "Range of max_tokens should be [1, 8192]" },
    "repair",
  ],
  [
    "alibaba coding plan window",
    { providerID: "alibaba", status: 429, message: "hour allocated quota exceeded" },
    "switch",
    { retries: 0 },
  ],
  // DeepSeek, xAI, Mistral
  ["deepseek balance", { providerID: "deepseek", status: 402, message: "Insufficient Balance" }, "disable"],
  [
    "deepseek overloaded",
    { providerID: "deepseek", status: 503, message: "Server Overloaded" },
    "switch",
    { retries: 2 },
  ],
  [
    "deepseek context",
    {
      providerID: "deepseek",
      status: 400,
      message: "This model's maximum context length is 131072 tokens. However, you requested 140000 tokens",
    },
    "shrink",
  ],
  [
    "deepseek rejects images",
    {
      providerID: "deepseek",
      status: 400,
      message: "Failed to deserialize: unknown variant `image_url`, expected `text`",
    },
    "repair",
  ],
  [
    "xai credits",
    {
      providerID: "xai",
      status: 403,
      message: "Your team has either used all available credits or reached its monthly spending limit.",
    },
    "disable",
  ],
  [
    "xai context",
    {
      providerID: "xai",
      status: 400,
      message: "This model's maximum prompt length is 131072 but the request contains 150000 tokens.",
    },
    "shrink",
  ],
  [
    "mistral tool-call id",
    {
      providerID: "mistral",
      status: 400,
      message: "Tool call id was toolu01XYz but must be a-z, A-Z, 0-9, with a length of 9.",
    },
    "repair",
  ],
  [
    "mistral throttle",
    { providerID: "mistral", status: 429, message: "Requests rate limit exceeded", headers: { "Retry-After": "1" } },
    "retry",
    { wait: 1000 },
  ],
  // OpenRouter: 402 can be transient
  [
    "openrouter credits",
    {
      providerID: "openrouter",
      status: 402,
      message: "Insufficient credits",
      body: json({ error: { code: 402, metadata: { limit_source: "openrouter_credits" } } }),
    },
    "disable",
  ],
  [
    "openrouter in-flight budget",
    {
      providerID: "openrouter",
      status: 402,
      message: "in flight",
      body: json({ error: { code: 402, metadata: { limit_source: "openrouter_in_flight_budget" } } }),
      headers: { "retry-after": "4" },
    },
    "retry",
    { wait: 4000 },
  ],
  [
    "openrouter provider overloaded",
    { providerID: "openrouter", status: 503, message: "provider_overloaded" },
    "switch",
    { retries: 2 },
  ],
  [
    "openrouter context",
    {
      providerID: "openrouter",
      status: 400,
      message: "context",
      body: json({ error: { metadata: { error_type: "context_length_exceeded" } } }),
    },
    "shrink",
  ],
  [
    "openrouter policy",
    {
      providerID: "openrouter",
      status: 403,
      message: "flagged",
      body: json({ error: { metadata: { error_type: "content_policy_violation" } } }),
    },
    "stop",
  ],
  // local servers and the harness's own signals
  [
    "ollama server error",
    { providerID: "ollama", status: 500, message: "llama runner process has terminated" },
    "switch",
    { retries: 2 },
  ],
  [
    "vllm context",
    {
      providerID: "vllm",
      status: 400,
      message:
        "This model's maximum context length is 32768 tokens. However, you requested 1000 output tokens and your prompt contains 40000 input tokens",
    },
    "shrink",
  ],
  [
    "stalled stream",
    { providerID: "anthropic", message: "SSE read timed out", kind: "stall" },
    "switch",
    { retries: 0 },
  ],
  ["connection reset", { providerID: "openai", message: "ECONNRESET", kind: "network" }, "switch", { retries: 2 }],
  [
    "request timeout",
    { providerID: "openai", message: "The operation timed out.", kind: "timeout" },
    "switch",
    { retries: 2 },
  ],
  ["user abort", { providerID: "anthropic", message: "aborted", kind: "aborted" }, "stop"],
  ["model not found", { providerID: "openai", status: 404, message: "The model does not exist" }, "disable"],
  [
    "a throttle message mid-stream is not a context error",
    { providerID: "moonshotai", message: "exceeded the tokens per minute rate limit" },
    "retry",
  ],
  [
    "another 400 is a harness bug",
    { providerID: "openai", status: 400, message: "Invalid value for 'service_tier'" },
    "stop",
  ],
  [
    "an unrecognised failure moves on",
    { providerID: "somewhere", message: "the vibes are off" },
    "switch",
    { retries: 2 },
  ],
]

describe("failover classifier", () => {
  for (const [name, failure, action, extra] of cases) {
    test(name, () => {
      const verdict = classify(failure, NOW)
      expect(verdict.action).toBe(action)
      if (extra?.wait !== undefined) expect(verdict.wait).toBe(extra.wait)
      if (extra?.until !== undefined) expect(verdict.until).toBe(extra.until)
      if (extra?.retries !== undefined) expect(verdict.retries).toBe(extra.retries)
    })
  }
})

describe("reset times", () => {
  test("every format providers use", () => {
    expect(parseReset({ "retry-after-ms": "1500" }, "", NOW)).toBe(NOW + 1500)
    expect(parseReset({ "Retry-After": "30" }, "", NOW)).toBe(NOW + 30_000)
    expect(parseReset({ "retry-after": "Sun, 27 Sep 2026 10:05:00 GMT" }, "", NOW)).toBe(
      Date.parse("2026-09-27T10:05:00Z"),
    )
    expect(
      parseReset(
        {
          "anthropic-ratelimit-tokens-reset": "2026-09-27T10:01:00Z",
          "anthropic-ratelimit-requests-reset": "2026-09-27T10:00:05Z",
        },
        "",
        NOW,
      ),
    ).toBe(Date.parse("2026-09-27T10:01:00Z"))
    expect(parseReset({ "x-ratelimit-reset-requests": "1s", "x-ratelimit-reset-tokens": "6m0s" }, "", NOW)).toBe(
      NOW + 360_000,
    )
    expect(parseReset({ "x-codex-primary-reset-at": "1790500000" }, "", NOW)).toBe(1_790_500_000_000)
    expect(parseReset({}, json({ error: { resets_at: 1790500123 } }), NOW)).toBe(1_790_500_123_000)
    expect(parseReset({}, json({ details: [{ retryDelay: "12.5s" }] }), NOW)).toBe(NOW + 12_500)
    expect(parseReset({}, "Please try again in 6m0s.", NOW)).toBe(NOW + 360_000)
    expect(parseReset({}, "You will regain access on 2026-10-01 at 00:00 UTC", NOW)).toBe(
      Date.parse("2026-10-01T00:00:00Z"),
    )
    expect(parseReset({}, "nothing to see", NOW)).toBeUndefined()
  })

  test("durations", () => {
    expect(duration("250ms")).toBe(250)
    expect(duration("1h2m3.5s")).toBe(3_723_500)
    expect(duration("0s")).toBe(0)
    expect(duration("soon")).toBeUndefined()
  })

  test("provider families", () => {
    expect(
      [
        "anthropic",
        "google-vertex-anthropic",
        "kimi-for-coding",
        "moonshotai-cn",
        "alibaba-cn",
        "openrouter",
        "azure",
      ].map(familyOf),
    ).toEqual(["anthropic", "anthropic", "kimi", "kimi", "alibaba", "openrouter", "openai"])
  })
})

describe("failures as the harness records them", () => {
  test("words in content don't make a network failure, and 'install' is not a stall", () => {
    const answered = {
      name: "APIError",
      data: { message: "Invalid schema for function 'browser_network_requests'", statusCode: 400 },
    }
    const quoted = {
      name: "AI_TypeValidationError",
      data: { message: 'Type validation failed:\nValue: {"text":"check the network socket"}' },
    }
    const install = { name: "UnknownError", data: { message: "npm install failed" } }
    for (const error of [answered, quoted, install])
      expect(FailoverFailure.failureOf(error as never, "openai").kind).toBeUndefined()
  })

  test("a refused or cut connection is a network failure that clears with time, however it is worded", () => {
    const api = {
      name: "APIError",
      data: {
        message: "Cannot connect to API: Unable to connect. Is the computer able to access the url?",
        isRetryable: true,
      },
    }
    const bun = {
      name: "UnknownError",
      data: { message: "Unable to connect. Is the computer able to access the url?" },
    }
    // a stream cut mid-answer: Bun raises this while the body is read, so it is not an APIError
    const cut = {
      name: "UnknownError",
      data: {
        message:
          "The socket connection was closed unexpectedly. For more information, pass `verbose: true` in the second argument to fetch()",
      },
    }
    for (const error of [api, bun, cut]) {
      const failure = FailoverFailure.failureOf(error as never, "ollama")
      expect(failure.kind).toBe("network")
      expect(FailoverClassify.clearsWithTime(classify(failure, NOW))).toBe(true)
    }
  })
})
