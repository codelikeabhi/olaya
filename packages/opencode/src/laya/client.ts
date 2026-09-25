/**
 * Client for the Laya decision sidecar.
 *
 * Every failure mode resolves to `{ ok: false }`. There is deliberately no path by which a
 * transport problem, a timeout, a warming model, or a malformed response can be read as an
 * approval: the caller's only affirmative signal is a probability that arrived intact.
 */

export interface Question {
  type: "noul" | "choice" | "score"
  instructions: string
  criteria?: unknown
}

export interface Health {
  ready: boolean
  checkpoint: string
  error?: string
  device?: string
  max_len?: number
  head_max_len?: number
  state_budget?: number
}

export type FailureReason = "disabled" | "not-ready" | "timeout" | "transport" | "malformed" | "out-of-range" | "http"

export type Judgment =
  | { ok: true; probabilities: Record<string, number>; latencyMs: number; checkpoint: string; inputTokens?: number }
  | { ok: false; reason: FailureReason; detail?: string }

export class LayaClient {
  private budgetCache?: number

  constructor(
    private readonly baseUrl: string,
    private readonly timeoutMs: number,
  ) {}

  async health(timeoutMs = this.timeoutMs): Promise<Health | undefined> {
    try {
      const res = await fetch(new URL("/health", this.baseUrl), { signal: AbortSignal.timeout(timeoutMs) })
      if (!res.ok) return undefined
      return (await res.json()) as Health
    } catch {
      return undefined
    }
  }

  /** State budget in tokens, read from the checkpoint. Cached once the model reports ready. */
  async stateBudget(): Promise<number | undefined> {
    if (this.budgetCache !== undefined) return this.budgetCache
    const health = await this.health()
    // Only cache a ready model's budget - a warming sidecar reports no budget, and caching
    // that absence would pin us to a fallback for the life of the process.
    if (health?.ready && typeof health.state_budget === "number") this.budgetCache = health.state_budget
    return this.budgetCache
  }

  async decide(state: unknown, questions: Record<string, Question>): Promise<Judgment> {
    let res: Response
    try {
      res = await fetch(new URL("/decide", this.baseUrl), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, questions }),
        signal: AbortSignal.timeout(this.timeoutMs),
      })
    } catch (error) {
      const name = error instanceof Error ? error.name : ""
      // TimeoutError is what AbortSignal.timeout raises; anything else is transport.
      return { ok: false, reason: name === "TimeoutError" ? "timeout" : "transport" }
    }

    if (res.status === 503) return { ok: false, reason: "not-ready" }
    if (!res.ok) return { ok: false, reason: "http", detail: String(res.status) }

    let body: any
    try {
      body = await res.json()
    } catch {
      return { ok: false, reason: "malformed" }
    }
    if (!body || typeof body !== "object" || typeof body.answers !== "object" || body.answers === null) {
      return { ok: false, reason: "malformed" }
    }

    const probabilities: Record<string, number> = {}
    for (const qid of Object.keys(questions)) {
      const answer = body.answers[qid]
      if (!answer || typeof answer !== "object") return { ok: false, reason: "malformed" }
      const p = (answer as { probability?: unknown }).probability
      if (typeof p !== "number" || !Number.isFinite(p)) return { ok: false, reason: "malformed" }
      if (p < 0 || p > 1) return { ok: false, reason: "out-of-range" }
      probabilities[qid] = p
    }

    return {
      ok: true,
      probabilities,
      latencyMs: typeof body.latency_ms === "number" ? body.latency_ms : 0,
      checkpoint: typeof body.checkpoint === "string" ? body.checkpoint : "unknown",
      inputTokens: body.usage?.input_tokens,
    }
  }
}
