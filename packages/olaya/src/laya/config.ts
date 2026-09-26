import { python as resolvePython } from "./runtime"

/**
 * Configuration for the Laya decision layer.
 *
 * Read from the environment and from plugin options rather than from olaya's config
 * schema. Extending that schema would spread Olaya's diff across several upstream files;
 * the whole integration is meant to touch exactly one.
 */

export interface LayaConfig {
  /** Master switch. Nothing runs, spawns, or is logged unless this is on. */
  enabled: boolean
  /** Record predictions and labels. Requires `enabled`. Opt-in: off by default. */
  shadow: boolean
  /**
   * "shadow" observes only. "live" may turn an `ask` into `allow`, and only when the loaded
   * checkpoint carries a passed certification gate; otherwise it behaves exactly like shadow.
   */
  mode: "shadow" | "live"
  /** Base URL of an externally managed sidecar. When set, Olaya does not spawn one. */
  url?: string
  /** Hard ceiling on a single decision request. */
  timeoutMs: number
  /** Python interpreter used to launch the sidecar: explicit, else `olaya laya setup`'s, else python3. */
  python: string
  /** Checkpoint id passed through to the sidecar. */
  checkpoint?: string
  /**
   * Extractive retention at compaction points. Independent of `enabled`: it needs no sidecar.
   * "shadow" logs the plan only; "live" replaces the model-written summary. Uncertified until
   * gate G9's A/B has run, so it is reachable from the environment alone.
   */
  retention: "off" | "shadow" | "live"
  /** Share of the replaced history's tokens retention keeps. */
  retentionBudget: number
  /**
   * Model routing within the user's pool. Independent of `enabled`. "shadow" logs decisions only;
   * "live" applies them when the user has turned routing on. Uncertified until gate G6.
   */
  routing: "off" | "shadow" | "live"
  /** Where a task starts: the prompt's model, or the cheapest in the pool (a cascade). */
  routingStart: "default" | "cheapest"
}

const DEFAULTS: LayaConfig = {
  enabled: false,
  shadow: false,
  mode: "shadow",
  // Steady state is 40-90 ms, but MPS recompiles on each new sequence shape and that
  // first hit costs 200-700 ms. A tight ceiling turns those into lost judgments - and in
  // shadow mode a lost judgment is lost training data - so the ceiling is generous. It is a
  // bound, not a typical cost, and exceeding it just falls back to asking the user.
  timeoutMs: 1500,
  python: "python3",
  retention: "off",
  retentionBudget: 0.2,
  routing: "off",
  routingStart: "default",
}

function bool(value: string | undefined): boolean | undefined {
  if (value === undefined) return undefined
  return ["1", "true", "yes", "on"].includes(value.toLowerCase())
}

function int(value: string | undefined): number | undefined {
  if (value === undefined) return undefined
  const n = Number.parseInt(value, 10)
  return Number.isFinite(n) && n > 0 ? n : undefined
}

function fraction(value: string | undefined): number | undefined {
  const n = Number(value)
  return value !== undefined && n > 0 && n <= 1 ? n : undefined
}

/**
 * The mode's user-facing names are Observe and Auto-approve; `shadow` and `live` are the original
 * spellings. Only an exact match selects a mode: anything else falls back to the default, Observe.
 */
export function parseMode(value: unknown): LayaConfig["mode"] | undefined {
  if (value === "live" || value === "auto-approve") return "live"
  if (value === "shadow" || value === "observe") return "shadow"
  return undefined
}

/** Plugin options win over the environment, which wins over defaults. */
export function resolve(options: Record<string, unknown> = {}, env = process.env): LayaConfig {
  const fromEnv: Partial<LayaConfig> = {
    enabled: bool(env["OLAYA_LAYA_ENABLED"]),
    shadow: bool(env["OLAYA_LAYA_SHADOW"]),
    mode: parseMode(env["OLAYA_LAYA_MODE"]),
    url: env["OLAYA_LAYA_URL"] || undefined,
    timeoutMs: int(env["OLAYA_LAYA_TIMEOUT_MS"]),
    python: env["OLAYA_LAYA_PYTHON"] || undefined,
    checkpoint: env["OLAYA_LAYA_MODEL"] || undefined,
    retention: (["shadow", "live"] as const).find((mode) => mode === env["OLAYA_LAYA_RETENTION"]),
    retentionBudget: fraction(env["OLAYA_LAYA_RETENTION_BUDGET"]),
    routing: (["shadow", "live"] as const).find((mode) => mode === env["OLAYA_LAYA_ROUTING"]),
    routingStart: env["OLAYA_LAYA_ROUTING_START"] === "cheapest" ? "cheapest" : undefined,
  }
  const merged = { ...DEFAULTS }
  const fromOptions = { ...options, mode: parseMode(options["mode"]) } as Partial<LayaConfig>
  for (const source of [fromEnv, fromOptions]) {
    for (const [key, value] of Object.entries(source)) {
      if (value !== undefined) (merged as Record<string, unknown>)[key] = value
    }
  }
  merged.python = resolvePython((options["python"] as string | undefined) || env["OLAYA_LAYA_PYTHON"] || undefined)
  // Shadow logging is meaningless with the layer off, and enabling it alone must not
  // silently start writing records.
  if (!merged.enabled) merged.shadow = false
  return merged
}
