/**
 * Laya's model router (laya-model-routing). Until the routing head (gate G4) exists, this is the
 * deterministic part of the design only:
 * - D0, task start: the prompt's model, or with `start: "cheapest"` the cheapest model in the
 *   user's pool (a cascade; opt-in until Track C shows it keeps quality);
 * - D3, hard triggers: one step up the pool by price after a doom loop, 2 malformed calls in a
 *   row, no test progress over 3 runs, or 4 edits to one file since the last test run; and a failed
 *   "done": the model ends its turn while its last test run failed (`failedDone`, at loop exit);
 * - D4, compaction: stay (the ratchet never steps down).
 *
 * Restart-smart (design D4): escalating with a large context would send the whole history, uncached,
 * to a pricier model. Live, the escalation instead waits for a compaction it asks for, and the
 * stronger model starts from the rebuilt, small context. If no compaction comes within two steps
 * (auto compaction off), it escalates anyway.
 *
 * Guardrails (spec "Guardrails"): at most 2 escalations per task, at least 10 steps between
 * switches, and no switch while an edit waits for its test run; a trigger then waits for the
 * next safe step. Shadow mode records every decision and changes nothing. Live mode (uncertified
 * until G6) acts, and only when the user has turned routing on.
 */

import type { Hooks, PoolModel, StepUsage } from "@olaya/plugin"
import type { ShadowLog } from "./shadow"

export type RoutingOptions = { mode: "shadow" | "live"; start: "default" | "cheapest"; shadow?: ShadowLog }

type Input = Parameters<NonNullable<Hooks["experimental.model.select"]>>[0]

type State = {
  /** Index into the price-ordered pool, or -1 while on a model outside it. */
  tier: number
  escalations: number
  lastSwitch: number
  malformedRun: number
  failures: number[]
  pending?: string
  /** An escalation held for a compaction (restart-smart): the tier, why, and when it was held. */
  restart?: { tier: number; reason: string; step: number }
  /** Live and turned on by the user: only then does a held escalation ask for a compaction. */
  live?: boolean
  /** Models in the user's pool, as of the last step. */
  poolSize?: number
}

const MAX_ESCALATIONS = 2
const MIN_STEPS_BETWEEN_SWITCHES = 10
/** Context from which an escalation restarts from a compaction instead of carrying the history. */
export const RESTART_TOKENS = 40_000

export const byPrice = (pool: PoolModel[]) =>
  [...pool].sort((a, b) => a.cost.output - b.cost.output || a.cost.input - b.cost.input)

const same = (a: { providerID: string; modelID: string }, b: { providerID: string; modelID: string }) =>
  a.providerID === b.providerID && a.modelID === b.modelID

/** The first hard trigger this step's usage fires, updating the running counters. */
export function trigger(state: State, usage: StepUsage | undefined) {
  if (!usage) return undefined
  const s = usage.signals
  state.malformedRun = s.malformed > 0 ? state.malformedRun + 1 : 0
  if (s.tests) state.failures.push(s.tests.failed)
  const last3 = state.failures.slice(-3)
  if (s.identicalRun >= 3) return "doom-loop"
  if (state.malformedRun >= 2) return "malformed-calls"
  if (last3.length === 3 && last3[2]! > 0 && last3[2]! >= last3[0]!) return "no-test-progress"
  if (s.sameFileEdits >= 4) return "edits-without-tests"
  return undefined
}

/** Decides the model for one step. Returns the pool index to run on, or undefined to leave it. */
export function decide(state: State, input: Input, start: RoutingOptions["start"]) {
  const pool = byPrice(input.routing.pool)
  state.poolSize = pool.length
  if (pool.length < 2) return { reason: "pool has fewer than two models" }
  if (input.point === "start") {
    // only a real change of model starts the spacing between switches
    if (start === "cheapest") {
      if (!same(pool[0]!, input.model)) state.lastSwitch = input.step
      return { tier: (state.tier = 0), reason: "start: cheapest in pool" }
    }
    state.tier = pool.findIndex((m) => same(m, input.model))
    return { reason: "start: prompt's model" }
  }
  // steps count per run, so a hold from an earlier user turn has a larger step than this one
  const due = state.restart && (input.step - state.restart.step >= 2 || input.step < state.restart.step)
  if (state.restart && (input.point === "compaction" || due)) {
    const held = state.restart
    state.restart = undefined
    state.tier = held.tier
    state.escalations++
    state.lastSwitch = input.step
    return {
      tier: held.tier,
      reason:
        input.point === "compaction" ? `${held.reason} (after compaction)` : `${held.reason} (no compaction came)`,
    }
  }
  if (state.restart) return { reason: `${state.restart.reason}: waiting for the compaction` }
  state.pending = trigger(state, input.usage) ?? state.pending
  if (!state.pending) return { reason: "no trigger" }
  // an edit is waiting for its test run: switching now would hand a half-done change to another
  // model. Piled-up edits without tests are themselves the trigger, so they are not held back, and
  // neither is a failed "done": the model has stopped, so no change is half-way.
  const done = state.pending === "failed-done"
  if (
    (input.usage?.signals.sameFileEdits ?? 0) > 0 &&
    !input.usage?.signals.tests &&
    state.pending !== "edits-without-tests" &&
    !done
  )
    return { reason: `${state.pending}: deferred, edit awaiting its test run` }
  if (state.escalations >= MAX_ESCALATIONS) return { reason: `${state.pending}: escalation cap reached` }
  if (input.step - state.lastSwitch < MIN_STEPS_BETWEEN_SWITCHES && !done)
    return { reason: `${state.pending}: too soon after the last switch` }
  // one step up from the current model: above its pool position, or above its price when outside the pool
  const current = state.tier >= 0 ? pool[state.tier]! : undefined
  const price = current?.cost.output ?? pool.find((m) => same(m, input.model))?.cost.output ?? -1
  const next = state.tier >= 0 ? state.tier + 1 : pool.findIndex((m) => m.cost.output > price)
  if (next < 0 || next >= pool.length) return { reason: `${state.pending}: already on the strongest model` }
  const reason = `escalate: ${state.pending}`
  state.pending = undefined
  const context = input.usage
    ? input.usage.tokens.input + input.usage.tokens.cacheRead + input.usage.tokens.cacheWrite
    : 0
  if (state.live && input.point === "step" && context >= RESTART_TOKENS) {
    state.restart = { tier: next, reason, step: input.step }
    return { reason: `${reason}: restart-smart, compacting first` }
  }
  state.tier = next
  state.escalations++
  state.lastSwitch = input.step
  return { tier: next, reason }
}

/** The `experimental.model.select` handler. Any failure leaves the harness's model. */
export function handler(
  options: RoutingOptions,
  // ponytail: per-process memory, one small record per session; a restarted process starts a
  // session's routing afresh at its next step, from the model it is on.
  sessions = new Map<string, State>(),
): NonNullable<Hooks["experimental.model.select"]> {
  return async (input, output) => {
    try {
      const state = sessions.get(input.sessionID) ?? {
        tier: -1,
        escalations: 0,
        lastSwitch: -Infinity,
        malformedRun: 0,
        failures: [],
      }
      sessions.set(input.sessionID, state)
      state.live = options.mode === "live" && input.routing.enabled
      const t0 = performance.now()
      const decision = decide(state, input, options.start)
      // The harness resolves the prompt's model at every step, so the routed model is restated
      // each step for as long as routing keeps it.
      const pool = byPrice(input.routing.pool)
      const chosen = state.tier >= 0 && pool.length >= 2 ? pool[state.tier] : undefined
      const acts = options.mode === "live" && input.routing.enabled && chosen && !same(chosen, input.model)
      if (acts) {
        output.model = { providerID: chosen.providerID, modelID: chosen.modelID }
        output.reason = decision.reason
      }
      await options.shadow?.route({
        sessionID: input.sessionID,
        step: input.step,
        point: input.point,
        mode: options.mode,
        enabled: input.routing.enabled,
        from: `${input.model.providerID}/${input.model.modelID}`,
        ...(chosen && { choice: `${chosen.providerID}/${chosen.modelID}` }),
        applied: Boolean(acts),
        reason: decision.reason,
        ms: Math.round((performance.now() - t0) * 1000) / 1000,
      })
    } catch (error) {
      console.error("laya routing failed; the harness's model is used:", error)
    }
  }
}

export const FAILED_DONE_PROMPT =
  "The tests still fail, so the task isn't done. You now run on a stronger model: find the cause, fix it, and run the tests until they pass."

/**
 * A failed "done" (design D4): the model ends its turn while its last test run failed. Live and
 * turned on, below the strongest model and under the escalation cap, the turn goes on, and the next
 * step escalates. Shares `handler`'s sessions.
 */
export function failedDone(sessions: Map<string, State>): NonNullable<Hooks["experimental.loop.exit"]> {
  return async (input, output) => {
    const state = sessions.get(input.sessionID)
    // a trigger still held back (say, an edit awaiting its test run) is replaced: the model has stopped
    if (output.continue || !state?.live || state.restart) return
    if (state.tier < 0 || state.tier >= (state.poolSize ?? 0) - 1 || state.escalations >= MAX_ESCALATIONS) return
    if (!((state.failures.at(-1) ?? 0) > 0)) return
    state.pending = "failed-done"
    output.continue = true
    output.prompt = FAILED_DONE_PROMPT
  }
}

/** Asks for a compaction while an escalation waits for one (restart-smart); shares `handler`'s sessions. */
export function restartPoint(
  sessions: Map<string, State>,
): NonNullable<Hooks["experimental.session.compaction.point"]> {
  return async (input, output) => {
    const state = sessions.get(input.sessionID)
    if (state?.live && state.restart) output.compact = true
  }
}

export type RoutingState = State
