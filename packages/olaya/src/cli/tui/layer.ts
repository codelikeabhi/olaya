import { run as runTui, type TuiInput } from "@olaya/tui"
import { Global } from "@olaya/core/global"
import { AppNodeBuilder } from "@olaya/core/effect/app-node-builder"
import { Effect } from "effect"

export function run(input: TuiInput) {
  return runTui(input).pipe(Effect.provide(AppNodeBuilder.build(Global.node)))
}
