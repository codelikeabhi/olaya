import type { Argv } from "yargs"
import { existsSync } from "fs"
import { Effect } from "effect"
import { InstallationVersion } from "@olaya/core/installation/version"
import { which } from "@olaya/core/util/which"
import { effectCmd } from "../effect-cmd"
import { UI } from "../ui"
import { Process } from "@/util/process"
import { resolve } from "../../laya/config"
import { launch, setupPlan, sourceDir, venvDir, venvPython } from "../../laya/runtime"

const SetupCommand = effectCmd({
  command: "setup",
  describe: "install the decision layer's local runtime (Python, torch and the Laya model code)",
  instance: false,
  builder: (yargs: Argv) => yargs,
  handler: Effect.fn("Cli.laya.setup")(function* () {
    const plan = setupPlan({
      uv: which("uv") ?? undefined,
      python3: which("python3") ?? undefined,
      venv: venvDir(),
      source: sourceDir(),
      version: InstallationVersion,
    })
    if ("error" in plan) {
      UI.error(`olaya laya setup ${plan.error}`)
      process.exitCode = 1
      return
    }
    for (const step of plan) {
      UI.println(UI.Style.TEXT_DIM + "$ " + step.join(" ") + UI.Style.TEXT_NORMAL)
      const code = yield* Effect.promise(() => Process.spawn(step, { stdout: "inherit", stderr: "inherit" }).exited)
      if (code !== 0) {
        UI.error(`step failed (exit ${code}); nothing else was run`)
        process.exitCode = code
        return
      }
    }
    UI.println("")
    UI.println(`Decision layer installed in ${venvDir()}`)
    UI.println("Turn it on with OLAYA_LAYA_ENABLED=1 (Observe mode; the model downloads on first use).")
  }),
})

const StatusCommand = effectCmd({
  command: "status",
  describe: "show how the decision layer is configured and whether its runtime works",
  instance: false,
  builder: (yargs: Argv) => yargs,
  handler: Effect.fn("Cli.laya.status")(function* () {
    const config = resolve()
    const how = launch(config.python, process.env, sourceDir())
    const probe = [how.argv[0]!, ...(how.argv.includes("-I") ? ["-I"] : []), "-c", "import service, laya"]
    const result = yield* Effect.promise(() => Process.run(probe, { cwd: how.cwd, nothrow: true }))
    const row = (label: string, value: string) => UI.println(`${label.padEnd(10)} ${value}`)
    row("enabled", config.enabled ? "yes" : "no (set OLAYA_LAYA_ENABLED=1)")
    row("mode", config.mode === "live" ? "auto-approve" : "observe")
    row("sidecar", config.url ? `external, ${config.url}` : how.argv.join(" ") + (how.cwd ? `  (in ${how.cwd})` : ""))
    row("runtime", existsSync(venvPython()) ? venvDir() : "not installed (run: olaya laya setup)")
    row("imports", result.code === 0 ? "ok" : `failed: ${result.stderr.toString().trim().split("\n").at(-1) ?? ""}`)
    if (!config.url && result.code !== 0) process.exitCode = 1
  }),
})

export const LayaCommand = effectCmd({
  command: "laya",
  describe: "set up and inspect the decision layer",
  instance: false,
  builder: (yargs: Argv) => yargs.command(SetupCommand).command(StatusCommand).demandCommand(),
  handler: Effect.fn("Cli.laya")(function* () {}),
})
