import { Schema } from "effect"
import { NamedError } from "@olaya/core/util/error"
import { Process } from "@/util/process"
import { IdeEvent } from "@olaya/schema/ide-event"

const SUPPORTED_IDES = [
  { name: "Windsurf" as const, cmd: "windsurf" },
  { name: "Visual Studio Code - Insiders" as const, cmd: "code-insiders" },
  { name: "Visual Studio Code" as const, cmd: "code" },
  { name: "Cursor" as const, cmd: "cursor" },
  { name: "VSCodium" as const, cmd: "codium" },
]

export const Event = IdeEvent

export const AlreadyInstalledError = NamedError.create("AlreadyInstalledError", {})

export const InstallFailedError = NamedError.create("InstallFailedError", {
  stderr: Schema.String,
})

export function ide() {
  if (process.env["TERM_PROGRAM"] === "vscode") {
    const v = process.env["GIT_ASKPASS"]
    for (const ide of SUPPORTED_IDES) {
      if (v?.includes(ide.name)) return ide.name
    }
  }
  return "unknown"
}

export function alreadyInstalled() {
  return process.env["OLAYA_CALLER"] === "vscode" || process.env["OLAYA_CALLER"] === "vscode-insiders"
}

/**
 * Olaya's editor extension, once it is on the Marketplace ("olaya-ai.olaya"). Until then install
 * refuses: fetching by a name nobody here has published would install whatever a stranger
 * uploads under it.
 */
export const EXTENSION_ID: string | undefined = undefined

export async function install(ide: (typeof SUPPORTED_IDES)[number]["name"]) {
  const cmd = SUPPORTED_IDES.find((i) => i.name === ide)?.cmd
  if (!cmd) throw new Error(`Unknown IDE: ${ide}`)
  if (!EXTENSION_ID) {
    throw new InstallFailedError({
      stderr: "The Olaya editor extension is not published yet. Build it from sdks/vscode in the Olaya repository.",
    })
  }

  const p = await Process.run([cmd, "--install-extension", EXTENSION_ID], {
    nothrow: true,
  })
  const stdout = p.stdout.toString()
  const stderr = p.stderr.toString()

  if (p.code !== 0) {
    throw new InstallFailedError({ stderr })
  }
  if (stdout.includes("already installed")) {
    throw new AlreadyInstalledError({})
  }
}

export * as Ide from "."
