/**
 * Where the decision layer's Python sidecar lives, and how to install it.
 *
 * Three ways to run it, in this order:
 *   1. OLAYA_LAYA_DIR: a checkout of `laya/` you point at yourself (runs `service.py` there);
 *   2. the source tree, when olaya itself runs from source (`bun dev`);
 *   3. the managed runtime: a virtualenv under Olaya's data dir with the `olaya-laya` package
 *      installed, created by `olaya laya setup` (runs `python -m service`).
 *
 * A compiled binary has no source tree (its import.meta.dir is Bun's virtual filesystem),
 * so installed copies of Olaya always use (1) or (3).
 */

import { existsSync } from "fs"
import os from "os"
import path from "path"
import { Global } from "@olaya/core/global"

export const REPO = "codelikeabhi/olaya"

export function venvDir(data = Global.Path.data): string {
  return path.join(data, "laya", "venv")
}

export function venvPython(venv = venvDir(), platform = process.platform): string {
  return platform === "win32" ? path.join(venv, "Scripts", "python.exe") : path.join(venv, "bin", "python")
}

/** The `laya/` directory of a source checkout, when olaya is running from one. */
export function sourceDir(metaDir = import.meta.dir): string | undefined {
  const dir = path.resolve(metaDir, "../../../../laya")
  return existsSync(path.join(dir, "service.py")) ? dir : undefined
}

/** The interpreter to launch: an explicit choice, else the managed runtime, else python3. */
export function python(explicit: string | undefined, managed = venvPython()): string {
  if (explicit) return explicit
  return existsSync(managed) ? managed : "python3"
}

export type Launch = { argv: string[]; cwd?: string }

/**
 * How to start the sidecar with a given interpreter. The installed form runs isolated (-I) from
 * a neutral directory: `python -m` otherwise puts the working directory first on sys.path, and
 * a repository containing its own service.py would be executed in place of the sidecar.
 */
export function launch(py: string, env: NodeJS.ProcessEnv, source: string | undefined): Launch {
  const dir = env["OLAYA_LAYA_DIR"] || source
  if (dir) return { argv: [py, "service.py"], cwd: dir }
  return { argv: [py, "-I", "-m", "service"], cwd: os.tmpdir() }
}

/**
 * The commands `olaya laya setup` runs. Pure, so the choice of installer, interpreter and
 * package source is testable without installing a 700 MB torch.
 */
export function setupPlan(input: {
  uv?: string
  python3?: string
  venv: string
  source?: string
  version: string
  platform?: NodeJS.Platform
}): string[][] | { error: string } {
  const py = venvPython(input.venv, input.platform)
  // A release installs its own tagged source; a dev build installs main.
  const ref = /^\d+\.\d+\.\d+$/.test(input.version) ? `v${input.version}` : "main"
  const pkg = input.source ?? `olaya-laya @ git+https://github.com/${REPO}@${ref}#subdirectory=laya`
  const editable = input.source ? ["-e"] : []
  if (input.uv) {
    return [
      [input.uv, "venv", "--python", "3.12", input.venv],
      // --torch-backend auto picks CPU wheels where there is no GPU, instead of multi-GB CUDA ones
      [input.uv, "pip", "install", "--python", py, "--torch-backend", "auto", ...editable, pkg],
    ]
  }
  if (input.python3) {
    return [
      [input.python3, "-m", "venv", input.venv],
      [py, "-m", "pip", "install", "--upgrade", "pip"],
      [py, "-m", "pip", "install", ...editable, pkg],
    ]
  }
  return { error: "needs uv (https://docs.astral.sh/uv/) or Python 3.10-3.12 on PATH" }
}
