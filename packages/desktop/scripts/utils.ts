import { $ } from "bun"
import { chmod, copyFile } from "node:fs/promises"
import { join } from "node:path"

export type Channel = "dev" | "beta" | "prod"

export function resolveChannel(): Channel {
  const raw = Bun.env.OLAYA_CHANNEL
  if (raw === "dev" || raw === "beta" || raw === "prod") return raw
  return "dev"
}

export const CLI_BINARIES: Array<{ rustTarget: string; os: string; cpu: string }> = [
  {
    rustTarget: "aarch64-apple-darwin",
    os: "darwin",
    cpu: "arm64",
  },
  {
    rustTarget: "x86_64-apple-darwin",
    os: "darwin",
    cpu: "x64",
  },
  {
    rustTarget: "aarch64-pc-windows-msvc",
    os: "win32",
    cpu: "arm64",
  },
  {
    rustTarget: "x86_64-pc-windows-msvc",
    os: "win32",
    cpu: "x64",
  },
  {
    rustTarget: "x86_64-unknown-linux-gnu",
    os: "linux",
    cpu: "x64",
  },
  {
    rustTarget: "aarch64-unknown-linux-gnu",
    os: "linux",
    cpu: "arm64",
  },
]

export const RUST_TARGET = Bun.env.RUST_TARGET

function nativeTarget() {
  const { platform, arch } = process
  if (platform === "darwin") return arch === "arm64" ? "aarch64-apple-darwin" : "x86_64-apple-darwin"
  if (platform === "win32") return arch === "arm64" ? "aarch64-pc-windows-msvc" : "x86_64-pc-windows-msvc"
  if (platform === "linux") return arch === "arm64" ? "aarch64-unknown-linux-gnu" : "x86_64-unknown-linux-gnu"
  throw new Error(`Unsupported platform: ${platform}/${arch}`)
}

export function getCurrentCli(target = RUST_TARGET ?? nativeTarget()) {
  const binaryConfig = CLI_BINARIES.find((item) => item.rustTarget === target)
  if (!binaryConfig) throw new Error(`CLI configuration not available for target '${target}'`)

  return binaryConfig
}

/**
 * The v2 background CLI (opt-in, OLAYA_SIDECAR_V2=1), built from packages/cli in this repo.
 * Never fetched by package name: Olaya publishes no @olaya/cli-* packages, and a name nobody
 * here owns is a name anyone could publish under.
 */
export async function buildCliToResources() {
  const cli = getCurrentCli()
  const dest = windowsify("resources/olaya-cli")
  const name = `cli-${cli.os === "win32" ? "windows" : cli.os}-${cli.cpu}`
  const binary = cli.os === "win32" ? "lildax.exe" : "lildax"
  // Local dev builds may run on a newer Bun; CI builds with the pinned one and keeps the check.
  const env = process.env.GITHUB_ACTIONS === "true" ? process.env : { ...process.env, OLAYA_ALLOW_BUN_MISMATCH: "1" }
  await $`bun ./script/build.ts --single --skip-install`.cwd("../cli").env(env)
  await copyFile(join("../cli/dist", name, "bin", binary), dest)
  if (process.platform !== "win32") await chmod(dest, 0o755)
  if (process.platform === "win32" && process.env.GITHUB_ACTIONS === "true") {
    await $`pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File ../../script/sign-windows.ps1 ${dest}`
  }
  if (process.platform === "darwin") await $`codesign --force --sign - ${dest}`

  console.log(`Built ${name} into ${dest}`)
}

export function windowsify(path: string) {
  if (path.endsWith(".exe")) return path
  return `${path}${process.platform === "win32" ? ".exe" : ""}`
}
