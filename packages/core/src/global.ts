import path from "path"
import fs from "fs/promises"
import { xdgData, xdgCache, xdgConfig, xdgState } from "xdg-basedir"
import os from "os"
import { Context, Effect, Layer } from "effect"
import { Flock } from "./util/flock"
import { Flag } from "./flag/flag"
import { makeGlobalNode } from "./effect/app-node"

const app = "olaya"
const data = path.join(xdgData!, app)
const cache = path.join(xdgCache!, app)
const config = path.join(xdgConfig!, app)
const state = path.join(xdgState!, app)
const tmp = path.join(os.tmpdir(), app)

const paths = {
  get home() {
    return process.env.OLAYA_TEST_HOME ?? os.homedir()
  },
  data,
  bin: path.join(cache, "bin"),
  log: path.join(data, "log"),
  repos: path.join(data, "repos"),
  cache,
  config,
  state,
  tmp,
}

export const Path = paths

/**
 * One-time import of an existing OpenCode install. If Olaya's directory does not exist yet and (olaya-rename:keep)
 * OpenCode's does, copy it (never share it: a user may still run OpenCode, and two apps (olaya-rename:keep)
 * writing one database is how state gets corrupted), rename top-level files that carry the old
 * name (opencode.json -> olaya.json, opencode.db -> olaya.db), and leave a marker. OpenCode's (olaya-rename:keep)
 * copy is not modified. Returns true when an import happened.
 */
export async function adoptLegacyDir(legacy: string, current: string): Promise<boolean> {
  const exists = (p: string) => fs.stat(p).then(() => true, () => false)
  // "Already set up" means real content, not the empty skeleton (and logs) that any earlier
  // run leaves behind, e.g. `olaya --version` before the first real session.
  const hasContent = async (dir: string): Promise<boolean> => {
    for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
      if (entry.name === "log") continue
      if (!entry.isDirectory() || (await hasContent(path.join(dir, entry.name)))) return true
    }
    return false
  }
  if (!(await exists(legacy)) || !(await hasContent(legacy))) return false
  if ((await exists(current)) && (await hasContent(current))) return false
  // Only real state is copied: config, credentials, databases. Trees that regenerate
  // themselves are skipped; they can be large (snapshot repos) and are pinned to OpenCode's (olaya-rename:keep)
  // versions (node_modules installed for its plugin SDK).
  const regenerable = new Set(["node_modules", "snapshot", "log", "repos", "bin", "cache"])
  await fs.cp(legacy, current, {
    recursive: true,
    filter: (src) => !path.relative(legacy, src).split(path.sep).some((part) => regenerable.has(part)),
  })
  for (const name of await fs.readdir(current)) {
    if (!name.startsWith("opencode")) continue // olaya-rename:keep
    const renamed = "olaya" + name.slice("opencode".length) // olaya-rename:keep
    if (!(await exists(path.join(current, renamed)))) await fs.rename(path.join(current, name), path.join(current, renamed))
  }
  await fs.writeFile(path.join(current, ".imported-from-opencode"), new Date().toISOString() + "\n") // olaya-rename:keep
  return true
}

if (!process.env.OLAYA_DISABLE_LEGACY_IMPORT) {
  const legacy = "opencode" // olaya-rename:keep
  const imported = await Promise.all([
    adoptLegacyDir(path.join(xdgConfig!, legacy), config),
    adoptLegacyDir(path.join(xdgData!, legacy), data),
  ])
  if (imported.some(Boolean))
    console.error("olaya: imported your OpenCode configuration and data (OpenCode's own copy is untouched)") // olaya-rename:keep
}

Flock.setGlobal({ state })

await Promise.all([
  fs.mkdir(Path.data, { recursive: true }),
  fs.mkdir(Path.config, { recursive: true }),
  fs.mkdir(Path.state, { recursive: true }),
  fs.mkdir(Path.tmp, { recursive: true }),
  fs.mkdir(Path.log, { recursive: true }),
  fs.mkdir(Path.bin, { recursive: true }),
  fs.mkdir(Path.repos, { recursive: true }),
])

export class Service extends Context.Service<Service, Interface>()("@olaya/Global") {}

export interface Interface {
  readonly home: string
  readonly data: string
  readonly cache: string
  readonly config: string
  readonly state: string
  readonly tmp: string
  readonly bin: string
  readonly log: string
  readonly repos: string
}

export function make(input: Partial<Interface> = {}): Interface {
  return {
    home: Path.home,
    data: Path.data,
    cache: Path.cache,
    config: Flag.OLAYA_CONFIG_DIR ?? Path.config,
    state: Path.state,
    tmp: Path.tmp,
    bin: Path.bin,
    log: Path.log,
    repos: Path.repos,
    ...input,
  }
}

const layer = Layer.effect(
  Service,
  Effect.sync(() => Service.of(make())),
)

export const node = makeGlobalNode({ service: Service, layer: layer, deps: [] })

export const layerWith = (input: Partial<Interface>) =>
  Layer.effect(
    Service,
    Effect.sync(() => Service.of(make(input))),
  )

export * as Global from "./global"
