import { mkdir } from "fs/promises"
import path from "path"
import { PluginSDK } from "../../src/config/plugin-sdk"

// Locks the SDK the config actually installs: a lock naming another package counts as stale and
// starts a real npm install.
export async function markPluginDependenciesReady(dir: string) {
  await mkdir(path.join(dir, "node_modules"), { recursive: true })
  await Bun.write(
    path.join(dir, "package-lock.json"),
    JSON.stringify({ packages: { "": { dependencies: { [PluginSDK.name]: PluginSDK.version } } } }),
  )
}
