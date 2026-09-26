import { Config } from "effect"

/**
 * Olaya was OpenCode. For one release cycle, every OPENCODE_* variable is honoured as its (olaya-rename:keep)
 * OLAYA_* equivalent when that is unset, so existing setups (shell profiles, CI, the VS Code
 * extension, Harbor's OPENCODE_FAKE_VCS) keep working (olaya-rename:keep). Copying into process.env here, the
 * first module to read the environment, covers every later reader too, not just this file.
 * OPENCODE_API_KEY is not ours: it is the OpenCode Zen provider's key, defined by models.dev.
 */
export function adoptLegacyEnv(env: Record<string, string | undefined> = process.env) {
  for (const [key, value] of Object.entries(env)) {
    if (!key.startsWith("OPENCODE_") || key === "OPENCODE_API_KEY") continue // olaya-rename:keep
    const renamed = "OLAYA_" + key.slice("OPENCODE_".length) // olaya-rename:keep
    if (env[renamed] === undefined) env[renamed] = value
  }
  if (env["OPENCODE"] !== undefined && env["OLAYA"] === undefined) env["OLAYA"] = env["OPENCODE"] // olaya-rename:keep
}
adoptLegacyEnv()

export function truthy(key: string) {
  const value = process.env[key]?.toLowerCase()
  return value === "true" || value === "1"
}

const copy = process.env["OLAYA_EXPERIMENTAL_DISABLE_COPY_ON_SELECT"]
const fff = process.env["OLAYA_DISABLE_FFF"]

function enabledByExperimental(key: string) {
  return process.env[key] === undefined ? truthy("OLAYA_EXPERIMENTAL") : truthy(key)
}

export const Flag = {
  OTEL_EXPORTER_OTLP_ENDPOINT: process.env["OTEL_EXPORTER_OTLP_ENDPOINT"],
  OTEL_EXPORTER_OTLP_HEADERS: process.env["OTEL_EXPORTER_OTLP_HEADERS"],

  OLAYA_AUTO_HEAP_SNAPSHOT: truthy("OLAYA_AUTO_HEAP_SNAPSHOT"),
  OLAYA_GIT_BASH_PATH: process.env["OLAYA_GIT_BASH_PATH"],
  OLAYA_CONFIG: process.env["OLAYA_CONFIG"],
  OLAYA_CONFIG_CONTENT: process.env["OLAYA_CONFIG_CONTENT"],
  OLAYA_DISABLE_AUTOUPDATE: truthy("OLAYA_DISABLE_AUTOUPDATE"),
  OLAYA_ALWAYS_NOTIFY_UPDATE: truthy("OLAYA_ALWAYS_NOTIFY_UPDATE"),
  OLAYA_DISABLE_PRUNE: truthy("OLAYA_DISABLE_PRUNE"),
  OLAYA_DISABLE_TERMINAL_TITLE: truthy("OLAYA_DISABLE_TERMINAL_TITLE"),
  OLAYA_SHOW_TTFD: truthy("OLAYA_SHOW_TTFD"),
  OLAYA_DISABLE_AUTOCOMPACT: truthy("OLAYA_DISABLE_AUTOCOMPACT"),
  OLAYA_DISABLE_MODELS_FETCH: truthy("OLAYA_DISABLE_MODELS_FETCH"),
  OLAYA_DISABLE_MOUSE: truthy("OLAYA_DISABLE_MOUSE"),
  OLAYA_FAKE_VCS: process.env["OLAYA_FAKE_VCS"],
  OLAYA_SERVER_PASSWORD: process.env["OLAYA_SERVER_PASSWORD"],
  OLAYA_SERVER_USERNAME: process.env["OLAYA_SERVER_USERNAME"],
  OLAYA_DISABLE_FFF: fff === undefined ? process.platform === "win32" : truthy("OLAYA_DISABLE_FFF"),

  // Experimental
  OLAYA_EXPERIMENTAL_FILEWATCHER: Config.boolean("OLAYA_EXPERIMENTAL_FILEWATCHER").pipe(
    Config.withDefault(false),
  ),
  OLAYA_EXPERIMENTAL_DISABLE_FILEWATCHER: Config.boolean("OLAYA_EXPERIMENTAL_DISABLE_FILEWATCHER").pipe(
    Config.withDefault(false),
  ),
  OLAYA_EXPERIMENTAL_DISABLE_COPY_ON_SELECT:
    copy === undefined ? process.platform === "win32" : truthy("OLAYA_EXPERIMENTAL_DISABLE_COPY_ON_SELECT"),
  OLAYA_MODELS_URL: process.env["OLAYA_MODELS_URL"],
  OLAYA_MODELS_PATH: process.env["OLAYA_MODELS_PATH"],
  OLAYA_DB: process.env["OLAYA_DB"],

  OLAYA_WORKSPACE_ID: process.env["OLAYA_WORKSPACE_ID"],
  OLAYA_EXPERIMENTAL_WORKSPACES: enabledByExperimental("OLAYA_EXPERIMENTAL_WORKSPACES"),

  // Evaluated at access time (not module load) because tests, the CLI, and
  // external tooling set these env vars at runtime.
  get OLAYA_DISABLE_PROJECT_CONFIG() {
    return truthy("OLAYA_DISABLE_PROJECT_CONFIG")
  },
  get OLAYA_EXPERIMENTAL_REFERENCES() {
    return enabledByExperimental("OLAYA_EXPERIMENTAL_REFERENCES")
  },
  get OLAYA_TUI_CONFIG() {
    return process.env["OLAYA_TUI_CONFIG"]
  },
  get OLAYA_CONFIG_DIR() {
    return process.env["OLAYA_CONFIG_DIR"]
  },
  get OLAYA_PURE() {
    return truthy("OLAYA_PURE")
  },
  get OLAYA_PERMISSION() {
    return process.env["OLAYA_PERMISSION"]
  },
  get OLAYA_PLUGIN_META_FILE() {
    return process.env["OLAYA_PLUGIN_META_FILE"]
  },
  get OLAYA_CLIENT() {
    return process.env["OLAYA_CLIENT"] ?? "cli"
  },
}
