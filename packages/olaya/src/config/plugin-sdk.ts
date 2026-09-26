/**
 * The plugin SDK package that user plugin directories (.olaya/, the global config dir) depend
 * on, so plugins written there can import its types.
 *
 * Olaya's own SDK is not on npm yet, and pinning Olaya's version would ask npm for a package
 * version that does not exist: the install fails and the process exits non-zero on shutdown.
 * So user dirs get upstream's SDK at the release Olaya is based on: the same plugin API, and it
 * resolves. Switch to { name: "@olaya/plugin", version: InstallationVersion } once published.
 */
export const PluginSDK = {
  name: "@opencode-ai/plugin", // olaya-rename:keep
  version: "1.18.32",
} as const
