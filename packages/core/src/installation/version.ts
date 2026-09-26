declare global {
  const OLAYA_VERSION: string
  const OLAYA_CHANNEL: string
}

export const InstallationVersion = typeof OLAYA_VERSION === "string" ? OLAYA_VERSION : "local"
export const InstallationChannel = typeof OLAYA_CHANNEL === "string" ? OLAYA_CHANNEL : "local"
export const InstallationLocal = InstallationChannel === "local"
