import { expect, test } from "bun:test"
import type { Configuration } from "electron-builder"

const legacyDesktopEntry = "resources/linux/olaya-desktop.desktop"

const channels = [
  { channel: "dev", appId: "io.olaya.desktop.dev" },
  { channel: "beta", appId: "io.olaya.desktop.beta" },
  { channel: "prod", appId: "io.olaya.desktop" },
] as const

for (const channel of channels) {
  test(`uses one Linux desktop identity for ${channel.channel}`, async () => {
    const previous = process.env.OLAYA_CHANNEL
    process.env.OLAYA_CHANNEL = channel.channel

    const module = await import(`./electron-builder.config.ts?channel=${channel.channel}`)
    const config = module.default as Configuration

    if (previous === undefined) delete process.env.OLAYA_CHANNEL
    else process.env.OLAYA_CHANNEL = previous

    expect(config.appId).toBe(channel.appId)
    expect(config.extraMetadata?.desktopName).toBe(`${channel.appId}.desktop`)
    expect(config.linux?.executableName).toBe(channel.appId)
    expect(config.linux?.desktop?.entry?.StartupWMClass).toBe(channel.appId)
    expect(config.deb?.fpm).toContainEqual(expect.stringContaining(`/usr/share/metainfo/${channel.appId}.metainfo.xml`))
    expect(config.rpm?.fpm).toContainEqual(expect.stringContaining(`/usr/share/metainfo/${channel.appId}.metainfo.xml`))
  })
}

test("keeps a hidden prod launcher for old Linux pins", async () => {
  const previous = process.env.OLAYA_CHANNEL
  process.env.OLAYA_CHANNEL = "prod"

  const module = await import("./electron-builder.config.ts?compat=prod")
  const config = module.default as Configuration

  if (previous === undefined) delete process.env.OLAYA_CHANNEL
  else process.env.OLAYA_CHANNEL = previous

  expect(
    config.deb?.fpm?.some((entry) =>
      entry.endsWith("olaya-desktop.desktop=/usr/share/applications/olaya-desktop.desktop"),
    ),
  ).toBe(true)
  expect(
    config.rpm?.fpm?.some((entry) =>
      entry.endsWith("olaya-desktop.desktop=/usr/share/applications/olaya-desktop.desktop"),
    ),
  ).toBe(true)

  const desktop = await Bun.file(legacyDesktopEntry).text()
  expect(desktop).toContain("Exec=/opt/Olaya/io.olaya.desktop %U")
  expect(desktop).toContain("Icon=io.olaya.desktop")
  expect(desktop).toContain("StartupWMClass=io.olaya.desktop")
  expect(desktop).toContain("NoDisplay=true")
})

test("bundles the CLI outside the dev app archive", async () => {
  const previous = process.env.OLAYA_CHANNEL
  process.env.OLAYA_CHANNEL = "dev"
  const module = await import("./electron-builder.config.ts?cli-resource")
  const config = module.default as Configuration
  if (previous === undefined) delete process.env.OLAYA_CHANNEL
  else process.env.OLAYA_CHANNEL = previous

  expect(config.files).toContain("!resources/olaya-cli*")
  expect(config.extraResources).toContainEqual({
    from: "resources/",
    to: "",
    filter: ["olaya-cli*"],
  })
})

for (const channel of ["beta", "prod"] as const) {
  test(`does not bundle the CLI in ${channel} builds`, async () => {
    const previous = process.env.OLAYA_CHANNEL
    process.env.OLAYA_CHANNEL = channel
    const module = await import(`./electron-builder.config.ts?no-cli-resource=${channel}`)
    const config = module.default as Configuration
    if (previous === undefined) delete process.env.OLAYA_CHANNEL
    else process.env.OLAYA_CHANNEL = previous

    expect(config.extraResources).not.toContainEqual({
      from: "resources/",
      to: "",
      filter: ["olaya-cli*"],
    })
  })
}

// The auto-updater installs whatever the publish target serves. It must point at a repository
// this project controls, never at another organisation, where anyone who owned that name could
// ship "updates" to every Olaya desktop install.
for (const channel of ["beta", "prod"]) {
  test(`${channel} updates come from the Olaya repository owner`, async () => {
    const previous = process.env.OLAYA_CHANNEL
    process.env.OLAYA_CHANNEL = channel
    const module = await import(`./electron-builder.config.ts?publish=${channel}`)
    if (previous === undefined) delete process.env.OLAYA_CHANNEL
    else process.env.OLAYA_CHANNEL = previous
    const publish = (module.default as Configuration).publish as { provider: string; owner: string }
    expect(publish.provider).toBe("github")
    expect(publish.owner).toBe("codelikeabhi")
  })
}
