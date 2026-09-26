import { expect, test } from "bun:test"
import { Installation } from "../../src/installation"

test("only the install-script channel is published by default", () => {
  delete process.env.OLAYA_PUBLISHED_CHANNELS
  expect(Installation.isPublished("curl")).toBe(true)
  for (const m of ["npm", "bun", "pnpm", "yarn", "brew", "scoop", "choco"] as const) expect(Installation.isPublished(m)).toBe(false)
})

test("a packager can publish a channel explicitly", () => {
  process.env.OLAYA_PUBLISHED_CHANNELS = "brew"
  expect(Installation.isPublished("brew")).toBe(true)
  expect(Installation.isPublished("npm")).toBe(false)
  delete process.env.OLAYA_PUBLISHED_CHANNELS
})

test("upgrades come from Olaya's own install script, never upstream's", () => {
  expect(Installation.INSTALL_SCRIPT).toBe("https://raw.githubusercontent.com/codelikeabhi/olaya/main/install")
})
