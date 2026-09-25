// olaya-rename:keep-file (tests the OpenCode import, so it names the upstream on purpose)
import { expect, test } from "bun:test"
import fs from "fs/promises"
import os from "os"
import path from "path"
import { adoptLegacyDir } from "../../src/global"

async function tmp() {
  return fs.mkdtemp(path.join(os.tmpdir(), "olaya-legacy-"))
}

test("imports an OpenCode dir once, renaming files that carry the old name", async () => {
  const root = await tmp()
  const legacy = path.join(root, "opencode")
  const current = path.join(root, "olaya")
  await fs.mkdir(legacy)
  await fs.writeFile(path.join(legacy, "opencode.json"), '{"theme":"x"}')
  await fs.writeFile(path.join(legacy, "opencode.db"), "db")
  await fs.writeFile(path.join(legacy, "auth.json"), "{}")

  expect(await adoptLegacyDir(legacy, current)).toBe(true)
  expect(await fs.readFile(path.join(current, "olaya.json"), "utf8")).toBe('{"theme":"x"}')
  expect(await fs.readFile(path.join(current, "olaya.db"), "utf8")).toBe("db")
  expect(await fs.readFile(path.join(current, "auth.json"), "utf8")).toBe("{}")
  // OpenCode's copy is untouched.
  expect((await fs.readdir(legacy)).sort()).toEqual(["auth.json", "opencode.db", "opencode.json"])
  // Second run is a no-op: Olaya's dir now exists.
  expect(await adoptLegacyDir(legacy, current)).toBe(false)
})

test("does nothing without an OpenCode dir or when Olaya's already exists", async () => {
  const root = await tmp()
  expect(await adoptLegacyDir(path.join(root, "missing"), path.join(root, "olaya"))).toBe(false)
  await fs.mkdir(path.join(root, "opencode"))
  await fs.mkdir(path.join(root, "olaya2"))
  expect(await adoptLegacyDir(path.join(root, "opencode"), path.join(root, "olaya2"))).toBe(false)
})

test("imports into the empty skeleton an earlier run left behind (e.g. `olaya --version`)", async () => {
  const root = await tmp()
  const legacy = path.join(root, "opencode")
  const current = path.join(root, "olaya")
  await fs.mkdir(legacy)
  await fs.writeFile(path.join(legacy, "opencode.json"), "{}")
  await fs.mkdir(path.join(current, "log"), { recursive: true })
  await fs.writeFile(path.join(current, "log", "run.log"), "x")
  await fs.mkdir(path.join(current, "repos"), { recursive: true })
  expect(await adoptLegacyDir(legacy, current)).toBe(true)
  expect(await fs.readFile(path.join(current, "olaya.json"), "utf8")).toBe("{}")
})

test("does not import over real content", async () => {
  const root = await tmp()
  await fs.mkdir(path.join(root, "opencode"))
  await fs.writeFile(path.join(root, "opencode", "opencode.json"), "{}")
  await fs.mkdir(path.join(root, "olaya"))
  await fs.writeFile(path.join(root, "olaya", "olaya.json"), '{"mine":true}')
  expect(await adoptLegacyDir(path.join(root, "opencode"), path.join(root, "olaya"))).toBe(false)
  expect(await fs.readFile(path.join(root, "olaya", "olaya.json"), "utf8")).toBe('{"mine":true}')
})
