// olaya-rename:keep-file (tests the OpenCode compatibility layer, so it names the upstream on purpose)
import { expect, test } from "bun:test"
import { adoptLegacyEnv } from "../../src/flag/flag"

test("OPENCODE_* variables are honoured when the OLAYA_* one is unset", () => {
  const env: Record<string, string | undefined> = { OPENCODE_CONFIG_DIR: "/old", OPENCODE_FAKE_VCS: "git" }
  adoptLegacyEnv(env)
  expect(env["OLAYA_CONFIG_DIR"]).toBe("/old")
  expect(env["OLAYA_FAKE_VCS"]).toBe("git")
})

test("an explicit OLAYA_* value wins over the legacy one", () => {
  const env: Record<string, string | undefined> = { OPENCODE_CONFIG_DIR: "/old", OLAYA_CONFIG_DIR: "/new" }
  adoptLegacyEnv(env)
  expect(env["OLAYA_CONFIG_DIR"]).toBe("/new")
})

test("the Zen provider key is not ours to rename", () => {
  const env: Record<string, string | undefined> = { OPENCODE_API_KEY: "sk" }
  adoptLegacyEnv(env)
  expect(env["OLAYA_API_KEY"]).toBeUndefined()
})
