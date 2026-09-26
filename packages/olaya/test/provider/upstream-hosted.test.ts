import { expect, test } from "bun:test"
import { Provider } from "../../src/provider/provider"

// olaya-rename:keep-file: "opencode" and "opencode-go" are upstream's provider ids under test.

test("upstream's hosted providers are hidden by default", () => {
  const hidden = Provider.disabledProviders({})
  expect([...hidden].sort()).toEqual(["opencode", "opencode-go"])
})

test("enabling, configuring or logging in to one opts in to that one only", () => {
  expect(Provider.disabledProviders({ enabled_providers: ["opencode"] }).has("opencode")).toBe(false)
  expect(Provider.disabledProviders({ provider: { "opencode-go": {} } }).has("opencode-go")).toBe(false)
  const loggedIn = Provider.disabledProviders({}, { opencode: { type: "api", key: "k" } })
  expect(loggedIn.has("opencode")).toBe(false)
  expect(loggedIn.has("opencode-go")).toBe(true)
})

test("the user's own disabled list still applies", () => {
  const hidden = Provider.disabledProviders({ disabled_providers: ["openai"], enabled_providers: ["opencode"] })
  expect(hidden.has("openai")).toBe(true)
  expect(hidden.has("opencode")).toBe(false)
})
