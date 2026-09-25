import { describe, expect, test } from "bun:test"
import { rename } from "./olaya-rename"

const r = (s: string, file = "") => rename(s, file).out

describe("olaya-rename", () => {
  test("renames the product in every case form", () => {
    expect(r('import { x } from "@opencode-ai/plugin"')).toBe('import { x } from "@olaya/plugin"')
    expect(r("OpenCode Opencode opencode OPENCODE_CONFIG")).toBe("Olaya Olaya olaya OLAYA_CONFIG")
    expect(r("x-opencode-directory")).toBe("x-olaya-directory")
    expect(r("npm i -g opencode-ai")).toBe("npm i -g olaya")
    expect(r("https://github.com/anomalyco/opencode/releases")).toBe("https://github.com/codelikeabhi/olaya/releases")
  })

  test("never touches protected upstream names", () => {
    for (const s of [
      "https://opencode.ai/docs/config",
      "https://models.opencode.ai/api.json",
      "see opencode.ai for more",
      "${url}/.well-known/opencode",
      "opencode-go",
      "OpenCode Zen",
      "provider.connect.opencodeZen.visit",
      "OPENCODE_API_KEY",
      "x-opencode-session",
      "@gitlab/opencode-gitlab-auth",
      "anomalyco/tree-sitter-vue",
      "sst-dev.opencode",
      "__OPENCODE_PHOTON_WASM_PATH",
      "https://github.com/anomalyco/opencode/issues/29997",
    ])
      expect(r(s)).toBe(s)
  })

  test("the provider id literal is protected only in provider files", () => {
    const code = 'if (providerID === "opencode") return'
    expect(r(code, "packages/opencode/src/provider/provider.ts")).toBe(code)
    expect(r(code, "packages/opencode/src/other.ts")).toBe('if (providerID === "olaya") return')
  })

  test("is idempotent", () => {
    const s = 'OpenCode at https://opencode.ai uses OPENCODE_API_KEY and "@opencode-ai/sdk"'
    expect(r(r(s))).toBe(r(s))
    expect(rename(r(s)).hits.every((h) => h === 0)).toBe(true)
  })
})
