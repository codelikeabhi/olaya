import { describe, expect, test } from "bun:test"
import { rename } from "./olaya-rename"

const r = (s: string, file = "") => rename(s, file).out

describe("olaya-rename", () => {
  test("renames the product in every case form", () => {
    expect(r('import { x } from "@olaya/plugin"')).toBe('import { x } from "@olaya/plugin"')
    expect(r("Olaya Olaya olaya OLAYA_CONFIG")).toBe("Olaya Olaya olaya OLAYA_CONFIG")
    expect(r("x-olaya-directory")).toBe("x-olaya-directory")
    expect(r("npm i -g olaya")).toBe("npm i -g olaya")
    expect(r("https://github.com/codelikeabhi/olaya/releases")).toBe("https://github.com/codelikeabhi/olaya/releases")
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
      "const origin = /^https:\\/\\/([a-z0-9-]+\\.)*opencode\\.ai$/",
      'new RegExp("^https://([a-z0-9-]+\\\\.)*opencode\\\\.ai$")',
    ])
      expect(r(s)).toBe(s)
  })

  test("the provider id literal is protected only in provider files", () => {
    const code = 'if (providerID === "olaya") return'
    expect(r(code, "packages/olaya/src/provider/provider.ts")).toBe(code)
    expect(r(code, "packages/olaya/src/other.ts")).toBe('if (providerID === "olaya") return')
  })

  test("the provider identifier survives in every syntactic form", () => {
    for (const s of [
      "catalog.provider.get(ProviderV2.ID.opencode)",
      "providerOptions?.opencode?.itemId",
      'language.t("dialog.provider.opencode.note")',
      'provider.request.headers["X-BILLING-INVOKE-ORIGIN"] ??= "OpenCode"',
      'const clientID = "opencode-cli"',
      "label: \"OpenCode Console account\"",
    ])
      expect(r(s)).toBe(s)
    // Object keys are the provider id only in provider files, and only in object literals.
    const table = "const priority = {\n  olaya: 0,\n  anthropic: 1,\n}\njobs:\n  olaya:\n"
    expect(r(table, "packages/olaya/src/cli/cmd/github.handler.ts")).toBe(
      "const priority = {\n  olaya: 0,\n  anthropic: 1,\n}\njobs:\n  olaya:\n",
    )
    expect(r("({ home, llm, olaya }) => olaya.run()", "packages/olaya/test/cli/acp/x.test.ts")).toBe(
      "({ home, llm, olaya }) => olaya.run()",
    )
    expect(r("fixture.olaya.run()")).toBe("fixture.olaya.run()")
    // ...while paths and plain words still rename.
    expect(r('path.join(dir, ".olaya", "agent")')).toBe('path.join(dir, ".olaya", "agent")')
    expect(r("resources/olaya-cli")).toBe("resources/olaya-cli")
  })

  test("is idempotent", () => {
    const s = 'Olaya at https://opencode.ai uses OPENCODE_API_KEY and "@olaya/sdk"'
    expect(r(r(s))).toBe(r(s))
    expect(rename(r(s)).hits.every((h) => h === 0)).toBe(true)
  })
})
