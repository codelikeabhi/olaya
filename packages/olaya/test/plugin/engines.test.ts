import { expect, test } from "bun:test"
import { checkPluginCompatibility } from "../../src/plugin/shared"

const pkg = (engines: Record<string, string>) => ({ dir: "/x", pkg: "/x/package.json", json: { engines } })
// olaya-rename:keep-file: `engines.opencode` is the upstream plugin contract under test.

test("an OpenCode engines range is checked against the upstream plugin API, not Olaya's version", async () => {
  await expect(checkPluginCompatibility("/x", "0.1.0", pkg({ opencode: ">=1.0.0" }))).resolves.toBeUndefined()
  await expect(checkPluginCompatibility("/x", "0.1.0", pkg({ opencode: ">=2.0.0" }))).rejects.toThrow("OpenCode >=2.0.0")
})

test("an Olaya engines range is checked against Olaya's version once it is 1.x", async () => {
  await expect(checkPluginCompatibility("/x", "0.1.0", pkg({ olaya: ">=5.0.0" }))).resolves.toBeUndefined()
  await expect(checkPluginCompatibility("/x", "1.2.0", pkg({ olaya: ">=5.0.0" }))).rejects.toThrow("olaya >=5.0.0")
  await expect(checkPluginCompatibility("/x", "1.2.0", pkg({ olaya: "^1.0.0" }))).resolves.toBeUndefined()
})
