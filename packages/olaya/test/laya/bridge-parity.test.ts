import { expect, test } from "bun:test"
import path from "path"
import { shadowHandler } from "../../src/laya/judgment"
import type { LayaClient } from "../../src/laya/client"

// Offline data must be built from exactly the states the plugin sends: the bridge and the live
// handler share `compact()`, and this pins that nothing between them diverges.
test("the compaction bridge produces the state the plugin sends for the same request", async () => {
  const request = {
    permission: "bash",
    patterns: ["rm -rf *"],
    metadata: { command: "rm -rf ./node_modules && bun install", description: "reinstall deps" },
  }
  const context = { task: "the build is broken, please fix the dependencies", cwd: "/repo" }
  const budget = 471

  const sent: unknown[] = []
  const client = {
    stateBudget: async () => budget,
    decide: async (state: unknown) => {
      sent.push(state)
      return { ok: true, probabilities: { auto_approve: 0.5 }, latencyMs: 1, checkpoint: "c" }
    },
  } as unknown as LayaClient
  await shadowHandler({ client: () => client, context: async () => context })({ id: "per_1", sessionID: "ses_1", ...request })

  const proc = Bun.spawn(["bun", path.join(import.meta.dir, "../../script/laya-compact.ts")], { stdin: "pipe", stdout: "pipe" })
  proc.stdin.write(JSON.stringify({ id: "r1", request, context, budget }) + "\n")
  proc.stdin.end()
  const out = JSON.parse((await new Response(proc.stdout).text()).trim())

  expect(sent).toHaveLength(1)
  expect(out.state).toEqual(sent[0])
})
