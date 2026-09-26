// The summary request Olaya sends at compaction, for a serialized conversation read from stdin:
// the compaction agent's prompt as system, and core's buildPrompt as the user message.
import { buildPrompt } from "../../../packages/core/src/session/compaction"

const system = await Bun.file(new URL("../../../packages/olaya/src/agent/prompt/compaction.txt", import.meta.url)).text()
console.log(JSON.stringify({ system, user: buildPrompt({ context: [await Bun.stdin.text()] }) }))
