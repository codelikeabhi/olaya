// @ts-nocheck

import { Olaya } from "@olaya/core"
import { ReadTool } from "@olaya/core/tools"

const olaya = Olaya.make({})

olaya.tool.add(ReadTool)

olaya.tool.add({
  name: "bash",
  schema: {
    type: "object",
    properties: {
      command: {
        type: "string",
        description: "The command to run.",
      },
    },
    required: ["command"],
  },
  execute(input, ctx) {},
})

olaya.auth.add({
  provider: "openai",
  type: "api",
  value: process.env.OPENAI_API_KEY,
})

olaya.agent.add({
  name: "build",
  permissions: [],
  model: {
    id: "gpt-5-5",
    provider: "openai",
    variant: "xhigh",
  },
})

const sessionID = await olaya.session.create({
  agent: "build",
})

olaya.subscribe((event) => {
  console.log(event)
})

await olaya.session.prompt({
  sessionID,
  text: "hey what is up",
})

await olaya.session.prompt({
  sessionID,
  text: "what is up with this",
  files: [
    {
      mime: "image/png",
      uri: "data:image/png;base64,xxxx",
    },
  ],
})

await olaya.session.wait()

console.log(await olaya.session.messages(sessionID))
