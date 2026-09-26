export * from "./client.js"
export * from "./server.js"

import { createOlayaClient } from "./client.js"
import { createOlayaServer } from "./server.js"
import type { ServerOptions } from "./server.js"

export async function createOlaya(options?: ServerOptions) {
  const server = await createOlayaServer({
    ...options,
  })

  const client = createOlayaClient({
    baseUrl: server.url,
  })

  return {
    client,
    server,
  }
}
