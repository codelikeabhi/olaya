import { $ } from "bun"
import { buildCliToResources } from "./utils"

await $`bun run install-electron`

await $`bun ./scripts/copy-icons.ts ${process.env.OLAYA_CHANNEL ?? "dev"}`

await $`cd ../olaya && bun script/build-node.ts`
await buildCliToResources()
