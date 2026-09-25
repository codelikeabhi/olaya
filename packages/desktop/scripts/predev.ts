import { $ } from "bun"
import { downloadCliToResources } from "./utils"

await $`bun run install-electron`

await $`bun ./scripts/copy-icons.ts ${process.env.OLAYA_CHANNEL ?? "dev"}`

await $`cd ../olaya && bun script/build-node.ts`
await downloadCliToResources()
