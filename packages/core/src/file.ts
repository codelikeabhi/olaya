export * as File from "./file"

import { Revert } from "@olaya/schema/revert"

export const Diff = Revert.FileDiff
export type Diff = typeof Diff.Type
