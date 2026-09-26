export * as ConfigPaths from "./paths"

import path from "path"
import { Flag } from "@olaya/core/flag/flag"
import { Global } from "@olaya/core/global"
import { unique } from "remeda"
import * as Effect from "effect/Effect"
import { FSUtil } from "@olaya/core/fs-util"

export const files = Effect.fn("ConfigPaths.projectFiles")(function* (
  name: string,
  directory: string,
  worktree?: string,
) {
  const afs = yield* FSUtil.Service
  // A project migrating from OpenCode keeps its files. They go last in the targets so that, (olaya-rename:keep)
  // after the reversal below, they are applied before Olaya's and lose any conflict.
  const legacy = name === "olaya" ? ["opencode.jsonc", "opencode.json"] : [] // olaya-rename:keep
  return (yield* afs.up({
    targets: [`${name}.jsonc`, `${name}.json`, ...legacy],
    start: directory,
    stop: worktree,
  })).toReversed()
})

export const directories = Effect.fn("ConfigPaths.directories")(function* (directory: string, worktree?: string) {
  const afs = yield* FSUtil.Service
  return unique([
    Global.Path.config,
    ...(!Flag.OLAYA_DISABLE_PROJECT_CONFIG
      ? yield* afs.up({
          // Not reversed: within a level the later entry wins, so the legacy dir goes first.
          targets: [".opencode", ".olaya"], // olaya-rename:keep
          start: directory,
          stop: worktree,
        })
      : []),
    ...(yield* afs.up({
      targets: [".opencode", ".olaya"], // olaya-rename:keep
      start: Global.Path.home,
      stop: Global.Path.home,
    })),
    ...(Flag.OLAYA_CONFIG_DIR ? [Flag.OLAYA_CONFIG_DIR] : []),
  ])
})

export function fileInDirectory(dir: string, name: string) {
  return [path.join(dir, `${name}.json`), path.join(dir, `${name}.jsonc`)]
}
