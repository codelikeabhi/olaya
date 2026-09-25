/// <reference path="../markdown.d.ts" />

export * as SkillPlugin from "./skill"

import { define } from "./internal"
import { Effect } from "effect"
import { AbsolutePath } from "../schema"
import { SkillV2 } from "../skill"
import customizeOlayaContent from "./skill/customize-olaya.md" with { type: "text" }

export const CustomizeOlayaContent = customizeOlayaContent

export const Plugin = define({
  id: "skill",
  effect: Effect.fn(function* (ctx) {
    yield* ctx.skill.transform((draft) => {
      draft.source(
        SkillV2.EmbeddedSource.make({
          type: "embedded",
          skill: SkillV2.Info.make({
            name: "customize-olaya",
            description:
              "Use ONLY when the user is editing or creating olaya's own configuration: olaya.json, olaya.jsonc, files under .olaya/, or files under ~/.config/olaya/. Also use when creating or fixing olaya agents, subagents, commands, skills, plugins, MCP servers, or permission rules. Do not use for the user's own application code, or for any project that is not configuring olaya itself.",
            location: AbsolutePath.make("/builtin/customize-olaya.md"),
            content: CustomizeOlayaContent,
          }),
        }),
      )
    })
  }),
})
