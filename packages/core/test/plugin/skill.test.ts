import { describe, expect } from "bun:test"
import { Effect } from "effect"
import { AppNodeBuilder } from "@olaya/core/effect/app-node-builder"
import { SkillPlugin } from "@olaya/core/plugin/skill"
import { SkillV2 } from "@olaya/core/skill"
import { testEffect } from "../lib/effect"
import { host } from "./host"

const it = testEffect(AppNodeBuilder.build(SkillV2.node))

describe("SkillPlugin.Plugin", () => {
  it.effect("registers the built-in customize-olaya skill", () =>
    Effect.gen(function* () {
      const skill = yield* SkillV2.Service
      yield* SkillPlugin.Plugin.effect(host({ skill: { ...skill, reload: skill.reload } }))

      expect(yield* skill.list()).toContainEqual(
        expect.objectContaining({
          name: "customize-olaya",
          description: expect.stringContaining("olaya's own configuration"),
        }),
      )
    }),
  )
})
