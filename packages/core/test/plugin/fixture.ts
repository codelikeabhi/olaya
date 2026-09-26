import { AgentV2 } from "@olaya/core/agent"
import { AISDK } from "@olaya/core/aisdk"
import { Catalog } from "@olaya/core/catalog"
import { CommandV2 } from "@olaya/core/command"
import { Credential } from "@olaya/core/credential"
import { AppNodeBuilder } from "@olaya/core/effect/app-node-builder"
import { LayerNodePlatform } from "@olaya/core/effect/app-node-platform"
import { LayerNode } from "@olaya/core/effect/layer-node"
import { EventV2 } from "@olaya/core/event"
import { FileSystem } from "@olaya/core/filesystem"
import { FSUtil } from "@olaya/core/fs-util"
import { Integration } from "@olaya/core/integration"
import { Location } from "@olaya/core/location"
import { Npm } from "@olaya/core/npm"
import { PluginV2 } from "@olaya/core/plugin"
import { Reference } from "@olaya/core/reference"
import { SkillV2 } from "@olaya/core/skill"
import { Effect, Layer } from "effect"
import { tempLocationLayer } from "../fixture/location"

const npmLayer = Layer.succeed(
  Npm.Service,
  Npm.Service.of({
    add: () => Effect.succeed({ directory: "", entrypoint: undefined }),
    install: () => Effect.void,
    which: () => Effect.succeed(undefined),
  }),
)

export const PluginTestLayer = AppNodeBuilder.build(
  LayerNode.group([
    FileSystem.node,
    FSUtil.node,
    Location.node,
    Npm.node,
    Credential.node,
    EventV2.node,
    LayerNodePlatform.httpClient,
    PluginV2.node,
    AgentV2.node,
    AISDK.node,
    Catalog.node,
    CommandV2.node,
    Integration.node,
    Reference.node,
    SkillV2.node,
  ]),
  [
    [Location.node, tempLocationLayer],
    [Npm.node, npmLayer],
  ],
)
