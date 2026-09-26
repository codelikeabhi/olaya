import { Context } from "effect"
import type { InstanceContext } from "@/project/instance-context"
import type { WorkspaceV2 } from "@olaya/core/workspace"

export const InstanceRef = Context.Reference<InstanceContext | undefined>("~olaya/InstanceRef", {
  defaultValue: () => undefined,
})

export const WorkspaceRef = Context.Reference<WorkspaceV2.ID | undefined>("~olaya/WorkspaceRef", {
  defaultValue: () => undefined,
})
