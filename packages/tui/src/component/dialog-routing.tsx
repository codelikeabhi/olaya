import { createMemo } from "solid-js"
import { TextAttributes } from "@opentui/core"
import { useSync } from "../context/sync"
import { useSDK } from "../context/sdk"
import { useTheme } from "../context/theme"
import { DialogSelect, type DialogSelectOption } from "../ui/dialog-select"

const ENABLED = "__routing_enabled__"

function Mark(props: { on: boolean; on_label: string; off_label: string }) {
  const { theme } = useTheme()
  if (props.on) return <span style={{ fg: theme.success, attributes: TextAttributes.BOLD }}>✓ {props.on_label}</span>
  return <span style={{ fg: theme.textMuted }}>○ {props.off_label}</span>
}

/**
 * Which models Olaya's decision layer may move a task between. Enter toggles an entry and keeps
 * the dialog open. The choice is saved to the global config, the same place the desktop app's
 * Settings > Routing writes, so both show the same pool.
 */
export function DialogRouting() {
  const sync = useSync()
  const sdk = useSDK()

  const routing = () => sync.data.config.routing ?? {}

  const save = (next: { enabled?: boolean; models?: string[] }) => {
    const before = routing()
    const merged = { ...before, ...next }
    sync.set("config", "routing", merged)
    void sdk.client.global.config.update({ config: { routing: merged } }).catch(() => sync.set("config", "routing", before))
  }

  const options = createMemo<DialogSelectOption<string>[]>(() => {
    const pool = new Set(routing().models ?? [])
    const enabled = routing().enabled === true
    const models = sync.data.provider.flatMap((provider) =>
      Object.entries(provider.models)
        .filter(([, info]) => info.status !== "deprecated")
        .map(([id, info]) => {
          const key = `${provider.id}/${id}`
          const cost = info.cost
          return {
            value: key,
            title: info.name ?? id,
            description: cost && (cost.input || cost.output) ? `$${cost.input} in, $${cost.output} out per 1M` : undefined,
            category: provider.name,
            footer: <Mark on={pool.has(key)} on_label="Included" off_label="Not included" />,
          }
        }),
    )
    return [
      {
        value: ENABLED,
        title: "Let Olaya choose the model",
        description: "moves a task between the models included below; your prompt's model stays the default",
        category: "Routing",
        footer: <Mark on={enabled} on_label="On" off_label="Off" />,
      },
      ...models,
    ]
  })

  return (
    <DialogSelect
      title="Model routing"
      placeholder="Search models"
      options={options()}
      onSelect={(option) => {
        if (option.value === ENABLED) return save({ enabled: routing().enabled !== true })
        const current = routing().models ?? []
        save({
          models: current.includes(option.value)
            ? current.filter((x) => x !== option.value)
            : [...current, option.value],
        })
      }}
    />
  )
}
