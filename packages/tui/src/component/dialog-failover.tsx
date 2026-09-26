import { createMemo } from "solid-js"
import { TextAttributes } from "@opentui/core"
import { useSync } from "../context/sync"
import { useSDK } from "../context/sdk"
import { useTheme } from "../context/theme"
import { DialogSelect, type DialogSelectOption } from "../ui/dialog-select"

const WAIT = "__failover_wait__"
const WAITS = [0, 10, 20, 30, 60]

function Position(props: { index: number }) {
  const { theme } = useTheme()
  if (props.index >= 0)
    return <span style={{ fg: theme.success, attributes: TextAttributes.BOLD }}>#{props.index + 1} in order</span>
  return <span style={{ fg: theme.textMuted }}>○ Not in list</span>
}

/**
 * The models a task moves to, in order, when its provider fails (config `failover`). Enter adds a
 * model to the end of the list or removes it. Saved to the global config, the same place the
 * app's Settings > Routing writes.
 */
export function DialogFailover() {
  const sync = useSync()
  const sdk = useSDK()

  const failover = () => sync.data.config.failover ?? {}

  const save = (next: { models?: string[]; wait_for_reset?: number }) => {
    const before = failover()
    const merged = { ...before, ...next }
    sync.set("config", "failover", merged)
    void sdk.client.global.config
      .update({ config: { failover: merged } })
      .catch(() => sync.set("config", "failover", before))
  }

  const options = createMemo<DialogSelectOption<string>[]>(() => {
    const chain = failover().models ?? []
    const wait = failover().wait_for_reset ?? 0
    const models = sync.data.provider.flatMap((provider) =>
      Object.entries(provider.models)
        .filter(([, info]) => info.status !== "deprecated")
        .map(([id, info]) => {
          const key = `${provider.id}/${id}`
          return {
            value: key,
            title: info.name ?? id,
            description: key,
            category: provider.name,
            footer: <Position index={chain.indexOf(key)} />,
          }
        }),
    )
    return [
      {
        value: WAIT,
        title: "Wait for your model",
        description: wait
          ? `up to ${wait} minutes if it will be back by then`
          : "don't wait: move down the list at once",
        category: "When a provider fails",
        footer: <span>{wait ? `${wait} min` : "off"}</span>,
      },
      ...models,
    ]
  })

  return (
    <DialogSelect
      title="Failover order"
      placeholder="Search models"
      options={options()}
      onSelect={(option) => {
        if (option.value === WAIT) {
          const current = WAITS.indexOf(failover().wait_for_reset ?? 0)
          return save({ wait_for_reset: WAITS[(current + 1) % WAITS.length] })
        }
        const chain = failover().models ?? []
        save({
          models: chain.includes(option.value) ? chain.filter((x) => x !== option.value) : [...chain, option.value],
        })
      }}
    />
  )
}
