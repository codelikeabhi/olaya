import { useFilteredList } from "@olaya/ui/hooks"
import { ProviderIcon } from "@olaya/ui/provider-icon"
import { Switch } from "@olaya/ui/v2/switch-v2"
import { ButtonV2 } from "@olaya/ui/v2/button-v2"
import { SelectV2 } from "@olaya/ui/v2/select-v2"
import { type Component, For, Show, createMemo } from "solid-js"
import { useLanguage } from "@/context/language"
import { useModels } from "@/context/models"
import { useServerSync } from "@/context/server-sync"
import { showToast } from "@/utils/toast"
import { popularProviders } from "@/hooks/use-providers"
import { SettingsListV2 } from "./parts/list"
import { SettingsRowV2 } from "./parts/row"
import "./settings-v2.css"

type ModelItem = ReturnType<ReturnType<typeof useModels>["list"]>[number]

const PROVIDER_ICON_SIZE = 16

/**
 * Which models Olaya's decision layer may move a task between. The selected model in the prompt
 * stays the default; routing never picks a model left out of this list.
 */
export const SettingsRoutingV2: Component = () => {
  const language = useLanguage()
  const models = useModels()
  const serverSync = useServerSync()

  const routing = () => serverSync().data.config.routing ?? {}
  const pool = createMemo(() => new Set(routing().models ?? []))

  const list = useFilteredList<ModelItem>({
    items: (_filter) => models.list(),
    key: (x) => `${x.provider.id}/${x.id}`,
    filterKeys: ["provider.name", "name", "id"],
    sortBy: (a, b) => (a.cost?.input ?? 0) - (b.cost?.input ?? 0) || a.name.localeCompare(b.name),
    groupBy: (x) => x.provider.id,
    sortGroupsBy: (a, b) => {
      const ai = popularProviders.indexOf(a.category)
      const bi = popularProviders.indexOf(b.category)
      if (ai >= 0 && bi < 0) return -1
      if (ai < 0 && bi >= 0) return 1
      if (ai >= 0 && bi >= 0) return ai - bi
      return a.items[0].provider.name.localeCompare(b.items[0].provider.name)
    },
  })

  const save = async (next: { enabled?: boolean; models?: string[] }) => {
    const before = routing()
    const merged = { ...before, ...next }
    serverSync().set("config", "routing", merged)
    await serverSync()
      .updateConfig({ routing: merged })
      .catch((err: unknown) => {
        serverSync().set("config", "routing", before)
        showToast({
          title: language.t("common.requestFailed"),
          description: err instanceof Error ? err.message : String(err),
        })
      })
  }

  const toggle = (id: string, on: boolean) => {
    const current = routing().models ?? []
    void save({ models: on ? [...new Set([...current, id])] : current.filter((x) => x !== id) })
  }

  // Failover: the ordered models a task moves to when its provider fails (config `failover`).
  const failover = () => serverSync().data.config.failover ?? {}
  const chain = () => failover().models ?? []
  const saveFailover = async (next: { models?: string[]; wait_for_reset?: number }) => {
    const before = failover()
    const merged = { ...before, ...next }
    serverSync().set("config", "failover", merged)
    await serverSync()
      .updateConfig({ failover: merged })
      .catch((err: unknown) => {
        serverSync().set("config", "failover", before)
        showToast({
          title: language.t("common.requestFailed"),
          description: err instanceof Error ? err.message : String(err),
        })
      })
  }
  const move = (index: number) => {
    const next = [...chain()]
    ;[next[index - 1], next[index]] = [next[index]!, next[index - 1]!]
    void saveFailover({ models: next })
  }
  const nameOf = (id: string) => {
    const item = models.list().find((x) => `${x.provider.id}/${x.id}` === id)
    return item ? `${item.name} · ${item.provider.name}` : id
  }
  const candidates = createMemo(() => models.list().filter((x) => !chain().includes(`${x.provider.id}/${x.id}`)))
  const waits = [0, 10, 20, 30, 60]

  const price = (item: ModelItem) => {
    const cost = item.cost
    if (!cost || (!cost.input && !cost.output)) return language.t("settings.routing.price.none")
    return language.t("settings.routing.price", { input: `$${cost.input}`, output: `$${cost.output}` })
  }

  return (
    <>
      <div class="settings-v2-tab-header settings-v2-tab-header--stacked">
        <h2 class="settings-v2-tab-title">{language.t("settings.routing.title")}</h2>
      </div>

      <div class="settings-v2-tab-body settings-v2-models">
        <div class="settings-v2-section">
          <SettingsListV2>
            <SettingsRowV2
              title={language.t("settings.routing.enabled.title")}
              description={language.t("settings.routing.enabled.description")}
            >
              <div>
                <Switch checked={routing().enabled === true} onChange={(on) => void save({ enabled: on })} hideLabel>
                  {language.t("settings.routing.enabled.title")}
                </Switch>
              </div>
            </SettingsRowV2>
          </SettingsListV2>
          <p class="settings-v2-models-status">{language.t("settings.routing.selected", { count: pool().size })}</p>
        </div>

        <div class="settings-v2-section" data-component="settings-failover">
          <h3 class="settings-v2-models-group-header">
            <span class="settings-v2-section-title">{language.t("settings.failover.title")}</span>
          </h3>
          <p class="settings-v2-models-status">{language.t("settings.failover.description")}</p>
          <SettingsListV2>
            <For each={chain()}>
              {(id, index) => (
                <SettingsRowV2 title={`${index() + 1}. ${nameOf(id)}`} description={id}>
                  <div class="flex gap-1">
                    <ButtonV2
                      size="normal"
                      variant="ghost-muted"
                      disabled={index() === 0}
                      onClick={() => move(index())}
                    >
                      {language.t("settings.failover.up")}
                    </ButtonV2>
                    <ButtonV2
                      size="normal"
                      variant="ghost-muted"
                      onClick={() => void saveFailover({ models: chain().filter((x) => x !== id) })}
                    >
                      {language.t("settings.failover.remove")}
                    </ButtonV2>
                  </div>
                </SettingsRowV2>
              )}
            </For>
            <SettingsRowV2
              title={language.t("settings.failover.add.title")}
              description={language.t("settings.failover.add.description")}
            >
              <SelectV2
                appearance="inline"
                data-action="settings-failover-add"
                options={candidates()}
                placeholder={language.t("settings.failover.add.placeholder")}
                placement="bottom-end"
                gutter={6}
                value={(item) => `${item.provider.id}/${item.id}`}
                label={(item) => `${item.name} · ${item.provider.name}`}
                onSelect={(item) =>
                  item && void saveFailover({ models: [...chain(), `${item.provider.id}/${item.id}`] })
                }
              />
            </SettingsRowV2>
            <SettingsRowV2
              title={language.t("settings.failover.wait.title")}
              description={language.t("settings.failover.wait.description")}
            >
              <SelectV2
                appearance="inline"
                data-action="settings-failover-wait"
                options={waits}
                current={waits.find((x) => x === (failover().wait_for_reset ?? 0)) ?? 0}
                placement="bottom-end"
                gutter={6}
                value={(x) => String(x)}
                label={(x) =>
                  x === 0
                    ? language.t("settings.failover.wait.never")
                    : language.t("settings.failover.wait.minutes", { count: x })
                }
                onSelect={(x) => x != null && void saveFailover({ wait_for_reset: x })}
              />
            </SettingsRowV2>
          </SettingsListV2>
          <p class="settings-v2-models-status">{language.t("settings.failover.privacy")}</p>
        </div>

        <Show
          when={list.flat().length > 0}
          fallback={<div class="settings-v2-models-status">{language.t("dialog.model.empty")}</div>}
        >
          <For each={list.grouped.latest}>
            {(group) => (
              <div class="settings-v2-section" data-component="settings-routing-provider">
                <h3 class="settings-v2-models-group-header">
                  <span class="settings-v2-models-group-label">
                    <ProviderIcon
                      id={group.category}
                      width={PROVIDER_ICON_SIZE}
                      height={PROVIDER_ICON_SIZE}
                      class="settings-v2-models-provider-icon shrink-0"
                    />
                    <span class="settings-v2-section-title">{group.items[0].provider.name}</span>
                  </span>
                </h3>
                <SettingsListV2>
                  <For each={group.items}>
                    {(item) => {
                      const id = `${item.provider.id}/${item.id}`
                      return (
                        <SettingsRowV2 title={item.name} description={`${item.id} · ${price(item)}`}>
                          <div>
                            <Switch checked={pool().has(id)} onChange={(on) => toggle(id, on)} hideLabel>
                              {item.name}
                            </Switch>
                          </div>
                        </SettingsRowV2>
                      )
                    }}
                  </For>
                </SettingsListV2>
              </div>
            )}
          </For>
        </Show>
      </div>
    </>
  )
}
