import { getComponentCatalogue } from "@opentui/solid/components"
import { registerSpinner } from "opentui-spinner/solid"

export function registerOlayaSpinner() {
  if (!getComponentCatalogue().spinner) registerSpinner()
}
