import { registerCustomTheme } from "@pierre/diffs"
import { OlayaTheme } from "./marked-theme"

let registered = false

export function registerOlayaTheme() {
  if (registered) return
  registered = true
  registerCustomTheme("Olaya", () => Promise.resolve(OlayaTheme))
}
