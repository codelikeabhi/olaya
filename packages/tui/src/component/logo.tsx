import { RGBA, TextAttributes } from "@opentui/core"
import { createMemo, For, type JSX } from "solid-js"
import { tint, useTheme } from "../context/theme"
import { markCells, POINT, word } from "../logo"

export function Logo() {
  const { theme } = useTheme()

  const renderLine = (line: string, fg: RGBA, bold: boolean): JSX.Element[] => {
    const shadow = tint(theme.background, fg, 0.25)
    const attrs = bold ? TextAttributes.BOLD : undefined
    return Array.from(line).map((char) => {
      if (char === "_") {
        return (
          <text fg={fg} bg={shadow} attributes={attrs} selectable={false}>
            {" "}
          </text>
        )
      }
      if (char === "^") {
        return (
          <text fg={fg} bg={shadow} attributes={attrs} selectable={false}>
            ▀
          </text>
        )
      }
      if (char === "~") {
        return (
          <text fg={shadow} attributes={attrs} selectable={false}>
            ▀
          </text>
        )
      }
      if (char === ",") {
        return (
          <text fg={shadow} attributes={attrs} selectable={false}>
            ▄
          </text>
        )
      }
      return (
        <text fg={fg} attributes={attrs} selectable={false}>
          {char}
        </text>
      )
    })
  }

  const cells = markCells()
  const point = createMemo(() => {
    const bg = theme.background
    const dark = 0.2126 * bg.r + 0.7152 * bg.g + 0.0722 * bg.b < 0.5
    return RGBA.fromHex(dark ? POINT.dark : POINT.light)
  })

  return (
    <box>
      <For each={word}>
        {(line, index) => (
          <box flexDirection="row" gap={2}>
            <box flexDirection="row">
              {cells[index()]!.map((cell) => (
                <text fg={cell.ink === "point" ? point() : theme.text} selectable={false}>
                  {cell.char}
                </text>
              ))}
            </box>
            <box flexDirection="row">{renderLine(line, theme.text, true)}</box>
          </box>
        )}
      </For>
    </box>
  )
}
