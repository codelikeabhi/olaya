// The Olaya logo for terminals: the Signal mark (the decision curve and its operating point),
// then the block wordmark. Two pixel rows make one text line, drawn with ▀ ▄ █.

/** Pixel rows of the mark: "i" ink, "a" the coral operating point, "." empty. */
export const mark = [
  "......aa.iii",
  ".....aaaaiii",
  ".....aaaa...",
  "....iaaa....",
  "...iii......",
  "..iii.......",
  "iiii........",
  "iii.........",
]

/** The wordmark. "_" is a shaded space, "^" a shaded upper half, "~" and "," shadow-only halves. */
export const word = ["     ▄                  ", "█▀▀█ █    ▀▀▀█ █  █ ▀▀▀█", "█__█ █___ █▀▀█ ▀▀▀█ █▀▀█", "▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▄▄▄▀ ▀▀▀▀"]

export const logo = { mark, word }

/** The point's colour: Coral 300 reads on dark grounds, Coral 600 on light ones. */
export const POINT = { dark: "#F08A70", light: "#D9573B" }

export type Cell = { char: string; ink?: "ink" | "point" }

/** The mark as text cells, one array per line. No cell mixes the two inks, so no cell needs a background. */
export function markCells(): Cell[][] {
  const lines: Cell[][] = []
  for (let row = 0; row < mark.length; row += 2) {
    const top = mark[row]!
    const bottom = mark[row + 1] ?? ""
    lines.push(
      Array.from(top, (t, col) => {
        const b = bottom[col] ?? "."
        if (t === "." && b === ".") return { char: " " }
        const ink = (t !== "." ? t : b) === "a" ? "point" : "ink"
        return { char: t !== "." && b !== "." ? "█" : t !== "." ? "▀" : "▄", ink }
      }),
    )
  }
  return lines
}

const plainWord = (line: string) => line.replaceAll("_", " ").replaceAll("^", "▀").replaceAll("~", " ").replaceAll(",", " ")

/** Uncoloured lines, for output that is not a terminal. */
export function plainLogo(): string[] {
  return markCells().map((cells, i) => (cells.map((c) => c.char).join("") + "  " + plainWord(word[i] ?? "")).trimEnd())
}

/** ANSI-coloured lines. The ink is the terminal's own foreground, so it suits light and dark themes. */
export function ansiLogo(pad = ""): string[] {
  const reset = "\x1b[0m"
  const point = "\x1b[38;2;224;105;77m" // between Coral 600 and 300: legible on light and dark terminals
  const shadow = "\x1b[38;5;238m"
  const shade = "\x1b[48;5;238m"
  const drawWord = (line: string) =>
    Array.from(line, (char) => {
      if (char === "_") return `${shade} ${reset}`
      if (char === "^") return `${shade}▀${reset}`
      if (char === "~") return `${shadow}▀${reset}`
      if (char === ",") return `${shadow}▄${reset}`
      return char
    }).join("")
  return markCells().map((cells, i) => {
    const drawn = cells.map((c) => (c.ink === "point" ? `${point}${c.char}${reset}` : c.char)).join("")
    return `${pad}${drawn}  ${drawWord(word[i] ?? "")}`
  })
}

export const go = {
  left: ["    ", "█▀▀▀", "█_^█", "▀▀▀▀"],
  right: ["    ", "█▀▀█", "█__█", "▀▀▀▀"],
}

export const marks = "_^~,"
