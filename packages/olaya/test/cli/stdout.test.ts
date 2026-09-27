import { describe, expect, test } from "bun:test"
import { drained, track } from "../../src/cli/stdout"

// A stream that finishes each write later, as Bun does for a pipe.
function fakeStream(delay: number | null) {
  const seen: unknown[][] = []
  const stream = {
    write(...args: unknown[]) {
      seen.push(args)
      const callback = args.find((arg) => typeof arg === "function") as ((error?: Error | null) => void) | undefined
      if (delay !== null) setTimeout(() => callback?.(null), delay)
      return true
    },
  }
  return { stream: stream as unknown as NodeJS.WriteStream, seen }
}

describe("stdout tracking", () => {
  test("keeps the chunk, encoding and caller's callback, and waits for queued writes", async () => {
    const { stream, seen } = fakeStream(30)
    track(stream)
    track(stream) // idempotent: wrapping twice would count each write twice
    let called = 0
    expect(stream.write("a", "utf8", () => called++)).toBe(true)
    stream.write("b")
    expect(seen.map((args) => args.slice(0, -1))).toEqual([["a", "utf8"], ["b"]])
    expect(await drained(1_000)).toBe(true)
    expect(called).toBe(1)
  })

  test("gives up after the limit when a write never finishes", async () => {
    const { stream } = fakeStream(null)
    track(stream)
    stream.write("stuck")
    const start = Date.now()
    expect(await drained(50)).toBe(false)
    expect(Date.now() - start).toBeLessThan(1_000)
  })
})
