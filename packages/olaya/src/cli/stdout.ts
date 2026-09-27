/**
 * Olaya: stdout writes that are still queued when the CLI exits.
 *
 * Bun 1.3 writes to a pipe asynchronously, and `process.exit()` drops whatever is still queued:
 * `olaya export <id> | jq` got a session cut at 64 KiB, and a slow reader of `olaya run --format json`
 * could lose the last events. Bun exposes no queue length (`writableLength` stays 0), but each write's
 * own callback fires once that chunk is out, so the writes are counted until their callbacks run.
 */

let queued = 0

/** Wraps `process.stdout.write` to count writes whose callback hasn't run yet. Idempotent. */
export function track(stream: NodeJS.WriteStream = process.stdout) {
  const marked = stream as NodeJS.WriteStream & { olayaTracked?: boolean }
  if (marked.olayaTracked) return
  marked.olayaTracked = true
  const write = stream.write.bind(stream) as (...args: unknown[]) => boolean
  stream.write = ((...args: unknown[]) => {
    const at = args.findIndex((arg) => typeof arg === "function")
    const callback = at >= 0 ? (args.splice(at, 1)[0] as (error?: Error | null) => void) : undefined
    queued++
    return write(...args, (error?: Error | null) => {
      queued--
      callback?.(error)
    })
  }) as typeof stream.write
}

/** Resolves once every counted write is out, or after `ms` (a closed or stalled reader). */
export async function drained(ms = 10_000) {
  const deadline = Date.now() + ms
  while (queued > 0 && Date.now() < deadline) await Bun.sleep(5)
  return queued === 0
}
