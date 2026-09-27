import type { SessionRetry } from "@/session/retry"
import type { Failure } from "./classify"

/** A failed step's error, as the harness records it, in the classifier's terms. */
export function failureOf(error: SessionRetry.Err, providerID: string): Failure {
  const data: Record<string, unknown> =
    typeof error.data === "object" && error.data !== null ? (error.data as Record<string, unknown>) : {}
  const message = typeof data["message"] === "string" ? data["message"] : error.name
  const body = typeof data["responseBody"] === "string" ? data["responseBody"] : undefined
  if (error.name === "ContextOverflowError") return { providerID, message, body, kind: "context" }
  if (error.name === "ProviderAuthError") return { providerID, message, kind: "auth" }
  if (error.name === "MessageAbortedError") return { providerID, message, kind: "aborted" }
  if (error.name === "ContentFilterError") return { providerID, message, kind: "refusal" }
  if (error.name === "APIError")
    return {
      providerID,
      message,
      body,
      status: typeof data["statusCode"] === "number" ? data["statusCode"] : undefined,
      headers: (data["responseHeaders"] as Record<string, string> | undefined) ?? undefined,
      kind: /timed? ?out|header timeout/i.test(message) ? "stall" : NETWORK.test(message) ? "network" : undefined,
    }
  return {
    providerID,
    message,
    kind: /stall/i.test(message)
      ? "stall"
      : /timed out/i.test(message)
        ? "timeout"
        : NETWORK.test(message)
          ? "network"
          : undefined,
  }
}

// Bun says "Unable to connect" (ConnectionRefused) and the AI SDK "Cannot connect to API": a local
// model server restarting looks like this, and it clears with time.
const NETWORK =
  /ECONNRESET|ECONNREFUSED|ConnectionRefused|ConnectionClosed|ENOTFOUND|EAI_AGAIN|ETIMEDOUT|socket|fetch failed|terminated|network|unable to connect|cannot connect|connection (?:refused|reset|closed|lost)/i

export * as FailoverFailure from "./failure"
