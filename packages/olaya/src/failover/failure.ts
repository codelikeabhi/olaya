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
  if (error.name === "APIError") {
    const status = typeof data["statusCode"] === "number" ? data["statusCode"] : undefined
    return {
      providerID,
      message,
      body,
      status,
      headers: (data["responseHeaders"] as Record<string, string> | undefined) ?? undefined,
      kind: /timed? ?out|header timeout/i.test(message)
        ? "stall"
        : // a provider that answered is not a network failure, whatever its message says
          status === undefined && NETWORK.test(firstLine(message))
          ? "network"
          : undefined,
    }
  }
  // Other errors can carry model output (a validation error quotes the streamed chunk), so only
  // their first line is read, and only for specific phrases.
  const head = firstLine(message)
  return {
    providerID,
    message,
    kind: /\bstall(?:ed)?\b/i.test(head)
      ? "stall"
      : /timed out/i.test(head)
        ? "timeout"
        : NETWORK.test(head)
          ? "network"
          : undefined,
  }
}

const firstLine = (text: string) => text.split("\n")[0] ?? ""

// Bun says "Unable to connect" (ConnectionRefused) and the AI SDK "Cannot connect to API": a local
// model server restarting looks like this, and it clears with time. A stream cut mid-answer is
// "The socket connection was closed unexpectedly" (ECONNRESET).
const NETWORK =
  /ECONNRESET|ECONNREFUSED|ConnectionRefused|ConnectionClosed|ENOTFOUND|EAI_AGAIN|ETIMEDOUT|socket connection was closed|socket hang up|fetch failed|(?:^|: )terminated$|network (?:error|connection)|unable to connect|cannot connect|connection (?:refused|reset|closed|lost)/i

export * as FailoverFailure from "./failure"
