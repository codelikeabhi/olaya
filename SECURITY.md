# Security

## IMPORTANT

We do not accept AI generated security reports. We receive a large number of
these and we absolutely do not have the resources to review them all. If you
submit one that will be an automatic ban from the project.

## Threat Model

### Overview

Olaya is an AI-powered coding assistant that runs locally on your machine. It provides an agent system with access to powerful tools including shell execution, file operations, and web access.

### No Sandbox

Olaya does **not** sandbox the agent. The permission system exists as a UX feature to help users stay aware of what actions the agent is taking - it prompts for confirmation before executing commands, writing files, etc. However, it is not designed to provide security isolation.

If you need true isolation, run Olaya inside a Docker container or VM.

### The decision layer

Olaya's decision layer (`laya/`, `packages/olaya/src/laya/`) can, in **live mode only**, turn a
permission prompt into an automatic approval. That makes it part of Olaya's security surface:

- **In scope:** live mode granting anything on the hard denylist; granting with a checkpoint
  whose manifest has no passed gate, or below the certified threshold; granting after a failed,
  timed-out or malformed judgment; any path by which content from a tool output, file or
  repository steers the judge into approving (prompt injection against the decision layer);
  the sidecar being reachable from anything but loopback; shadow logs leaving the machine or
  retaining secrets the redaction pass should have removed.
- **Also worth a private report, even though it is not a vulnerability in the strict sense:**
  a certified checkpoint approving an action it should have asked about. A certified gate
  bounds the *rate* of such false approvals, so individual cases are expected, but each one is
  evidence for recalibration. Please report the action and the task, never real secrets.

Shadow mode, the default, observes only and cannot change a permission.

### Server Mode

Server mode is opt-in only. When enabled, set `OLAYA_SERVER_PASSWORD` to require HTTP Basic Auth. Without this, the server runs unauthenticated (with a warning). It is the end user's responsibility to secure the server - any functionality it provides is not a vulnerability.

### Out of Scope

| Category                        | Rationale                                                               |
| ------------------------------- | ----------------------------------------------------------------------- |
| **Server access when opted-in** | If you enable server mode, API access is expected behavior              |
| **Sandbox escapes**             | The permission system is not a sandbox (see above)                      |
| **LLM provider data handling**  | Data sent to your configured LLM provider is governed by their policies |
| **MCP server behavior**         | External MCP servers you configure are outside our trust boundary       |
| **Malicious config files**      | Users control their own config; modifying it is not an attack vector    |

---

# Reporting Security Issues

We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

To report a security issue, please use the GitHub Security Advisory ["Report a Vulnerability"](https://github.com/codelikeabhi/olaya/security/advisories/new) tab.

The team will send a response indicating the next steps in handling your report. After the initial reply to your report, the security team will keep you informed of the progress towards a fix and full announcement, and may ask for additional information or guidance.

## Escalation

If you do not receive an acknowledgement of your report within 6 business days, open a
minimal public issue that asks the maintainer (@codelikeabhi) to check the security advisories,
without including any details of the vulnerability.
