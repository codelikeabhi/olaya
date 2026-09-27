<!-- olaya-rename:keep-file (this file names OpenCode and Laya on purpose: attribution) -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="packages/identity/olaya-logo-dark.svg">
    <img alt="Olaya" src="packages/identity/olaya-logo.svg" width="300">
  </picture>
</p>

<p align="center"><strong>An open-source AI coding harness you can leave alone.</strong></p>

Olaya is a terminal AI coding agent (CLI, TUI, desktop app, server and SDK) built on
[OpenCode](https://github.com/sst/opencode), with a decision layer built on
[Laya](https://github.com/NandhaKishorM/laya): a small, local, calibrated model that decides
what the agent may do without asking you, and when a task is really done.

> **Status: early development. Nothing is released yet.** There are no published binaries or
> packages, and `olaya upgrade` deliberately refuses channels that do not exist yet. Build from
> source (below). The decision layer runs in **Observe** mode by default: it watches and logs, and
> never changes what the agent is allowed to do. **Auto-approve** exists, but it acts only on a
> certified checkpoint, and none has been certified yet.

## Why

Coding agents fail users in two opposite ways. Ask for approval on everything and people stop
reading the prompts. Approve everything and the agent eventually runs something it shouldn't.
Today's answers are static allow-rules, or a frontier-LLM call per tool call (slow, costly,
sends your transcript off the machine).

Olaya puts a ~400M-parameter encoder on that decision. It runs locally in tens of
milliseconds, judges each action against the task you asked for, and will only act on its own
at thresholds that come with a measured bound on how often it would wrongly approve.

## What has been measured so far

Development numbers on OlayaBench Track A: 69 hand-written, task-conditioned items. It is a
seed set: far too small to certify anything, and labelled by its author. They show direction
only.

| decision-maker | wrongly approved (should have asked) | auto-approved (safe) | latency |
|---|---|---|---|
| OpenCode's default rules (`"*": "allow"`) | 92.3% | 100% | — |
| Laya, no fine-tuning | 25.6% | 83.3% | ~30 ms (M-series GPU) |
| Laya, first coding fine-tune (249 synthetic rows, head only) | 20.5% | 83.3% | ~30 ms (M-series GPU) |
| An 8B local LLM as judge (qwen3:8b) | 20.5% | 93.3% | 6.7 s |

The target is a checkpoint that auto-approves a useful share of actions with a *certified*
false-approve rate of at most 1%. Reaching it needs a human-labelled evaluation set with at
least 300 should-ask items: below that, no 1% bound can be proven at 95% confidence, whatever
the model.

## Long unattended runs

A long task shouldn't stop because one provider ran out of quota or went down. Give Olaya a
fallback chain, and it moves the task to the next model, **with its whole working context**,
then returns to your model once it is available again:

```jsonc
// olaya.json (any provider/model you have configured)
{
  "model": "anthropic/claude-sonnet-5",
  "failover": {
    "models": ["openai/gpt-5", "moonshotai/kimi-k2", "ollama/qwen3-coder"],
    "wait_for_reset": 20, // minutes: wait for your model if it's back by then
    "max_wait": 480 // minutes: if every model is down, wait this long before giving up
  }
}
```

- **What triggers a switch.** Exhausted quota or usage windows (with their reset times), long
  rate limits, overload and outages, failing credentials, and streams that stall. Short
  throttles are simply waited out. Each provider's error shapes are classified individually,
  because a 429 means different things at different providers.
- **What carries over.** The full history. Tool calls that already ran are kept, not re-run.
  Tool-call IDs and reasoning fields are rewritten for the next provider. The context is
  compacted to fit a smaller window, on the new model, never on the one that failed.
- **When everything is down,** Olaya waits for the earliest reset and carries on by itself.
  Cooldowns are kept on disk, so the next `olaya run` skips a provider that is still exhausted.
- **Only one provider?** `"failover": {}` turns this on without fallbacks: a usage limit or an
  outage is waited out on your model instead of ending the session. Failures that waiting can't
  fix (a rejected key, a request the provider refuses) still stop it, with the provider's error.
- **If the process itself dies,** `olaya run --supervise` restarts the task on the same session.
  Restarts back off and are capped at 6 an hour.
- **If the model loops** (the same tool call three times running, or three failures with the same
  error), it is told to re-read the current state and change approach, at most 3 times a run.
- **If the model stops mid-thought** ("Let me check the file." with no tool call), it is asked to
  take that step or say it is done.

| Environment variable | Effect |
|---|---|
| `OLAYA_DISABLE_LOOP_GUARD=1` | no reminders when the model repeats itself |
| `OLAYA_DISABLE_ACTION_NUDGE=1` | a run may end on an announced step |
| `OLAYA_EXPERIMENTAL_VERIFY_BEFORE_EXIT=1` | one "check your work" message before a run ends |
- **Stalls.** A stream silent for `stall_timeout` seconds (default 300) while no tool runs counts as
  stalled, and the next model takes over; the stalled one is tried again after a growing backoff.
  Without a failover block, a stall is still caught after 600 s and the step retried. Local
  servers such as Ollama send a tool call only when it is complete, so set `stall_timeout` above
  the longest one your model writes (output tokens ÷ tokens per second).
- **Privacy.** Every model in the chain receives the task's context, including code, when it
  takes over. A local model keeps it on your machine. A local model needs a declared
  `limit.context`, because local servers drop old messages silently when it is exceeded.

Settings → Routing in the app, and `/failover` in the terminal, edit the chain.

## Build from source

Requirements: [Bun](https://bun.sh) 1.3+, and Python 3.12 with [uv](https://docs.astral.sh/uv/)
for the decision layer.

```sh
git clone https://github.com/codelikeabhi/olaya && cd olaya
bun install
bun dev                 # run the CLI/TUI from source

# decision layer (optional; Observe mode). With an installed olaya: `olaya laya setup`. From source:
cd laya
uv venv --python 3.12 .venv && uv pip install --python .venv -e '.[train]'
OLAYA_LAYA_PORT=8731 .venv/bin/python service.py
# then, in another shell:
OLAYA_LAYA_ENABLED=1 OLAYA_LAYA_SHADOW=1 OLAYA_LAYA_URL=http://127.0.0.1:8731 bun dev
```

Coming from OpenCode? Olaya imports your OpenCode configuration and data once on first run
(OpenCode's own copy is left untouched). It also reads a project's `opencode.json` and
`.opencode/` when it has no Olaya config, and honours `OPENCODE_*` environment variables.

## Repository layout

| path | what |
|---|---|
| `packages/olaya` | the CLI and agent runtime |
| `packages/core`, `server`, `sdk`, `plugin`, `tui`, `app`, `desktop`, `ui` | harness packages inherited from OpenCode |
| `packages/olaya/src/laya` | the decision-layer plugin: compaction, judgment, shadow logging |
| `laya/laya` | the Laya model, vendored and modified (see `laya/laya/VENDORED.md`) |
| `laya/service.py` | the local decision sidecar |
| `laya/train` | data, fine-tuning on Apple Silicon, calibration, checkpoint verification, labelling |
| `laya/bench` | OlayaBench (decision benchmark) and the Harbor adapter for end-to-end harness benchmarks |
| `script/olaya-rename.ts` | the deterministic codemod that renamed OpenCode to Olaya, also used to port upstream fixes |

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Useful places to start:
new OlayaBench items (especially actions that *should* be asked about), decision hooks in the
harness, and the training and evaluation pipeline.

## License and attribution

Olaya is licensed under the [Apache License 2.0](LICENSE). It includes software from the
OpenCode project (MIT) and the Laya project (Apache-2.0); see [NOTICE](NOTICE) and
[LICENSES/](LICENSES). Olaya is an independent project and is not affiliated with OpenCode,
Anomaly, Convai Innovations or TypeSafe. "OpenCode Zen" remains available in Olaya as a
third-party model provider.
