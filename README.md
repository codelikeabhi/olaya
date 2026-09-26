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
> source (below). The decision layer runs in shadow mode only: it observes and logs, and never
> changes what the agent is allowed to do.

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

## Build from source

Requirements: [Bun](https://bun.sh) 1.3+, and Python 3.12 with [uv](https://docs.astral.sh/uv/)
for the decision layer.

```sh
git clone https://github.com/codelikeabhi/olaya && cd olaya
bun install
bun dev                 # run the CLI/TUI from source

# decision layer (optional; shadow mode only)
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
