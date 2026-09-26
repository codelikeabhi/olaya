# Laya decision sidecar

Loopback HTTP service wrapping `laya.Agent`. Olaya calls it to judge permission requests.

    uv venv --python 3.12 .venv
    uv pip install --python .venv -e '.[train]'     # installs the vendored Laya in ./laya
    .venv/bin/python -m service            # or: OLAYA_LAYA_PORT=8731 .venv/bin/python service.py

Endpoints: `GET /health`, `POST /decide`. See `service.py`.

The checkpoint is downloaded from Hugging Face on first run (~848 MB) and loading takes
around 21 s, so this process is long-lived and pre-warmed. It is never spawned per request.

## Running the decision layer

Everything is off by default. Enable it with environment variables:

| variable | meaning |
|---|---|
| `OLAYA_LAYA_ENABLED` | master switch; nothing spawns, runs or is logged without it |
| `OLAYA_LAYA_SHADOW` | record predictions and labels (requires `ENABLED`) |
| `OLAYA_LAYA_OBSERVE_OUTPUTS` | with shadow on, tool outputs are also scored for injection and logged locally (never modified, never awaited). Set to `0` to turn that off. |
| `OLAYA_LAYA_MODE` | `shadow` (default) observes only. `live` may turn an `ask` into `allow`, but only on a checkpoint whose manifest carries a passed certification `gate`, at or above its certified threshold; it never denies, and every grant is audit-logged. On an uncertified checkpoint `live` behaves exactly like `shadow`. |
| `OLAYA_LAYA_PYTHON` | interpreter that has `laya` installed |
| `OLAYA_LAYA_URL` | attach to a sidecar you started yourself instead of spawning one |
| `OLAYA_LAYA_TIMEOUT_MS` | per-decision ceiling (default 400) |
| `OLAYA_LAYA_STDERR` | route the sidecar's stderr to a file for diagnostics |
| `OLAYA_LAYA_SHADOW_DIR` | override the shadow log location |

Shadow records land in `$XDG_DATA_HOME/olaya/laya-shadow/` by default.

## Driving a real session with a local model

The decision layer only sees permission requests the static rules left as `ask`, so you
need a model that actually emits tool calls. Ollama works and needs no API key:

    ollama pull qwen3:8b

Then point olaya at it with a project `olaya.json`. Note that upstream's `.gitignore`
already ignores `/olaya.json`, so this file stays local and is not committed - which is
why the config is reproduced here:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": { "baseURL": "http://localhost:11434/v1" },
      "models": { "qwen3:8b": { "name": "Qwen3 8B (local)", "tools": true } }
    }
  }
}
```

If tool calls come back malformed, raise Ollama's context window
(`OLLAMA_CONTEXT_LENGTH=32768`); olaya's tool schemas are large and small models
truncate them at the default.

Both reply paths produce labelled training rows:

    olaya run --auto "..."   # replies "once"  -> approve labels
    olaya run "..."          # auto-rejects    -> reject labels

## Tests

    for t in tests/laya_upstream/test_*.py; do .venv/bin/python $t; done   # vendored Laya

    python test_service.py                                    # sidecar self-check, no model needed
    bun test test/laya                                        # unit tests, from packages/olaya
    OLAYA_LAYA_INTEGRATION=1 OLAYA_LAYA_PYTHON=... bun test test/laya/integration.test.ts

## From training to live mode

Live mode lets the decision layer approve an action instead of asking you. It only acts on a
checkpoint that passed certification: a statistical bound on how often it would approve
something you should have been asked about. The pipeline, all local:

```sh
cd laya
# 1. data: synthetic task-conditioned items, real agent actions, soft teacher labels
.venv/bin/python -m train.synth --out bench/items/synth-v3.jsonl
.venv/bin/python -m train.replay --out ~/.local/share/olaya/laya/data/replay.jsonl
.venv/bin/python -m train.teacher --items <items> --out <labelled items>

# 2. train (Apple Silicon: --device mps) and verify the checkpoint
.venv/bin/python -m train.train --data ~/.local/share/olaya/laya/data/<dataset> --out <checkpoint>
.venv/bin/python -m train.verify <checkpoint>

# 3. label a gold set yourself (only human labels can certify), rejects first
.venv/bin/python -m train.label --items <items> --out ~/.local/share/olaya/laya/gold/gold-v1.jsonl

# 4. certify: writes a pass/fail gate into the checkpoint's manifest
.venv/bin/python -m train.certify <checkpoint> --gold ~/.local/share/olaya/laya/gold/gold-v1.jsonl

# 5. serve it; live mode acts only if the gate passed, only above its threshold
OLAYA_LAYA_MODEL=<checkpoint> .venv/bin/python service.py
OLAYA_LAYA_ENABLED=1 OLAYA_LAYA_MODE=live OLAYA_LAYA_URL=http://127.0.0.1:<port> olaya
```

Certification needs at least 300 should-ask items in each half of the gold set. Below 299, a
1% false-approve bound cannot be shown at 95% confidence even with zero mistakes. Every
approval in live mode is written to the local audit log (`kind: "auto-approved"`).

## Benchmarks

- `python -m bench.run`: OlayaBench Track A (should this action run without asking?)
- `python -m bench.inject`: Track B (does this tool output carry instructions to the agent?)
- `bench/harbor_olaya.py`: Harbor adapters for end-to-end harness runs, with L3 tripwires and gated variants
- `python -m bench.combined`: one report across the decision layer and the harness
