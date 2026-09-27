"""Does a harness nudge help? A seeded paired A/B on Track C, one feature at a time.

    python -m bench.loop_ab run --binary <build> [--feature loop-guard|verify|action] [--model qwen3-8b-16k]
    python -m bench.loop_ab report [--feature ...]     # <feature>/ab.json

- loop-guard: a reminder after three repeated or identically failing tool calls (on by default;
  the "off" arm sets OLAYA_DISABLE_LOOP_GUARD);
- verify: HR1, one "check your work" nudge before the run ends (off by default; the "on" arm sets
  OLAYA_EXPERIMENTAL_VERIFY_BEFORE_EXIT). Its Terminal-Bench ablation sat on the floor (qwen3:8b
  solved nothing either way); Track C's 8b solve rate is 28%;
- action: when a run is about to end on a sentence announcing its next step with no tool call, one
  "take that step now" nudge (on by default; "off" sets OLAYA_DISABLE_ACTION_NUDGE).

Every Track C item runs once per arm, on one build, through Track E's proxy with no fault, which
puts the same seed on every request. Both arms are identical until the nudge first fires, so a
paired difference is the nudge's doing. Items where it never fires pair up as ties.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time

from . import cachesim, route
from .outage import free_port
from .retention_ab import paired
from .run import REPORTS

# per feature: run-directory prefix, each arm's environment, and the nudge's metadata as
# `olaya run --format json` prints it
FEATURES = {
    "loop-guard": {"prefix": "loop-ab", "env": {"off": {"OLAYA_DISABLE_LOOP_GUARD": "1"}, "on": {}},
                   "marker": '"loop_guard_nudge":true'},
    "verify": {"prefix": "verify-ab", "env": {"off": {}, "on": {"OLAYA_EXPERIMENTAL_VERIFY_BEFORE_EXIT": "1"}},
               "marker": '"loop_exit_nudge":true'},
    "action": {"prefix": "action-ab", "env": {"off": {"OLAYA_DISABLE_ACTION_NUDGE": "1"}, "on": {}},
               "marker": '"action_nudge":true'},
}
HERE = os.path.dirname(os.path.abspath(__file__))


def variant(feature, arm, binary, port, model):
    f = FEATURES[feature]
    entry = {"name": model, "tools": True, "options": {"reasoningEffort": "none"}}
    return {"name": f"{f['prefix']}-{arm}", "binary": binary, "cli_model": f"prova/{model}", "env": f["env"][arm],
            "config": {"provider": {"prova": {"npm": "@ai-sdk/openai-compatible", "name": "Seeded Ollama",
                                              "options": {"baseURL": f"http://host.docker.internal:{port}/a/v1"},
                                              "models": {model: entry}}}}}


def fired(out_dir, feature):
    """Nudges given in a run, counted from its events."""
    path = os.path.join(out_dir, "0.events.jsonl")
    if not os.path.exists(path):
        return 0
    return sum(1 for l in open(path) if FEATURES[feature]["marker"] in l)


def run(binary, model, feature):
    prefix = FEATURES[feature]["prefix"]
    for it in (json.loads(l) for l in open(route.ITEMS)):
        for arm in ("off", "on"):
            port = free_port()
            log = os.path.join(os.path.dirname(route.RUNS), "variants", f"{prefix}-{arm}", model, it["id"], "0.proxy.jsonl")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            proxy = subprocess.Popen([sys.executable, "-m", "bench.outage", "proxy", "--port", str(port), "--scenario", "none",
                                      "--log", log], cwd=os.path.dirname(HERE))
            try:
                for _ in range(50):
                    with socket.socket() as s:
                        if s.connect_ex(("127.0.0.1", port)) == 0:
                            break
                    time.sleep(0.2)
                r = route.run_one(it, model, 0, variant(feature, arm, binary, port, model))
            finally:
                proxy.terminate()
                proxy.wait(timeout=10)
            print(f"{arm:3} {it['id']:28} passed={r['passed']} steps={len(r['calls'])} {r['wall_s']}s", flush=True)


def report(model, feature):
    base = os.path.join(os.path.dirname(route.RUNS), "variants")
    prefix = FEATURES[feature]["prefix"]
    runs = {arm: {} for arm in ("off", "on")}
    for arm in runs:
        for it in (json.loads(l) for l in open(route.ITEMS)):
            path = os.path.join(base, f"{prefix}-{arm}", model, it["id"], "0.json")
            if os.path.exists(path):
                runs[arm][it["id"]] = (json.load(open(path)), fired(os.path.dirname(path), feature))
    both = [i for i in runs["off"] if i in runs["on"]]
    cost = lambda r: cachesim.bill([cachesim.Call(**c) for c in r["calls"]]).total
    arm = lambda a: {
        "resolved": round(sum(runs[a][i][0]["passed"] for i in both) / max(1, len(both)), 4),
        "mean_steps": round(sum(len(runs[a][i][0]["calls"]) for i in both) / max(1, len(both)), 2),
        "mean_cost": round(sum(cost(runs[a][i][0]) for i in both) / max(1, len(both)), 6),
        "timeouts": sum(runs[a][i][0]["timed_out"] for i in both),
    }
    touched = [i for i in both if runs["on"][i][1]]
    rep = {
        "experiment": f"{feature} A/B", "generated": time.strftime("%Y-%m-%d %H:%M"), "model": model,
        "priced_as": cachesim.PROXY.get(model, model), "pairs": len(both), "arms": {"off": arm("off"), "on": arm("on")},
        "fired_in": len(touched),
        "resolve_delta": paired([float(runs["off"][i][0]["passed"]) for i in both], [float(runs["on"][i][0]["passed"]) for i in both]) if both else None,
        "where_it_fired": {i: {"off": runs["off"][i][0]["passed"], "on": runs["on"][i][0]["passed"], "nudges": runs["on"][i][1],
                               "steps_off": len(runs["off"][i][0]["calls"]), "steps_on": len(runs["on"][i][0]["calls"])} for i in touched},
    }
    os.makedirs(os.path.join(REPORTS, feature), exist_ok=True)
    path = os.path.join(REPORTS, feature, "ab.json")
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print("->", path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--binary", required=True)
    p = sub.add_parser("report")
    for x in (r, p):
        x.add_argument("--model", default="qwen3-8b-16k"); x.add_argument("--feature", choices=FEATURES, default="loop-guard")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.binary, a.model, a.feature)
    else:
        report(a.model, a.feature)


if __name__ == "__main__":
    main()
