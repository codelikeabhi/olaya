"""Local pilot of gate G9: Olaya's compaction against Laya's extractive retention, forced often.

    python -m bench.retention_ab run --model qwen3-8b-16k --binary <linux build with the retention seam>
    python -m bench.retention_ab report                  # retention/pilot-ab.json
    python -m bench.retention_ab demo                    # paired-statistics self-test

Each Track C item runs once per arm on one local model whose usable history is declared as 10k
tokens, so the harness compacts every few steps:
- arm "compaction": Olaya's summary request (what ships today);
- arm "retention": OLAYA_LAYA_RETENTION=live, the extractive plan instead of a summary request.

This is a pilot. The design's non-inferiority margin (-5 points) needs about 630 paired runs;
32 pairs only show whether extractive retention works end to end, and roughly how it moves
resolve rate, steps and cache-aware cost. It never writes G9's live-ab.json.
"""

import argparse
import glob
import json
import os
import random
import time

from . import cachesim
from . import route
from .run import REPORTS

# Usable history is limit.input - reserved = 10k tokens, with a 5k output allowance inside the model's
# real 16k window, and a 1k verbatim tail so one compaction does not trigger the next at once.
LIMIT = {"limit": {"context": 16_384, "input": 11_000, "output": 5_000}}
COMPACTION = {"compaction": {"reserved": 1_000, "preserve_recent_tokens": 1_000}}
ARMS = {"compaction": {}, "retention": {"OLAYA_LAYA_RETENTION": "live"}}
# providers reached through the owner's sign-in (ChatGPT), not an API key
SIGNIN = {"openai"}
# GPT's system prompt is ~5.8k tokens against Claude's ~8.5k: its declared input limit is lowered so the
# headroom over the prompt, and so how often it compacts, matches the Claude pilot
INPUT = {"openai": 8_500}
# Track C's first 32 items and the 56 Exercism items added for the routing head, once they exist
ITEM_FILES = [route.ITEMS, os.path.join(route.HOME, "items-exercism-rest.jsonl")]


def load_items():
    return [json.loads(l) for f in ITEM_FILES if os.path.exists(f) for l in open(f)]


def variant(arm, binary, cloud=None):
    # forced compaction adds summary requests, and local models write long; 20 minutes per run
    v = {"name": f"retention-ab-{arm}", "model": LIMIT, "config": COMPACTION, "env": ARMS[arm], "binary": binary,
         "timeout": 1200}
    if cloud:  # e.g. anthropic/claude-haiku-4-5: the same forced window on the provider's own model entry
        provider, model_id = cloud.split("/", 1)
        limit = {"limit": {**LIMIT["limit"], "input": INPUT.get(provider, LIMIT["limit"]["input"])}}
        v.update(cli_model=cloud, secrets=provider not in SIGNIN, timeout=900,
                 config={**COMPACTION, "provider": {provider: {"models": {model_id: limit}}}})
        if provider in SIGNIN:  # the sandbox gets the sign-in's access token only (route.signin_env)
            v.update(signin=provider)
    return v


def solvable(model):
    """Items the model solved in Track C's first pass: where retention can make a difference.
    On an item the model cannot solve anyway, both arms fail and the pair says nothing."""
    return {r["item"] for f in glob.glob(os.path.join(route.RUNS, model, "*", "0.json"))
            if (r := json.load(open(f)))["passed"] and r["item"] not in route.STUB_PASSES}


def run(model, binary, only_solvable=False, cloud=None, only=None, budget=None):
    """`cloud`: run on a provider's model (its key from the harness secrets file) instead of a local
    one; `model` then only names the run directories. `budget`: stop before spending more (USD)."""
    items = load_items()
    if only_solvable:
        keep = solvable(model)
        items = [it for it in items if it["id"] in keep]
        print(f"{len(items)} items {model} solved in Track C", flush=True)
    if only:
        items = [it for it in items if it["id"] in only]
    spent = 0.0
    for it in items:  # arms alternate per item, so drift in machine load hits both alike
        if budget is not None and spent >= budget:
            print(f"budget reached: ${spent:.2f} of ${budget:.2f}; stopping", flush=True)
            break
        for arm in ARMS:
            r = route.run_through_limits(it, model, 0, variant(arm, binary, cloud))
            spent += r["list_cost"] if "list_cost" in r else (r.get("cost") or 0)
            print(f"{arm:10} {it['id']:28} passed={r['passed']} steps={len(r['calls'])} "
                  f"compactions={r.get('compactions', 0)} {r['wall_s']}s ${r.get('cost') or 0:.4f} (total ${spent:.2f})", flush=True)


def paired(a, b, rng=None, n=2000):
    """Mean of b - a with a percentile bootstrap 95% interval over pairs."""
    rng = rng or random.Random(0)
    d = [y - x for x, y in zip(a, b)]
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return {"delta": round(sum(d) / len(d), 4), "lcb95": round(means[int(0.025 * n)], 4), "ucb95": round(means[int(0.975 * n) - 1], 4)}


def reacquisitions(r):
    """Calls to `recall` plus repeats of an identical earlier call (re-reading the same file)."""
    uses = [tuple(u) for u in r.get("tool_uses", [])]
    return sum(1 for tool, _ in uses if tool == "recall") + len(uses) - len(set(uses))


def report(model):
    items = load_items()
    runs = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for it in items:
            path = os.path.join(os.path.dirname(route.RUNS), "variants", f"retention-ab-{arm}", model, it["id"], "0.json")
            if os.path.exists(path):
                runs[arm][it["id"]] = json.load(open(path))
    both = [i for i in runs["compaction"] if i in runs["retention"]]
    per = lambda arm, f: [f(runs[arm][i]) for i in both]
    # a sign-in run carries its list price; local runs are billed by the cache simulator
    cost = lambda r: r["list_cost"] if "list_cost" in r else cachesim.bill([cachesim.Call(**c) for c in r["calls"]]).total
    summary = {arm: {"resolved": round(sum(per(arm, lambda r: r["passed"])) / max(1, len(both)), 4),
                     "mean_steps": round(sum(per(arm, lambda r: len(r["calls"]))) / max(1, len(both)), 2),
                     "mean_compactions": round(sum(per(arm, lambda r: r.get("compactions", 0))) / max(1, len(both)), 2),
                     "mean_cost": round(sum(per(arm, cost)) / max(1, len(both)), 6),
                     "mean_reacquisitions": round(sum(per(arm, reacquisitions)) / max(1, len(both)), 2),
                     "timeouts": sum(per(arm, lambda r: r["timed_out"]))} for arm in ARMS}
    base = summary["compaction"]
    rep = {
        "pilot": "G9 local" if model in cachesim.PROXY else f"G9 {model}", "generated": time.strftime("%Y-%m-%d %H:%M"), "model": model,
        "priced_as": cachesim.PROXY.get(model, model), "declared_window": LIMIT["limit"], "compaction_config": COMPACTION["compaction"], "pairs": len(both),
        "self_test": "pass" if self_test() else "fail", "arms": summary,
        "resolve_delta": paired(per("compaction", lambda r: float(r["passed"])), per("retention", lambda r: float(r["passed"]))) if both else None,
        "cost_reduction": round(1 - summary["retention"]["mean_cost"] / base["mean_cost"], 4) if base["mean_cost"] else None,
        "turns_increase": round(summary["retention"]["mean_steps"] / base["mean_steps"] - 1, 4) if base["mean_steps"] else None,
        "reacquisition_increase": round(summary["retention"]["mean_reacquisitions"] - base["mean_reacquisitions"], 2),
        "note": "pilot on items the model solved in Track C: it cannot certify the -5 pp margin; not G9's live-ab.json",
    }
    os.makedirs(os.path.join(REPORTS, "retention"), exist_ok=True)
    path = os.path.join(REPORTS, "retention", "pilot-ab.json" if model == "qwen3-8b-16k" else f"pilot-ab-{model}.json")
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print("->", path)


def self_test():
    p = paired([0, 0, 1, 1], [1, 1, 1, 1])
    assert p["delta"] == 0.5 and p["lcb95"] <= 0.5 <= p["ucb95"]
    assert reacquisitions({"tool_uses": [["read", "a"], ["read", "a"], ["recall", "h"], ["bash", "x"]]}) == 2
    same = paired([1, 0, 1], [1, 0, 1])
    assert same == {"delta": 0.0, "lcb95": 0.0, "ucb95": 0.0}
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--model", default="qwen3-8b-16k"); r.add_argument("--binary", required=True)
    r.add_argument("--solvable", action="store_true", help="only items the model solved in Track C")
    r.add_argument("--cloud", help="a provider's model, e.g. anthropic/claude-haiku-4-5 (key from the harness secrets file)")
    r.add_argument("--only", help="comma-separated item ids"); r.add_argument("--budget", type=float, help="stop at this spend (USD)")
    p = sub.add_parser("report"); p.add_argument("--model", default="qwen3-8b-16k")
    sub.add_parser("demo")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.model, a.binary, a.solvable, a.cloud, set(a.only.split(",")) if a.only else None, a.budget)
    elif a.cmd == "report":
        report(a.model)
    else:
        print("retention A/B self-test", "passed" if self_test() else "FAILED")


if __name__ == "__main__":
    main()
