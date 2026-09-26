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
import json
import os
import random
import time

from . import cachesim
from . import route
from .run import REPORTS

# Usable history is limit.input - reserved = 10k tokens, with a 4k output allowance inside the model's
# real 16k window, and a 1k verbatim tail so one compaction does not trigger the next at once.
LIMIT = {"limit": {"context": 16_384, "input": 11_000, "output": 4_096}}
COMPACTION = {"compaction": {"reserved": 1_000, "preserve_recent_tokens": 1_000}}
ARMS = {"compaction": {}, "retention": {"OLAYA_LAYA_RETENTION": "live"}}


def variant(arm, binary):
    return {"name": f"retention-ab-{arm}", "model": LIMIT, "config": COMPACTION, "env": ARMS[arm], "binary": binary}


def run(model, binary):
    items = [json.loads(l) for l in open(route.ITEMS)]
    for it in items:  # arms alternate per item, so drift in machine load hits both alike
        for arm in ARMS:
            r = route.run_one(it, model, 0, variant(arm, binary))
            print(f"{arm:10} {it['id']:28} passed={r['passed']} steps={len(r['calls'])} "
                  f"compactions={len(r.get('summaries', []))} {r['wall_s']}s", flush=True)


def paired(a, b, rng=None, n=2000):
    """Mean of b - a with a percentile bootstrap 95% interval over pairs."""
    rng = rng or random.Random(0)
    d = [y - x for x, y in zip(a, b)]
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return {"delta": round(sum(d) / len(d), 4), "lcb95": round(means[int(0.025 * n)], 4), "ucb95": round(means[int(0.975 * n) - 1], 4)}


def report(model):
    items = [json.loads(l) for l in open(route.ITEMS)]
    runs = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for it in items:
            path = os.path.join(os.path.dirname(route.RUNS), "variants", f"retention-ab-{arm}", model, it["id"], "0.json")
            if os.path.exists(path):
                runs[arm][it["id"]] = json.load(open(path))
    both = [i for i in runs["compaction"] if i in runs["retention"]]
    per = lambda arm, f: [f(runs[arm][i]) for i in both]
    cost = lambda r: cachesim.bill([cachesim.Call(**c) for c in r["calls"]]).total
    summary = {arm: {"resolved": round(sum(per(arm, lambda r: r["passed"])) / max(1, len(both)), 4),
                     "mean_steps": round(sum(per(arm, lambda r: len(r["calls"]))) / max(1, len(both)), 2),
                     "mean_compactions": round(sum(per(arm, lambda r: len(r.get("summaries", [])))) / max(1, len(both)), 2),
                     "mean_cost": round(sum(per(arm, cost)) / max(1, len(both)), 6),
                     "timeouts": sum(per(arm, lambda r: r["timed_out"]))} for arm in ARMS}
    base = summary["compaction"]
    rep = {
        "pilot": "G9 local", "generated": time.strftime("%Y-%m-%d %H:%M"), "model": model,
        "priced_as": cachesim.PROXY.get(model, model), "declared_window": LIMIT["limit"], "compaction_config": COMPACTION["compaction"], "pairs": len(both),
        "self_test": "pass" if self_test() else "fail", "arms": summary,
        "resolve_delta": paired(per("compaction", lambda r: float(r["passed"])), per("retention", lambda r: float(r["passed"]))) if both else None,
        "cost_reduction": round(1 - summary["retention"]["mean_cost"] / base["mean_cost"], 4) if base["mean_cost"] else None,
        "turns_increase": round(summary["retention"]["mean_steps"] / base["mean_steps"] - 1, 4) if base["mean_steps"] else None,
        "note": "pilot: 32 pairs cannot certify the -5 pp margin; not G9's live-ab.json",
    }
    os.makedirs(os.path.join(REPORTS, "retention"), exist_ok=True)
    path = os.path.join(REPORTS, "retention", "pilot-ab.json")
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print("->", path)


def self_test():
    p = paired([0, 0, 1, 1], [1, 1, 1, 1])
    assert p["delta"] == 0.5 and p["lcb95"] <= 0.5 <= p["ucb95"]
    same = paired([1, 0, 1], [1, 0, 1])
    assert same == {"delta": 0.0, "lcb95": 0.0, "ucb95": 0.0}
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--model", default="qwen3-8b-16k"); r.add_argument("--binary", required=True)
    p = sub.add_parser("report"); p.add_argument("--model", default="qwen3-8b-16k")
    sub.add_parser("demo")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.model, a.binary)
    elif a.cmd == "report":
        report(a.model)
    else:
        print("retention A/B self-test", "passed" if self_test() else "FAILED")


if __name__ == "__main__":
    main()
