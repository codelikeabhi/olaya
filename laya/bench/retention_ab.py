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
import shutil
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


BUNDLES = os.path.join(route.HOME, "bundles")


def bundles(size, seed=0):
    """Long tasks from short ones: `size` Exercism exercises in one workspace and one instruction to
    make every test pass, so the context grows past a realistic window. Groups follow item order,
    or a shuffle of it for another `seed` (more pairs from the same exercises)."""
    singles = [it for it in load_items() if it["id"] not in route.STUB_PASSES]
    if seed:
        random.Random(seed).shuffle(singles)
    tag = f"b{size}" + (f"s{seed}" if seed else "")
    out = []
    for b in range(len(singles) // size):
        group = singles[b * size:(b + 1) * size]
        files = [f for it in group for f in it["solution"] + it["test"]]
        assert len(files) == len(set(files)), f"file names collide in bundle {b}"
        d = os.path.join(BUNDLES, f"{tag}-{b}")
        os.makedirs(d, exist_ok=True)
        for it in group:
            for f in it["solution"] + it["test"]:
                shutil.copy(os.path.join(it["dir"], f), os.path.join(d, f))
        out.append({"id": f"{tag}-{b}", "difficulty": max(it["difficulty"] for it in group), "dir": d,
                    "solution": [f for it in group for f in it["solution"]], "test": [f for it in group for f in it["test"]],
                    "instructions": "\n\n".join(f"## {it['id']}\n\n{it['instructions']}" for it in group)})
    return out


def variant(arm, binary, cloud=None, window=None):
    """`window`: the declared input limit for a cloud model, in place of the forced-small default."""
    # forced compaction adds summary requests, and local models write long; 20 minutes per run
    v = {"name": f"retention-ab-{arm}", "model": LIMIT, "config": COMPACTION, "env": ARMS[arm], "binary": binary,
         "timeout": 1200}
    if cloud:  # e.g. anthropic/claude-haiku-4-5: the same forced window on the provider's own model entry
        provider, model_id = cloud.split("/", 1)
        limit = {"limit": {**LIMIT["limit"], "input": window or INPUT.get(provider, LIMIT["limit"]["input"])}}
        if window:
            limit["limit"]["context"] = max(LIMIT["limit"]["context"], window + LIMIT["limit"]["output"])
        v.update(cli_model=cloud, secrets=provider not in SIGNIN, timeout=1800 if window else 900,
                 config={**COMPACTION, "provider": {provider: {"models": {model_id: limit}}}})
        if provider in SIGNIN:  # the sandbox gets the sign-in's access token only (route.signin_env)
            v.update(signin=provider)
    return v


def solvable(model):
    """Items the model solved in Track C's first pass: where retention can make a difference.
    On an item the model cannot solve anyway, both arms fail and the pair says nothing."""
    return {r["item"] for f in glob.glob(os.path.join(route.RUNS, model, "*", "0.json"))
            if (r := json.load(open(f)))["passed"] and r["item"] not in route.STUB_PASSES}


def run(model, binary, only_solvable=False, cloud=None, only=None, budget=None, bundle=None, window=None, seed=0):
    """`cloud`: run on a provider's model (its key from the harness secrets file) instead of a local
    one; `model` then only names the run directories. `budget`: stop before spending more (USD).
    `bundle`: long tasks of that many exercises each; `window`: the cloud model's declared input limit."""
    items = bundles(bundle, seed) if bundle else load_items()
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
            r = route.run_through_limits(it, model, 0, variant(arm, binary, cloud, window))
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
    runs = runs_of(model)
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


def runs_of(model):
    """arm -> item -> run record, for every run of this model (single items or bundles)."""
    runs = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for path in glob.glob(os.path.join(os.path.dirname(route.RUNS), "variants", f"retention-ab-{arm}", model, "*", "0.json")):
            runs[arm][os.path.basename(os.path.dirname(path))] = json.load(open(path))
    return runs


def cache_hit(model, arm, item):
    """(prompt tokens read from a cache, all prompt tokens) over a run's steps."""
    path = os.path.join(os.path.dirname(route.RUNS), "variants", f"retention-ab-{arm}", model, item, "0.events.jsonl")
    hit = total = 0
    for e in (json.loads(l) for l in open(path) if l.startswith("{")) if os.path.exists(path) else ():
        if e.get("type") == "step_finish":
            tok = (e.get("part") or {}).get("tokens") or {}
            cache = tok.get("cache") or {}
            hit += cache.get("read") or 0
            total += (tok.get("input") or 0) + (cache.get("read") or 0) + (cache.get("write") or 0)
    return hit, total


def gate(model):
    """G9's report, retention/live-ab.json, from this model's pairs, with its deviations written in."""
    from .routing_ab import ratio_ucb

    runs = runs_of(model)
    both = [i for i in runs["compaction"] if i in runs["retention"]]
    if not both:
        raise SystemExit("no pairs yet")
    cost = lambda r: r["list_cost"] if "list_cost" in r else cachesim.bill([cachesim.Call(**c) for c in r["calls"]]).total
    per = lambda arm, f: [f(runs[arm][i]) for i in both]
    ratio, ucb = ratio_ucb(per("compaction", cost), per("retention", cost))
    prompt = lambda r: sum(c["prompt"] for c in r["calls"])
    mean = lambda xs: sum(xs) / len(xs)
    hits = {arm: [cache_hit(model, arm, i) for i in both] for arm in ARMS}
    hit = {arm: sum(h for h, _ in v) / max(1, sum(t for _, t in v)) for arm, v in hits.items()}
    reacq = {arm: mean(per(arm, reacquisitions)) for arm in ARMS}
    resolve = paired(per("compaction", lambda r: float(r["passed"])), per("retention", lambda r: float(r["passed"])))
    rep = {
        "gate": "G9", "generated": time.strftime("%Y-%m-%d %H:%M"), "model": model, "pairs": len(both), "k": 1,
        "declared_window": LIMIT["limit"], "compaction_config": COMPACTION["compaction"],
        "metrics": {
            "resolve_delta_lcb95": resolve["lcb95"],
            "input_token_reduction": round(1 - sum(per("retention", prompt)) / max(1, sum(per("compaction", prompt))), 4),
            "cost_reduction": round(1 - ratio, 4),
            "cost_reduction_lcb95": round(1 - ucb, 4),
            "turns_increase": round(mean(per("retention", lambda r: len(r["calls"]))) / max(1e-9, mean(per("compaction", lambda r: len(r["calls"])))) - 1, 4),
            "reacquisition_increase": round(reacq["retention"] / reacq["compaction"] - 1, 4) if reacq["compaction"] else round(reacq["retention"], 4),
            "cache_hit_ratio_delta": round(hit["retention"] - hit["compaction"], 4),
            # none of these tasks carries injected content, so nothing here can be promoted; the
            # retention tests (provenance pinning) are where that is checked
            "injected_promotions": None,
        },
        "detail": {"resolve_delta": resolve, "resolved": {arm: round(mean(per(arm, lambda r: float(r["passed"]))), 4) for arm in ARMS},
                   "compactions": {arm: round(mean(per(arm, lambda r: r.get("compactions", 0))), 2) for arm in ARMS},
                   "cache_hit": {arm: round(v, 4) for arm, v in hit.items()}, "reacquisitions": {arm: round(v, 2) for arm, v in reacq.items()}},
        "deviations": ["k = 1 run per arm and task", "window forced small so compaction happens every few steps",
                       "Track C Exercism tasks, not long Harbor tasks", "injected_promotions not measurable here (no injected content)"],
    }
    os.makedirs(os.path.join(REPORTS, "retention"), exist_ok=True)
    path = os.path.join(REPORTS, "retention", "live-ab.json")
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ("pairs", "metrics", "detail")}, indent=1))
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
    r.add_argument("--bundle", type=int, help="long tasks: this many exercises per workspace")
    r.add_argument("--window", type=int, help="the cloud model's declared input limit (tokens)")
    r.add_argument("--seed", type=int, default=0, help="another grouping of the exercises into bundles")
    p = sub.add_parser("report"); p.add_argument("--model", default="qwen3-8b-16k")
    g = sub.add_parser("gate"); g.add_argument("--model", required=True)
    sub.add_parser("demo")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.model, a.binary, a.solvable, a.cloud, set(a.only.split(",")) if a.only else None, a.budget, a.bundle, a.window, a.seed)
    elif a.cmd == "report":
        report(a.model)
    elif a.cmd == "gate":
        gate(a.model)
    else:
        print("retention A/B self-test", "passed" if self_test() else "FAILED")


if __name__ == "__main__":
    main()
