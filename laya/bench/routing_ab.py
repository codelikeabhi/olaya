"""A first pilot for gate G6: Olaya's router live against always the strongest model.

    python -m bench.routing_ab run --binary <linux build> --only a,b,c [--budget 1.5] [--pool local]
    python -m bench.routing_ab report [--pool local]      # router/live-pilot[-local].json

- routed:    starts on the cheapest model of the pool and escalates, one step up by price, only on
             the router's hard triggers (a doom loop, malformed calls, no test progress, edits
             piling up without tests);
- strongest: the pool's strongest model throughout.

Pools: `claude` (Haiku 4.5 -> Sonnet 5, billed by Anthropic; needs API credit), `local` (qwen3
4b -> 8b in Ollama, priced as the Claude tiers they stand in for, bench.cachesim.PROXY) and
`chatgpt` (gpt-6-luna -> gpt-6-sol through the owner's ChatGPT sign-in: the sandbox gets the
access token only; priced per step at OpenAI's list prices). A run stopped by a plan's usage limit is set aside and
retried after a wait. Each Track
C item runs once per arm. This checks the live path and gives a first read of cost against success
and of protocol failures (calls to a tool that does not exist or with invalid input); G6 itself asks
for a paired A/B at k = 5 on an agreed task set and budget, which this is not.
"""

import argparse
import json
import os
import random
import time

from . import route
from .retention_ab import paired
from .run import REPORTS

POOLS = {
    "claude": {"cheap": "anthropic/claude-haiku-4-5", "strong": "anthropic/claude-sonnet-5", "secrets": True},
    "local": {"cheap": "ollama/qwen3-4b-16k", "strong": "ollama/qwen3-8b-16k", "secrets": False},
    # the owner's ChatGPT sign-in, on this machine; priced at OpenAI's list prices
    "chatgpt": {"cheap": "openai/gpt-6-luna", "strong": "openai/gpt-6-sol", "secrets": False, "signin": "openai"},
}
ARMS = ("routed", "strongest")


def name(pool):
    return f"{pool}-pool"  # names the run directories


def variant(arm, binary, pool="claude"):
    p = POOLS[pool]
    v = {"name": f"routing-ab-{arm}", "binary": binary, "secrets": p["secrets"], "timeout": 900, "signin": p.get("signin"),
         "pool": tuple(m.split("/", 1)[1] for m in (p["cheap"], p["strong"]) if m.startswith("ollama/"))}
    if arm == "routed":
        return {**v, "cli_model": p["cheap"], "env": {"OLAYA_LAYA_ROUTING": "live", "OLAYA_LAYA_ROUTING_START": "cheapest"},
                "config": {"routing": {"enabled": True, "models": [p["cheap"], p["strong"]]}}}
    return {**v, "cli_model": p["strong"]}


def run(binary, only, budget=None, pool="claude"):
    items = [it for f in (route.ITEMS, route.ITEMS.replace("items.jsonl", "items-easy.jsonl"))
             for it in (json.loads(l) for l in open(f)) if it["id"] in only]
    spent = 0.0
    for it in items:
        if budget is not None and spent >= budget:
            print(f"budget reached: ${spent:.2f} of ${budget:.2f}; stopping", flush=True)
            break
        for arm in ARMS:
            r = route.run_through_limits(it, name(pool), 0, variant(arm, binary, pool))
            spent += cost(r)
            print(f"{arm:9} {it['id']:26} passed={r['passed']} steps={len(r['calls'])} {r['wall_s']}s "
                  f"${cost(r):.4f} (total ${spent:.2f}) {r.get('steps_by_model', '')}", flush=True)


def cost(r):
    """Billed cost, or list price where a sign-in bills nothing per token."""
    return r["list_cost"] if "list_cost" in r else (r.get("cost") or 0)


def protocol_failures(r):
    return sum(1 for tool, _ in r.get("tool_uses", []) if tool == "invalid")


def run_dir(pool, arm, item):
    return os.path.join(os.path.dirname(route.RUNS), "variants", f"routing-ab-{arm}", name(pool), item)


def load(pool):
    """arm -> item -> run record, and the items both arms ran."""
    runs = {arm: {} for arm in ARMS}
    for arm in ARMS:
        d = os.path.dirname(run_dir(pool, arm, "x"))
        for f in os.listdir(d) if os.path.isdir(d) else []:
            if os.path.exists(os.path.join(d, f, "0.json")):
                runs[arm][f] = json.load(open(os.path.join(d, f, "0.json")))
    return runs, sorted(i for i in runs["routed"] if i in runs["strongest"])


def uncached(pool, arm, item):
    """(prompt tokens not read from a cache, all prompt tokens) over a run's steps: cache writes on
    Anthropic, uncached input on OpenAI, which caches without a write price."""
    path = os.path.join(run_dir(pool, arm, item), "0.events.jsonl")
    fresh = total = 0
    for e in (json.loads(l) for l in open(path) if l.startswith("{")) if os.path.exists(path) else ():
        if e.get("type") == "step_finish":
            tok = (e.get("part") or {}).get("tokens") or {}
            cache = tok.get("cache") or {}
            fresh += (tok.get("input") or 0) + (cache.get("write") or 0)
            total += (tok.get("input") or 0) + (cache.get("write") or 0) + (cache.get("read") or 0)
    return fresh, total


def ratio_ucb(a, b, n=2000, rng=None):
    """Point ratio sum(b)/sum(a) and its percentile-bootstrap 95% upper bound over pairs."""
    rng = rng or random.Random(0)
    boot = sorted(sum(b[j] for j in s) / max(1e-12, sum(a[j] for j in s))
                  for s in ([rng.randrange(len(a)) for _ in a] for _ in range(n)))
    return round(sum(b) / max(1e-12, sum(a)), 4), round(boot[int(0.975 * n) - 1], 4)


def gate(pool="chatgpt"):
    """G6's report, router/live-ab.json, from this pool's pairs. Its deviations from the design (k and
    tiers) are written into it."""
    runs, both = load(pool)
    if not both:
        raise SystemExit("no pairs yet")
    cost_ratio, cost_ucb = ratio_ucb([cost(runs["strongest"][i]) for i in both], [cost(runs["routed"][i]) for i in both])
    share = {a: [uncached(pool, a, i) for i in both] for a in ARMS}
    share = {a: sum(f for f, _ in v) / max(1, sum(t for _, t in v)) for a, v in share.items()}
    resolve = paired([float(runs["strongest"][i]["passed"]) for i in both], [float(runs["routed"][i]["passed"]) for i in both])
    rep = {
        "gate": "G6", "generated": time.strftime("%Y-%m-%d %H:%M"), "pool": pool, **{k: POOLS[pool][k] for k in ("cheap", "strong")},
        "pairs": len(both), "k": 1,
        "metrics": {"resolve_delta_lcb95": resolve["lcb95"], "cost_ratio_ucb95": cost_ucb,
                    "protocol_failures": sum(protocol_failures(runs["routed"][i]) for i in both),
                    "cache_write_share_ratio": round(share["routed"] / share["strongest"], 4) if share["strongest"] else None},
        "detail": {"resolve_delta": resolve, "cost_ratio": cost_ratio, "uncached_prompt_share": {a: round(v, 4) for a, v in share.items()},
                   "resolved": {a: round(sum(runs[a][i]["passed"] for i in both) / len(both), 4) for a in ARMS},
                   "escalated_runs": sum(len(runs["routed"][i].get("steps_by_model") or {}) > 1 for i in both)},
        "deviations": ["k = 1 run per arm and task, not the design's k = 5",
                       {"chatgpt": "OpenAI tiers through the owner's ChatGPT sign-in, priced at OpenAI's list prices, not Claude tiers billed by Anthropic",
                        "local": "local qwen3 tiers priced as Claude tiers", "claude": ""}[pool],
                       "Track C Exercism tasks, not the agreed Harbor set"],
    }
    os.makedirs(os.path.join(REPORTS, "router"), exist_ok=True)
    path = os.path.join(REPORTS, "router", "live-ab.json")
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ("pairs", "metrics", "detail")}, indent=1))
    print("->", path)


def report(pool="claude"):
    runs, both = load(pool)
    n = max(1, len(both))
    arm = lambda a: {"resolved": round(sum(runs[a][i]["passed"] for i in both) / n, 4),
                     "mean_cost": round(sum(cost(runs[a][i]) for i in both) / n, 4),
                     "mean_steps": round(sum(len(runs[a][i]["calls"]) for i in both) / n, 2),
                     "protocol_failures": sum(protocol_failures(runs[a][i]) for i in both)}
    rep = {"pilot": f"G6 live ({pool})", "generated": time.strftime("%Y-%m-%d %H:%M"), **{k: POOLS[pool][k] for k in ("cheap", "strong")},
           "pairs": len(both), "arms": {a: arm(a) for a in ARMS},
           "resolve_delta": paired([float(runs["strongest"][i]["passed"]) for i in both], [float(runs["routed"][i]["passed"]) for i in both]) if both else None,
           "items": {i: {a: {"passed": runs[a][i]["passed"], "cost": cost(runs[a][i]), "steps_by_model": runs[a][i].get("steps_by_model")} for a in ARMS} for i in both},
           "note": "pilot, one run per arm: it cannot certify G6's margin" + {"local": "; local tiers priced as the Claude tiers they stand in for",
                                                                       "chatgpt": "; ChatGPT sign-in, priced per step at OpenAI's list prices"}.get(pool, "")}
    os.makedirs(os.path.join(REPORTS, "router"), exist_ok=True)
    path = os.path.join(REPORTS, "router", "live-pilot.json" if pool == "claude" else f"live-pilot-{pool}.json")
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ("pairs", "arms", "resolve_delta")}, indent=1))
    print("->", path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--binary", required=True); r.add_argument("--only", required=True)
    r.add_argument("--budget", type=float); r.add_argument("--pool", choices=POOLS, default="claude")
    p = sub.add_parser("report"); p.add_argument("--pool", choices=POOLS, default="claude")
    g = sub.add_parser("gate"); g.add_argument("--pool", choices=POOLS, default="chatgpt")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.binary, set(a.only.split(",")), a.budget, a.pool)
    elif a.cmd == "gate":
        gate(a.pool)
    else:
        report(a.pool)


if __name__ == "__main__":
    main()
