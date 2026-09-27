"""A first real-model pilot for gate G6: Olaya's router live against always the strongest model.

    python -m bench.routing_ab run --binary <linux build> --only a,b,c [--budget 1.5]
    python -m bench.routing_ab report                    # router/live-pilot.json

- routed:    starts on the cheapest model of the pool (claude-haiku-4-5) and escalates, one step up
             by price, only on the router's hard triggers (a doom loop, malformed calls, no test
             progress, edits piling up without tests);
- strongest: claude-sonnet-5 throughout.

Each Track C item runs once per arm; cost is what Anthropic billed. This checks the live path and
gives a first read of cost against success; G6 itself asks for a paired A/B at k = 5 on an agreed
task set and budget, which this is not.
"""

import argparse
import json
import os
import time

from . import route
from .retention_ab import paired
from .run import REPORTS

CHEAP, STRONG = "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"
ARMS = {
    "routed": {"cli_model": CHEAP, "env": {"OLAYA_LAYA_ROUTING": "live", "OLAYA_LAYA_ROUTING_START": "cheapest"},
               "config": {"routing": {"enabled": True, "models": [CHEAP, STRONG]}}},
    "strongest": {"cli_model": STRONG},
}
NAME = "claude-pool"  # names the run directories


def variant(arm, binary):
    return {"name": f"routing-ab-{arm}", "binary": binary, "secrets": True, "timeout": 900, **ARMS[arm]}


def run(binary, only, budget=None):
    items = [it for it in (json.loads(l) for l in open(route.ITEMS)) if it["id"] in only]
    spent = 0.0
    for it in items:
        if budget is not None and spent >= budget:
            print(f"budget reached: ${spent:.2f} of ${budget:.2f}; stopping", flush=True)
            break
        for arm in ARMS:
            r = route.run_one(it, NAME, 0, variant(arm, binary))
            spent += r.get("cost") or 0
            print(f"{arm:9} {it['id']:26} passed={r['passed']} steps={len(r['calls'])} {r['wall_s']}s "
                  f"${r.get('cost') or 0:.4f} (total ${spent:.2f})", flush=True)


def report():
    base = os.path.join(os.path.dirname(route.RUNS), "variants")
    runs = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for f in os.listdir(os.path.join(base, f"routing-ab-{arm}", NAME)) if os.path.isdir(os.path.join(base, f"routing-ab-{arm}", NAME)) else []:
            p = os.path.join(base, f"routing-ab-{arm}", NAME, f, "0.json")
            if os.path.exists(p):
                runs[arm][f] = json.load(open(p))
    both = [i for i in runs["routed"] if i in runs["strongest"]]
    arm = lambda a: {"resolved": round(sum(runs[a][i]["passed"] for i in both) / max(1, len(both)), 4),
                     "mean_cost": round(sum(runs[a][i].get("cost") or 0 for i in both) / max(1, len(both)), 4),
                     "mean_steps": round(sum(len(runs[a][i]["calls"]) for i in both) / max(1, len(both)), 2)}
    rep = {"pilot": "G6 live (Claude)", "generated": time.strftime("%Y-%m-%d %H:%M"), "cheap": CHEAP, "strong": STRONG,
           "pairs": len(both), "arms": {a: arm(a) for a in ARMS},
           "resolve_delta": paired([float(runs["strongest"][i]["passed"]) for i in both], [float(runs["routed"][i]["passed"]) for i in both]) if both else None,
           "items": {i: {a: {"passed": runs[a][i]["passed"], "cost": runs[a][i].get("cost")} for a in ARMS} for i in both},
           "note": "pilot, one run per arm: it cannot certify G6's margin"}
    os.makedirs(os.path.join(REPORTS, "router"), exist_ok=True)
    path = os.path.join(REPORTS, "router", "live-pilot.json")
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ("pairs", "arms", "resolve_delta")}, indent=1))
    print("->", path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--binary", required=True); r.add_argument("--only", required=True)
    r.add_argument("--budget", type=float)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.binary, set(a.only.split(",")), a.budget)
    else:
        report()


if __name__ == "__main__":
    main()
