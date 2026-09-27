"""Laya as a model router, gate G4: "will a model of this tier solve this coding task?"

    python -m train.router build                                   # Track C outcomes -> data/route-v1
    python -m train.train --data ~/.local/share/olaya/laya/data/route-v1 --out <checkpoint>
    python -m train.router report --checkpoint <checkpoint>        # held-out items -> router/latest.json
    python -m train.router demo                                    # self-test

Decision point D0, task start (design D1). Laya sees a deterministic state of at most 470 tokens
(design D6): a header, facts about the repository's files and tests, and the head of the task. There
is no scout handoff: offline runs have no scout turns. For each tier of the ladder Laya answers one
calibrated yes/no, P(success | tier t); the answers are made monotone in t (a stronger tier is never
predicted to do worse), which is the ordinal answer P(success | tier >= t) of design D2. A question
names the tier's rank only, so prices and providers stay out of the model.

The harness routes to the lowest tier whose probability clears one threshold, else the strongest.
The threshold is certified with Learn-Then-Test on the calibration items: harmful downgrades (routed
below the strongest tier, failed there, and the strongest tier succeeded) at most 5% with 95%
confidence (design D3). Items are split by id, so train, calibration and test share no task; every
Track C item is its own repository. Costs are the cache simulator's bills at the real tier each
local model stands in for.
"""

import argparse
import json
import os
import random
import re
import time

import numpy as np

from bench import cachesim, route
from bench.metrics import INF, binom_cdf, cp_upper
from bench.retention_ab import paired
from bench.run import REPORTS

from . import data as D
from .retain import auroc, fit_platt, model_probs, recalibrate

TIERS = ["qwen3-0.6b-16k", "qwen3-4b-16k", "qwen3-8b-16k"]
ITEM_FILES = ["items.jsonl", "items-easy.jsonl", "items-evalplus.jsonl", "items-exercism-rest.jsonl"]
STATE_TOKENS = 470
TASK_CHARS = 1500
STUB_CHARS = 1200
ALPHA, DELTA = 0.05, 0.05


def question(t, n=None):
    n = n or len(TIERS)
    return {f"tier_{t}": {"type": "noul", "instructions":
                          f"Will a tier-{t + 1} model (of {n}; tier {n} is the strongest) solve this coding task?"}}


def items():
    out = {}
    for f in ITEM_FILES:
        path = os.path.join(route.HOME, f)
        for it in (json.loads(l) for l in open(path)) if os.path.exists(path) else ():
            out.setdefault(it["id"], it)
    return out


def facts(item):
    read = lambda f: open(os.path.join(item["dir"], f), errors="replace").read()
    tests = "".join(read(f) for f in item["test"])
    stub = "".join(read(f) for f in item["solution"])
    return {
        "language": "python",
        "files": [{"path": f, "role": "test" if f in item["test"] else "solution", "lines": read(f).count("\n") + 1}
                  for f in (item["solution"] + item["test"])[:8]],
        "stub_functions": len(re.findall(r"^\s*def ", stub, re.M)),
        "test_functions": len(re.findall(r"^\s*def test_", tests, re.M)),
        "asserts": len(re.findall(r"\bassert", tests)),
        "test_imports": sorted(set(re.findall(r"^\s*(?:from|import) (\w+)", tests, re.M)))[:10],
    }


def state(item, tok=None):
    """The D0 state. The stub stands in for the scout turns' reads (a HumanEval task is its
    docstring). With a tokenizer, the longer of task and stub is cut until the state fits."""
    full = {"task": item["instructions"].strip(),
            "stub": "".join(open(os.path.join(item["dir"], f), errors="replace").read() for f in item["solution"]).strip()}
    s = {"decision": "D0", "step": 0, "context": "0-4k", "cache": "cold", "repo": facts(item),
         "task": full["task"][:TASK_CHARS], "stub": full["stub"][:STUB_CHARS]}
    size = lambda: len(tok(json.dumps(s, ensure_ascii=False), add_special_tokens=False)["input_ids"])
    while tok is not None and size() > STATE_TOKENS and (s["task"] or s["stub"]):
        k = "task" if len(s["task"]) >= len(s["stub"]) else "stub"
        s[k] = s[k][:max(0, len(s[k]) - 100)]
    return s


def tokenizer():
    from transformers import AutoTokenizer

    from laya.agent import _fix_tokenizer_config
    from .train import base_dir

    src = base_dir("convaiinnovations/laya")
    _fix_tokenizer_config(src)
    return AutoTokenizer.from_pretrained(os.path.join(src, "tokenizer"))


def outcomes():
    """item -> tier -> {"p", "cost", ...} for items run on every tier, less those whose stub passes."""
    return {i: row for i, row in route.table(TIERS).items() if i not in route.STUB_PASSES}


def split(item_id):
    return D.split_of(item_id, calib_share=0.2, test_share=0.2)


def build(name="route-v1"):
    tab, its, tok = outcomes(), items(), tokenizer()
    rows = []
    for i, row in sorted(tab.items()):
        st = json.dumps(state(its[i], tok), ensure_ascii=False)
        for t in range(len(TIERS)):
            p = row[t]["p"]
            q = question(t)
            rows.append({"state": st, "questions": json.dumps(q),
                         "gold": json.dumps({f"tier_{t}": {"probabilities": {"true": p, "false": 1 - p}}}),
                         "id": f"{i}-t{t}", "group": i, "split": split(i),
                         "source": "evalplus" if re.match(r"(he|mbpp)-", i) else "exercism",
                         "labeler": "outcome", "label": "solved" if p >= 0.5 else "unsolved"})
    d, manifest = D.write_dataset(rows, name, {k: v for t in range(len(TIERS)) for k, v in question(t).items()})
    by = {k: sum(r["split"] == k for r in rows) // len(TIERS) for k in ("train", "calib", "test")}
    print(f"{len(tab)} items on {len(TIERS)} tiers, {len(rows)} rows, items per split {by} -> {d}")


# ------------------------------------------------------------------ routing and the G4 report
def monotone(ps):
    return list(np.maximum.accumulate(ps))


def route_to(ps, tau):
    """The lowest tier whose probability clears tau, else the strongest."""
    return next((t for t in range(len(ps) - 1) if ps[t] >= tau), len(ps) - 1)


def harmful(row, t):
    top = len(row) - 1
    return t < top and row[t]["p"] < 0.5 and row[top]["p"] >= 0.5


def certify(scored, tab, alpha=ALPHA, delta=DELTA):
    """Fixed-sequence Learn-Then-Test (arXiv 2110.01052): walk thresholds from the most conservative
    (everything to the strongest tier, harmless by definition) down, test H0 "harmful downgrades >
    alpha" at each, and stop at the first that fails. Returns the lowest certified threshold."""
    best = INF
    for tau in sorted({p for ps in scored.values() for p in ps[:-1]}, reverse=True):
        k = sum(harmful(tab[i], route_to(ps, tau)) for i, ps in scored.items())
        if binom_cdf(k, len(scored), alpha) > delta:
            break
        best = tau
    return best


def heuristic_tier(row):
    return min(next(t for d, t in route.HEURISTIC if row[0]["difficulty"] <= d), len(row) - 1)


def route_auc(scored, tab, ids):
    """AUROC over (item, lower tier) pairs of "this tier solves it": Laya's probability against the
    heuristic's score (the tier's distance above the heuristic's pick)."""
    pairs = [(i, t) for i in ids for t in range(len(TIERS) - 1)]
    y = [tab[i][t]["p"] >= 0.5 for i, t in pairs]
    return auroc([scored[i][t] for i, t in pairs], y), auroc([t - heuristic_tier(tab[i]) for i, t in pairs], y)


def auc_delta(scored, tab, ids, n=2000, rng=None):
    rng = rng or random.Random(0)
    deltas = []
    for _ in range(n):
        a, h = route_auc(scored, tab, [rng.choice(ids) for _ in ids])
        if a is not None and h is not None:
            deltas.append(a - h)
    a, h = route_auc(scored, tab, ids)
    deltas.sort()
    return {"laya": a, "heuristic": h, "delta": None if a is None or h is None else round(a - h, 4),
            "lcb95": round(deltas[int(0.025 * len(deltas))], 4) if deltas else None}


def evaluate(scored, tab, tau):
    ids = sorted(scored)
    top = len(TIERS) - 1
    routed = {i: route_to(scored[i], tau) for i in ids}
    cost = lambda i, t: tab[i][t]["cost"]
    c_r, c_s = sum(cost(i, routed[i]) for i in ids), sum(cost(i, top) for i in ids)
    harm = sum(harmful(tab[i], routed[i]) for i in ids)
    # failure-aware bills: a failure is re-run once on the strongest tier. The tiers already run with
    # thinking off, so "strongest at low effort" is the strongest tier as run.
    bill_r = sum(cost(i, routed[i]) + (1 - tab[i][routed[i]]["p"]) * cost(i, top) for i in ids)
    bill_s = sum(cost(i, top) * (2 - tab[i][top]["p"]) for i in ids)
    quality = paired([tab[i][top]["p"] for i in ids], [tab[i][routed[i]]["p"] for i in ids])
    auc = auc_delta(scored, tab, ids)
    return {
        "metrics": {
            "cost_reduction_vs_strongest": round(1 - c_r / c_s, 4) if c_s else None,
            "quality_delta_lcb95": quality["lcb95"],
            "route_auc_minus_heuristic_lcb95": auc["lcb95"],
            "harmful_downgrade_ucb95": round(cp_upper(harm, len(ids)), 4),
            "bill_vs_low_effort_strongest": round(bill_r / bill_s, 4) if bill_s else None,
        },
        "detail": {"items": len(ids), "routed_per_tier": {TIERS[t]: sum(v == t for v in routed.values()) for t in range(len(TIERS))},
                   "harmful_downgrades": harm, "quality_delta": quality, "route_auc": auc,
                   "cost_routed": round(c_r / len(ids), 5), "cost_strongest": round(c_s / len(ids), 5)},
    }


def score(tab, checkpoint, device=None):
    """item -> per-tier probability for every calibration and test item, cached next to the checkpoint."""
    cache = os.path.join(checkpoint, "route-scores.json")
    ids = sorted(i for i in tab if split(i) != "train")
    got = json.load(open(cache)) if os.path.exists(cache) else {}
    todo = [i for i in ids if i not in got]
    if todo:
        its, tok = items(), tokenizer()
        states = {i: state(its[i], tok) for i in todo}
        pairs = [(i, t) for i in todo for t in range(len(TIERS))]
        probs = model_probs([states[i] for i, _ in pairs], checkpoint, device, [question(t) for _, t in pairs])
        for (i, t), p in zip(pairs, probs):
            got.setdefault(i, [0.0] * len(TIERS))[t] = p
        json.dump(got, open(cache, "w"))
    return {i: got[i] for i in ids}


def report(checkpoint, device=None, out="latest.json"):
    tab = outcomes()
    raw = score(tab, checkpoint, device)
    calib = [i for i in raw if split(i) == "calib"]
    test = [i for i in raw if split(i) == "test"]
    # per-tier Platt scaling on the calibration items; a tier with one class there keeps its raw scores
    platt = []
    for t in range(len(TIERS)):
        y = [float(tab[i][t]["p"] >= 0.5) for i in calib]
        platt.append(fit_platt([raw[i][t] for i in calib], y) if 0 < sum(y) < len(y) else (1.0, 0.0))
    cal = {i: monotone([recalibrate(raw[i][t], *platt[t]) for t in range(len(TIERS))]) for i in raw}
    tau = certify({i: cal[i] for i in calib}, tab)
    result = evaluate({i: cal[i] for i in test}, tab, tau)
    sub = lambda ids: {i: tab[i] for i in ids}
    rep = {
        "gate": "G4", "generated": time.strftime("%Y-%m-%d %H:%M"), "checkpoint": checkpoint,
        "decision_point": "D0 (task start), no scout turns",
        "tiers": TIERS, "priced_as": [cachesim.PROXY.get(t, t) for t in TIERS],
        "items": {"train": sum(split(i) == "train" for i in tab), "calib": len(calib), "test": len(test)},
        "threshold": None if tau == INF else round(tau, 4), "certified": tau != INF,
        "platt": [[round(a, 4), round(b, 4)] for a, b in platt],
        **result,
        "baselines_test": route.baselines(sub(test), len(TIERS)) if test else None,
        "notes": {"certified": "false means no threshold passed LTT on the calibration items: every task goes to the strongest tier",
                  "low_effort": "every tier runs with thinking off, so the strongest tier as run is the low-effort strongest",
                  "ladder": "local proxy tiers billed at real prices; G6 recalibrates on real tiers"},
        "self_test": "pass" if self_test() else "fail",
    }
    os.makedirs(os.path.join(REPORTS, "router"), exist_ok=True)
    path = os.path.join(REPORTS, "router", out)
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ("items", "threshold", "certified", "metrics")}, indent=2))
    print("->", path)


def self_test():
    row = lambda *ps: {t: {"p": p, "cost": [1.0, 4.0, 10.0][t], "difficulty": 1} for t, p in enumerate(ps)}
    assert monotone([0.2, 0.1, 0.5]) == [0.2, 0.2, 0.5]
    assert route_to([0.9, 0.95, 1.0], 0.8) == 0 and route_to([0.1, 0.85, 0.9], 0.8) == 1 and route_to([0.1, 0.2, 0.3], 0.8) == 2
    assert harmful(row(0, 0, 1), 0) and not harmful(row(0, 0, 0), 0) and not harmful(row(0, 0, 1), 2) and not harmful(row(1, 1, 1), 0)
    # 100 items that tier 0 solves exactly when Laya says so (tier 1 solves all): thresholds down to
    # 0.3 are safe, and 0.1 would send the unsolved half to tier 0
    tab = {f"i{k}": row(k % 2, 1, 1) for k in range(100)}
    scored = {f"i{k}": [0.9 if k % 2 else 0.1, 0.3, 0.99] for k in range(100)}
    assert certify(scored, tab) == 0.3
    # the same items when Laya is inverted: the first threshold already harms half, so none is certified
    inverted = {i: [1 - ps[0], ps[1], ps[2]] for i, ps in scored.items()}
    assert certify(inverted, tab) == INF
    ev = evaluate(scored, tab, 0.5)
    m = ev["metrics"]
    assert ev["detail"]["routed_per_tier"] == {TIERS[0]: 50, TIERS[1]: 0, TIERS[2]: 50}
    assert m["cost_reduction_vs_strongest"] == 0.45 and ev["detail"]["harmful_downgrades"] == 0 and m["quality_delta_lcb95"] == 0.0
    assert m["bill_vs_low_effort_strongest"] == 0.55 and ev["detail"]["route_auc"]["laya"] == 1.0
    assert ev["detail"]["route_auc"]["heuristic"] == 0.8333  # difficulty 1 puts every item at tier 0: ties
    # too few items certify nothing
    assert certify(dict(list(scored.items())[:10]), tab) == INF
    return True


def main(argv=None):
    global TIERS
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--name", default="route-v1")
    r = sub.add_parser("report"); r.add_argument("--checkpoint", required=True); r.add_argument("--device")
    r.add_argument("--out", default="latest.json", help="file under reports/router (G4 reads latest.json)")
    for x in (b, r):
        x.add_argument("--tiers", default=",".join(TIERS), help="the ladder, cheapest first (default: the local proxy ladder)")
    sub.add_parser("demo")
    a = ap.parse_args(argv)
    if a.cmd in ("build", "report"):
        TIERS = a.tiers.split(",")
    if a.cmd == "build":
        build(a.name)
    elif a.cmd == "report":
        report(a.checkpoint, a.device, a.out)
    else:
        print("router self-test", "passed" if self_test() else "FAILED")


if __name__ == "__main__":
    main()
