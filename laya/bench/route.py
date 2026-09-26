"""OlayaBench Track C (Route): which model tier each coding task needs, and what it costs.

    python -m bench.route build --exercism <exercism/python checkout> --n 32      # choose items
    python -m bench.route run --tiers qwen3-0.6b-16k,qwen3-4b-16k,qwen3-8b-16k,qwen3-14b-16k --k 2
    python -m bench.route report                                                   # track-c/latest.json
    python -m bench.route demo                                                     # metrics self-test

Items are Exercism Python exercises (MIT), sampled across Exercism's own difficulty ratings. Each
run puts one exercise (stub, tests and instructions, never the reference solution) in a sandbox
container, runs `olaya run` on one local model, and grades with pytest against a pristine copy of
the tests. Local models are free, so every run is billed by replaying its token events through the
cache simulator at the real tier that model stands in for (bench.cachesim.PROXY).
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

from . import cachesim
from .run import REPORTS

HOME = os.path.join(os.path.dirname(REPORTS), "track-c")  # ~/.local/share/olaya/laya/track-c
ITEMS = os.path.join(HOME, "items.jsonl")
RUNS = os.path.join(HOME, "runs")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE = "olaya-trackc:py312"
TIMEOUT_S = 600

# The heuristic baseline, fixed before any outcome was seen: Exercism difficulty -> tier index.
HEURISTIC = [(2, 0), (4, 1), (6, 2), (10, 3)]


# ------------------------------------------------------------------ items
def build(exercism, n, seed=0):
    config = json.load(open(os.path.join(exercism, "config.json")))
    pool = [e for e in config["exercises"]["practice"] if e.get("status", "active") not in ("deprecated", "wip")]
    bands = [(1, 2), (3, 4), (5, 5), (6, 10)]
    per = [n * 1 // 4, n * 3 // 8, n * 1 // 8]
    per.append(n - sum(per))
    rng = random.Random(seed)
    items = []
    for (lo, hi), k in zip(bands, per):
        band = sorted((e for e in pool if lo <= e.get("difficulty", 1) <= hi), key=lambda e: e["slug"])
        for e in rng.sample(band, min(k, len(band))):
            d = os.path.join(exercism, "exercises", "practice", e["slug"])
            meta = json.load(open(os.path.join(d, ".meta", "config.json")))
            docs = "".join(open(os.path.join(d, ".docs", f)).read() + "\n" for f in ("instructions.md", "instructions.append.md")
                           if os.path.exists(os.path.join(d, ".docs", f)))
            items.append({"id": e["slug"], "difficulty": e.get("difficulty", 1), "dir": d,
                          "solution": meta["files"]["solution"], "test": meta["files"]["test"], "instructions": docs})
    os.makedirs(HOME, exist_ok=True)
    with open(ITEMS, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"{len(items)} items -> {ITEMS}")


def task_text(item):
    return (f"Implement the exercise in {', '.join(item['solution'])} so that every test in "
            f"{', '.join(item['test'])} passes. Run the tests with `python -m pytest -q` to check your work. "
            f"Do not modify the test files.\n\n{item['instructions']}")


# ------------------------------------------------------------------ runs
def binary():
    arch = "arm64" if os.uname().machine in ("arm64", "aarch64") else "x64"
    path = os.path.join(REPO, "packages", "olaya", "dist", f"olaya-linux-{arch}", "bin", "olaya")
    if not os.path.exists(path):
        sys.exit(f"no Linux build at {path}; run packages/olaya/script/build.ts")
    return path


def olaya_config(model):
    return json.dumps({
        "provider": {"ollama": {"npm": "@ai-sdk/openai-compatible", "name": "Ollama",
                                "options": {"baseURL": "http://host.docker.internal:11434/v1"},
                                # qwen3 thinks by default and spends a 16k window on thinking before it edits
                                # anything; the tiers are compared with thinking off
                                "models": {model: {"name": model, "tools": True,
                                                   "options": {"reasoningEffort": "none"}}}}},
    })


def run_one(item, model, k):
    out_dir = os.path.join(RUNS, model, item["id"])
    out = os.path.join(out_dir, f"{k}.json")
    if os.path.exists(out):
        return json.load(open(out))
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work, pristine = os.path.join(tmp, "work"), os.path.join(tmp, "pristine")
        os.makedirs(work); os.makedirs(pristine)
        for f in item["solution"] + item["test"]:
            shutil.copy(os.path.join(item["dir"], f), os.path.join(work, f))
        for f in item["test"]:
            shutil.copy(os.path.join(item["dir"], f), os.path.join(pristine, f))
        os.chmod(work, 0o777)
        name = "trackc-" + hashlib.sha1(f"{model}{item['id']}{k}{time.time()}".encode()).hexdigest()[:10]
        script = (
            f"olaya --model ollama/{model} run --format json --dangerously-skip-permissions -- \"$TASK\" "
            "> /tmp/events.jsonl 2>/tmp/stderr; echo $? > /tmp/olaya-exit; "
            "cp /pristine/* /work/; python -m pytest -q > /tmp/pytest.txt 2>&1; echo $? > /tmp/pytest-exit; "
            "cp /tmp/events.jsonl /tmp/stderr /tmp/olaya-exit /tmp/pytest.txt /tmp/pytest-exit /work/ 2>/dev/null; true"
        )
        t0 = time.time()
        cmd = ["docker", "run", "--rm", "--name", name, "--add-host", "host.docker.internal:host-gateway",
               "-v", f"{work}:/work", "-v", f"{pristine}:/pristine:ro", "-v", f"{binary()}:/usr/local/bin/olaya:ro",
               "-e", f"OLAYA_CONFIG_CONTENT={olaya_config(model)}", "-e", f"TASK={task_text(item)}",
               "-e", "OLAYA_DISABLE_AUTOUPDATE=1", IMAGE, "sh", "-c", script]
        timed_out = False
        try:
            subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["docker", "kill", name], capture_output=True)
        read = lambda f, d="": open(os.path.join(work, f)).read() if os.path.exists(os.path.join(work, f)) else d
        events = [json.loads(l) for l in read("events.jsonl").splitlines() if l.startswith("{")]
        record = {
            "item": item["id"], "difficulty": item["difficulty"], "model": model, "k": k,
            "passed": (not timed_out) and read("pytest-exit").strip() == "0",
            "timed_out": timed_out, "olaya_exit": read("olaya-exit").strip() or None,
            "wall_s": round(time.time() - t0, 1), "calls": calls_from(events, model),
            "pytest_tail": read("pytest.txt")[-400:],
        }
    json.dump(record, open(out, "w"))
    return record


def calls_from(events, model):
    """One cache-simulator call per model step, from the step_finish events of `olaya run --format json`."""
    calls = []
    for e in events:
        if e.get("type") != "step_finish":
            continue
        tok = (e.get("part") or {}).get("tokens") or {}
        cache = tok.get("cache") or {}
        prompt = (tok.get("input") or 0) + (cache.get("read") or 0) + (cache.get("write") or 0)
        calls.append({"model": model, "t": e.get("timestamp", 0) / 1000, "prompt": prompt,
                      "output": (tok.get("output") or 0) + (tok.get("reasoning") or 0)})
    return calls


def run(tiers, k, only=None):
    items = [json.loads(l) for l in open(ITEMS)]
    if only:
        items = [it for it in items if it["id"] in only]
    for model in tiers:            # one tier at a time keeps one model resident
        for rep in range(k):
            for it in items:
                r = run_one(it, model, rep)
                print(f"{model:16} {it['id']:28} k={rep} passed={r['passed']} steps={len(r['calls'])} {r['wall_s']}s", flush=True)


# ------------------------------------------------------------------ metrics
def table(tiers):
    """item -> tier index -> {"p": success rate over k, "cost": mean $, "runs": n}."""
    out = {}
    for ti, model in enumerate(tiers):
        base = os.path.join(RUNS, model)
        for item in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            runs = [json.load(open(os.path.join(base, item, f))) for f in sorted(os.listdir(os.path.join(base, item)))]
            costs = [cachesim.bill([cachesim.Call(**c) for c in r["calls"]]).total for r in runs]
            out.setdefault(item, {})[ti] = {"p": sum(r["passed"] for r in runs) / len(runs),
                                            "cost": sum(costs) / len(costs), "runs": len(runs),
                                            "difficulty": runs[0]["difficulty"]}
    return {i: t for i, t in out.items() if len(t) == len(tiers)}


def policy(tab, choose):
    """Mean quality and cost of a per-item tier choice."""
    q = c = 0.0
    for item, row in tab.items():
        t = choose(item, row)
        q += row[t]["p"]; c += row[t]["cost"]
    n = max(1, len(tab))
    return {"quality": q / n, "cost": c / n}


def cascade(tab, tiers_n):
    """Try the cheapest tier first; escalate on failure. Pays for every attempt."""
    q = c = 0.0
    for row in tab.values():
        fail = 1.0
        for t in range(tiers_n):
            c += fail * row[t]["cost"]
            q += fail * row[t]["p"]
            fail *= 1 - row[t]["p"]
    n = max(1, len(tab))
    return {"quality": q / n, "cost": c / n}


def baselines(tab, tiers_n):
    top = tiers_n - 1
    heuristic = lambda _i, row: next(t for d, t in HEURISTIC if row[0]["difficulty"] <= d)
    def oracle(_i, row):
        ok = [t for t in range(tiers_n) if row[t]["p"] >= 0.5]
        return min(ok, key=lambda t: row[t]["cost"]) if ok else top
    b = {
        "always_strongest": policy(tab, lambda _i, _r: top),
        "always_cheapest": policy(tab, lambda _i, _r: 0),
        "heuristic_difficulty": policy(tab, lambda i, r: min(heuristic(i, r), top)),
        "cascade": cascade(tab, tiers_n),
        "random_tier": {k: sum(policy(tab, lambda _i, _r, t=t: t)[k] for t in range(tiers_n)) / tiers_n for k in ("quality", "cost")},
        "oracle": policy(tab, oracle),
    }
    strongest = b["always_strongest"]["cost"] or 1e-12
    for v in b.values():
        v["cost_vs_strongest"] = v["cost"] / strongest
    return b


def report(tiers):
    tab = table(tiers)
    b = baselines(tab, len(tiers))
    ok = b["oracle"]["quality"] >= b["heuristic_difficulty"]["quality"] - 1e-9 >= b["always_cheapest"]["quality"] - 2e-9
    per_tier = {tiers[t]: {"quality": sum(r[t]["p"] for r in tab.values()) / max(1, len(tab)),
                           "cost": sum(r[t]["cost"] for r in tab.values()) / max(1, len(tab)),
                           "priced_as": cachesim.PROXY.get(tiers[t], tiers[t])} for t in range(len(tiers))}
    rep = {"track": "C (Route)", "generated": time.strftime("%Y-%m-%d %H:%M"), "items": len(tab), "tiers": tiers,
           "tier_source": "local proxy ladder, billed at real tier prices", "self_test": "pass" if self_test() else "fail",
           "baselines_ok": bool(ok and len(tab) > 0), "per_tier": per_tier, "baselines": b}
    os.makedirs(os.path.join(REPORTS, "track-c"), exist_ok=True)
    path = os.path.join(REPORTS, "track-c", "latest.json")
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print("->", path)


def self_test():
    # 3 items x 2 tiers with known answers
    tab = {"a": {0: {"p": 1.0, "cost": 1.0, "difficulty": 1}, 1: {"p": 1.0, "cost": 10.0, "difficulty": 1}},
           "b": {0: {"p": 0.0, "cost": 1.0, "difficulty": 5}, 1: {"p": 1.0, "cost": 10.0, "difficulty": 5}},
           "c": {0: {"p": 0.5, "cost": 1.0, "difficulty": 9}, 1: {"p": 0.5, "cost": 10.0, "difficulty": 9}}}
    b = baselines(tab, 2)
    assert abs(b["always_strongest"]["quality"] - 2.5 / 3) < 1e-9 and abs(b["always_strongest"]["cost"] - 10) < 1e-9
    assert abs(b["always_cheapest"]["quality"] - 1.5 / 3) < 1e-9
    # oracle: a -> 0 (cost 1), b -> 1 (10), c -> 0 (p 0.5 >= 0.5, cheaper): quality (1+1+0.5)/3, cost 12/3
    assert abs(b["oracle"]["quality"] - 2.5 / 3) < 1e-9 and abs(b["oracle"]["cost"] - 4) < 1e-9
    # cascade on b: pay 1 (fail), then 10 (pass) = 11; on c: 1 + 0.5*10 = 6; on a: 1
    assert abs(b["cascade"]["cost"] - (1 + 11 + 6) / 3) < 1e-9
    assert abs(b["cascade"]["quality"] - (1 + 1 + 0.75) / 3) < 1e-9
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--exercism", required=True); b.add_argument("--n", type=int, default=32)
    r = sub.add_parser("run"); r.add_argument("--tiers", required=True); r.add_argument("--k", type=int, default=2); r.add_argument("--only")
    p = sub.add_parser("report"); p.add_argument("--tiers", default="qwen3-0.6b-16k,qwen3-4b-16k,qwen3-8b-16k,qwen3-14b-16k")
    sub.add_parser("demo")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.exercism, a.n)
    elif a.cmd == "run":
        run(a.tiers.split(","), a.k, set(a.only.split(",")) if a.only else None)
    elif a.cmd == "report":
        report(a.tiers.split(","))
    else:
        print("route self-test", "passed" if self_test() else "FAILED")


if __name__ == "__main__":
    main()
