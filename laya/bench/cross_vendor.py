"""Failover between real vendors: Claude (Anthropic API) failing mid-task, ChatGPT (gpt-5.5) taking over.

    python -m bench.cross_vendor run [--items 3] [--binary <local olaya build>]
    python -m bench.cross_vendor report                  # failover/cross-vendor.json

Track E (gate F6) checked the mechanism with one local model behind two fake providers. What it
could not check is what differs between real vendors: Anthropic's and OpenAI's message formats,
tool-call IDs and reasoning fields, carried across a switch with the whole history. Here the
primary is `anthropic/claude-haiku-4-5`, reached through Track E's proxy in front of the real API
(raw: nothing rewritten), and the fallback is `openai/gpt-5.5` through the owner's ChatGPT sign-in,
reached directly. From Anthropic's third request (the first is the title) the proxy answers:
- overload: 529 `overloaded_error`, Anthropic's shape (two retries, then the switch);
- credit:   400 "Your credit balance is too low" (disabled at once, then the switch);
- none:     nothing (the baseline, all on Claude).

Runs are local, not in Docker: the ChatGPT credential stays in Olaya's store on this machine. The
owner's global skills are kept out of the prompt. Costs are the Anthropic steps' reported cost.
"""

import argparse
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from . import route
from .outage import codename, free_port
from .run import REPORTS

PRIMARY = "anthropic/claude-haiku-4-5"
FALLBACK = "openai/gpt-5.5"
SCENARIOS = ["none", "overload", "credit"]
BINARY = os.path.expanduser("~/Desktop/OLaya/olaya-pilot/packages/olaya/dist/olaya-darwin-arm64/bin/olaya")
OUT = os.path.join(os.path.dirname(route.RUNS), "variants", "cross-vendor")
STORE = os.path.expanduser("~/.local/share/olaya/failover/availability.json")
HERE = os.path.dirname(os.path.abspath(__file__))


def forget():
    """Clear this benchmark's two models from the shared cooldown store, so one run's failure does not
    move the next run's baseline to the fallback."""
    try:
        entries = json.load(open(STORE))
    except (OSError, ValueError):
        return
    kept = {k: v for k, v in entries.items() if k not in (PRIMARY, FALLBACK)}
    if kept != entries:
        tmp = STORE + ".tmp"
        json.dump(kept, open(tmp, "w"))
        os.replace(tmp, STORE)


def run(n_items, binary=BINARY, after=2, timeout=600):
    items = [json.loads(l) for l in open(route.ITEMS)][:n_items]
    for it in items:
        for scenario in SCENARIOS:
            out = os.path.join(OUT, scenario, it["id"])
            if os.path.exists(os.path.join(out, "0.json")):
                continue
            os.makedirs(out, exist_ok=True)
            forget()
            port = free_port()
            proxy = subprocess.Popen([sys.executable, "-m", "bench.outage", "proxy", "--port", str(port), "--scenario", scenario,
                                      "--after", str(after), "--log", os.path.join(out, "0.proxy.jsonl"),
                                      "--upstream", "https://api.anthropic.com", "--raw"], cwd=os.path.dirname(HERE))
            with tempfile.TemporaryDirectory() as work:
                for f in it["solution"] + it["test"]:
                    shutil.copy(os.path.join(it["dir"], f), work)
                task = route.task_text(it) + f"\n\nProject note: this release is codenamed {codename(it['id'])}. Mention the codename in your final message."
                config = {"provider": {"anthropic": {"options": {"baseURL": f"http://127.0.0.1:{port}/a/v1"}}},
                          "failover": {"models": [FALLBACK], "stall_timeout": 120}}
                env = {**os.environ, "OLAYA_CONFIG_CONTENT": json.dumps(config), "OLAYA_DISABLE_EXTERNAL_SKILLS": "1",
                       "OLAYA_DISABLE_CLAUDE_CODE": "1", "OLAYA_DISABLE_AUTOUPDATE": "1"}
                try:
                    for _ in range(50):
                        with socket.socket() as s:
                            if s.connect_ex(("127.0.0.1", port)) == 0:
                                break
                        time.sleep(0.2)
                    t0 = time.time()
                    timed_out = False
                    with open(os.path.join(out, "0.events.jsonl"), "w") as events:
                        try:
                            done = subprocess.run([binary, "run", "-m", PRIMARY, "--format", "json", "--dangerously-skip-permissions",
                                                   "--", task], cwd=work, env=env, stdout=events, stderr=subprocess.DEVNULL, timeout=timeout)
                            code = done.returncode
                        except subprocess.TimeoutExpired:
                            timed_out, code = True, None
                    wall = round(time.time() - t0, 1)
                    for f in it["test"]:
                        shutil.copy(os.path.join(it["dir"], f), work)  # the pristine tests
                    tests = subprocess.run([os.path.join(os.path.dirname(HERE), ".venv", "bin", "python"), "-m", "pytest", "-q"],
                                           cwd=work, capture_output=True, text=True, timeout=300)
                finally:
                    proxy.terminate()
                    proxy.wait(timeout=10)
            record = {"item": it["id"], "scenario": scenario, "passed": (not timed_out) and tests.returncode == 0,
                      "timed_out": timed_out, "exit": code, "wall_s": wall, **facts(out, it["id"])}
            json.dump(record, open(os.path.join(out, "0.json"), "w"), indent=1)
            print(json.dumps({k: record[k] for k in ("scenario", "item", "passed", "switched", "steps", "anthropic_cost", "wall_s")}), flush=True)


def facts(out, item_id):
    """From a run's events and proxy log: whether it switched, steps, the Anthropic cost, and whether
    the final answer still carries the planted codename."""
    events = [json.loads(l) for l in open(os.path.join(out, "0.events.jsonl")) if l.startswith("{")]
    log = [json.loads(l) for l in open(os.path.join(out, "0.proxy.jsonl"))] if os.path.exists(os.path.join(out, "0.proxy.jsonl")) else []
    retries = [e["status"]["message"] for e in events if e.get("type") == "retry"]
    finals = [(e.get("part") or {}).get("text", "") for e in events if e.get("type") == "text"]
    return {
        "switched": any(FALLBACK in m for m in retries),
        "faults": sum(1 for e in log if e.get("fault") is not None),
        "steps": sum(1 for e in events if e.get("type") == "step_finish"),
        "anthropic_cost": round(sum((e.get("part") or {}).get("cost") or 0 for e in events if e.get("type") == "step_finish"), 4),
        "codename_kept": codename(item_id) in (finals[-1] if finals else ""),
        "errors": [json.dumps(e.get("error"))[:200] for e in events if e.get("type") == "error"],
        "retries": retries,
    }


def report():
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "*", "*", "0.json")))]
    by = {s: [r for r in runs if r["scenario"] == s] for s in SCENARIOS}
    rep = {
        "track": "cross-vendor failover", "generated": time.strftime("%Y-%m-%d %H:%M"), "primary": PRIMARY, "fallback": FALLBACK,
        "per_scenario": {s: {"runs": len(rs), "passed": sum(r["passed"] for r in rs), "switched": sum(r["switched"] for r in rs),
                             "codename_kept": sum(r["codename_kept"] for r in rs), "stopped": sum(bool(r["errors"]) or r["timed_out"] for r in rs)}
                         for s, rs in by.items()},
        "anthropic_cost_usd": round(sum(r["anthropic_cost"] for r in runs), 4),
        "runs": runs,
    }
    os.makedirs(os.path.join(REPORTS, "failover"), exist_ok=True)
    path = os.path.join(REPORTS, "failover", "cross-vendor.json")
    json.dump(rep, open(path, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("per_scenario", "anthropic_cost_usd")}, indent=1))
    print("->", path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--items", type=int, default=3); r.add_argument("--binary", default=BINARY)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.items, a.binary)
    else:
        report()


if __name__ == "__main__":
    main()
