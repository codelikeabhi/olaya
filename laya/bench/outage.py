"""OlayaBench Track E (Outage): does a task survive its provider failing? (gate F6)

    python -m bench.outage run --binary <linux build with failover> [--items 6]
    python -m bench.outage report                      # failover/track-e.json
    python -m bench.outage demo                        # self-test (metrics and the proxy's faults)
    python -m bench.outage proxy --port 9101 --scenario quota --after 1 --log run.jsonl   # used by `run`

A fault-injecting proxy in front of the local Ollama serves two "providers", `a` and `b`, with the
same model. Olaya runs a Track C item on `a` with `b` as its fallback (`failover.models`), and
from `a`'s (after+1)-th request the proxy injects one scenario:

- quota:    429 `usage_limit_reached` with `resets_at` an hour away (Codex's shape)
- throttle: 429 `rate_limit_exceeded` with `retry-after: 3600`
- overload: 529 `overloaded_error` (Anthropic's shape)
- server:   500 on every request
- stall:    200 and SSE keep-alive comments, but no content
- drop:     the real stream, cut after a few chunks
- outage:   `a` and `b` both 503 for 60 s, then both recover
- none:     no fault (the paired baseline)

Because both providers serve the same model, a difference between a scenario and its baseline
is the failover mechanism's doing, not model quality's.

Metrics (thresholds in gates.json, F6):
- session_stops: fault runs that ended with an error exit, or timed out when their baseline did not;
- success_delta_lcb95: paired pass rate, fault minus baseline, lower 95% bound;
- duplicate_tool_calls: file changes repeated identically, and completed, after the switch, or one call executed twice
  (re-running a command such as the tests is normal work and does not count);
- lost_tool_results: tool calls completed before the switch whose results the fallback never saw;
- needles_after_switch: runs whose first request to `b` still carries the task's planted codename;
- failover_p95_s: first fault to first answer from `b`; for a stall, from when it is detectable (the
  stall limit after the first fault), with the raw stall times reported beside it.
"""

import argparse
import asyncio
import glob
import hashlib
import json
import os
import random
import re
import socket
import subprocess
import sys
import time

from . import route
from .retention_ab import COMPACTION, LIMIT
from .run import REPORTS

SCENARIOS = ["none", "quota", "throttle", "overload", "server", "stall", "drop", "outage"]
UPSTREAM = "http://localhost:11434"
MODEL = "qwen3-8b-16k"
OUTAGE_S = 60
# Earlier runs are kept: "outage-*" had an 8k usable window, which the system prompt nearly fills
# (every scenario compacted over and over); "outage2-*" sampled, so a fault run and its baseline
# diverged by chance; "outage3-*" decoded greedily, and qwen3-8b looped on its own output (178
# requests in 11 minutes); "outage4-*" ran a build without the loop guard, where qwen3-8b called
# `task` with task_id "1" hundreds of times; "outage5-*" still drifted between runs (random tool-call
# IDs, test timings); "outage6-*" ran build f85db16, whose loop guard missed a task re-delegated 207
# times. "outage7-*": the G9 pilot's window, a fixed seed and normalised requests (SEED, normalise),
# on build 944b6b3 (both review rounds, the announced-step nudge); its 30 s stall limit stalled
# healthy runs whose model was writing a long tool call; "outage8-*" (bd6155e) still retried a stall
# once, another 240 s. "outage9-*": a 240 s stall limit, on build 0b0f524, where a stall switches at
# once and cools for a minute.
PREFIX = "outage9-"
STALL_S = 240
# Forced on every chat request: sampling as usual, but the same request gets the same answer, so a
# fault run and its baseline stay identical up to the fault and a difference after it is the
# failover's doing.
SEED = {"seed": 7}
# Two runs of the same item would still drift apart on things that are not the model's doing: Ollama
# makes up random tool-call IDs, test runners print their duration, and the system prompt names
# the provider. The proxy numbers tool calls in order, zeroes durations, and shows `b` the same
# provider name as `a`, so a perfect failover reproduces its baseline exactly.
DURATION = re.compile(r"\b(in )\d+\.\d+(s\b)")


def normalise(body, provider):
    text = DURATION.sub(r"\g<1>0.00\g<2>", json.dumps({**json.loads(body), **SEED}))
    return (text.replace("provb/", "prova/") if provider == "b" else text).encode()


def renumber(line, ids):
    """One SSE line from upstream, with its tool-call IDs replaced by call_1, call_2, ... in order."""
    if not line.startswith(b"data: {"):
        return line
    event = json.loads(line[6:])
    for choice in event.get("choices") or []:
        for call in (choice.get("delta") or {}).get("tool_calls") or []:
            if call.get("id"):
                call["id"] = ids.setdefault(call["id"], f"call_{len(ids) + 1}")
    return b"data: " + json.dumps(event).encode() + b"\n"
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ the fault-injecting proxy
def fault(scenario, provider, n_a, after, started_fault):
    """The fault for this request, or None to pass it through. `started_fault` is when the
    outage began (for the time-bounded scenario)."""
    if scenario == "outage":
        if n_a <= after and provider == "a":
            return None
        if started_fault is not None and time.time() - started_fault > OUTAGE_S:
            return None
        return (503, {"error": {"message": "Service unavailable", "type": "server_error"}}, {})
    if provider != "a" or n_a <= after or scenario == "none":
        return None
    if scenario == "quota":
        return (429, {"error": {"type": "usage_limit_reached", "message": "You've hit your usage limit",
                                "resets_at": int(time.time()) + 3600, "limit_window_minutes": 300}}, {})
    if scenario == "throttle":
        return (429, {"error": {"code": "rate_limit_exceeded", "message": "Rate limit reached"}}, {"retry-after": "3600"})
    if scenario == "overload":
        return (529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}, {})
    if scenario == "server":
        return (500, {"error": {"message": "internal server error"}}, {})
    if scenario == "credit":  # Anthropic's out-of-credit answer
        return (400, {"type": "error", "error": {"type": "invalid_request_error", "message":
                "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."}}, {})
    return scenario  # "stall" and "drop" are streamed faults


async def serve(port, scenario, after, log_path, upstream=UPSTREAM, raw=False):
    """`upstream`: where requests go (Ollama by default). `raw`: forward bodies and streams as they
    are, for a real provider: the seed and ID rewriting are for making local runs repeat."""
    from aiohttp import ClientSession, ClientTimeout, web

    state = {"a": 0, "fault_at": None, "ids": {}}
    log = open(log_path, "a")

    def record(**entry):
        log.write(json.dumps({"t": time.time(), **entry}) + "\n")
        log.flush()

    async def handle(request):
        provider = request.match_info["provider"]
        path = request.match_info["path"]
        body = await request.read()
        chat = request.method == "POST" and (path.endswith("chat/completions") or path.endswith("messages"))
        if chat and not raw:
            body = normalise(body, provider)
        if chat and provider == "a":
            state["a"] += 1
        f = fault(scenario, provider, state["a"], after, state["fault_at"]) if chat else None
        if f is not None and state["fault_at"] is None:
            state["fault_at"] = time.time()
        if chat:
            record(provider=provider, n=state["a"], fault=f if isinstance(f, str) else (f[0] if f else None),
                   **({"body": json.loads(body)} if provider == "b" else {}))
        if isinstance(f, tuple):
            status, payload, headers = f
            return web.json_response(payload, status=status, headers=headers)
        if f == "stall":
            resp = web.StreamResponse(status=200, headers={"content-type": "text/event-stream"})
            await resp.prepare(request)
            try:
                while True:
                    await resp.write(b": keep-alive\n\n")
                    await asyncio.sleep(5)
            except (ConnectionResetError, asyncio.CancelledError):
                return resp
        async with ClientSession(timeout=ClientTimeout(total=1200)) as session:
            headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
            async with session.request(request.method, f"{upstream}/{path}", data=body, headers=headers) as up:
                resp = web.StreamResponse(status=up.status, headers={"content-type": up.headers.get("content-type", "application/json")})
                await resp.prepare(request)
                chunks = 0
                async for line in up.content:
                    await resp.write(renumber(line, state["ids"]) if chat and not raw else line)
                    chunks += 1
                    if f == "drop" and chunks >= 3:
                        request.transport.close()  # cut the stream mid-answer
                        return resp
                if chat:
                    record(provider=provider, n=state["a"], answered=up.status)
                await resp.write_eof()
                return resp

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_route("*", "/{provider:a|b}/{path:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", port).start()
    while True:
        await asyncio.sleep(3600)


# ------------------------------------------------------------------ runs
def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def codename(item_id):
    """A planted fact the fallback must receive; stable across processes (unlike hash())."""
    return "ZX-" + hashlib.sha1(f"{item_id}olaya".encode()).hexdigest()[:6]


def variant(scenario, binary, port):
    base = f"http://host.docker.internal:{port}"
    model = {"name": MODEL, "tools": True, "options": {"reasoningEffort": "none"}, **LIMIT}
    return {
        "name": f"{PREFIX}{scenario}", "binary": binary, "cli_model": f"prova/{MODEL}",
        "config": {
            "provider": {
                "prova": {"npm": "@ai-sdk/openai-compatible", "name": "Provider A", "options": {"baseURL": f"{base}/a/v1"}, "models": {MODEL: model}},
                "provb": {"npm": "@ai-sdk/openai-compatible", "name": "Provider B", "options": {"baseURL": f"{base}/b/v1"}, "models": {MODEL: model}},
            },
            # Ollama sends a tool call only when it is complete: a 5,000-token output cap at ~28
            # tokens a second is ~180 s of silence, plus prompt processing. 30 s stalled healthy runs.
            "failover": {"models": [f"provb/{MODEL}"], "stall_timeout": STALL_S}, **COMPACTION,
        },
        "timeout": 1500,
    }


def run(binary, n_items, after=3, scenarios=SCENARIOS, seed=0):
    import random as _r
    items = [json.loads(l) for l in open(route.ITEMS)]
    solved = {r["item"] for f in glob.glob(os.path.join(route.RUNS, MODEL, "*", "0.json")) if (r := json.load(open(f)))["passed"]}
    items = [it for it in items if it["id"] in solved]
    _r.Random(seed).shuffle(items)
    for it in items[:n_items]:
        it = dict(it, instructions=it["instructions"] + f"\n\nProject note: this release is codenamed {codename(it['id'])}. "
                                                         f"Mention the codename in your final message.")
        for scenario in scenarios:
            port = free_port()
            out_dir = os.path.join(os.path.dirname(route.RUNS), "variants", f"{PREFIX}{scenario}", f"prova-{MODEL}", it["id"])
            os.makedirs(out_dir, exist_ok=True)
            log = os.path.join(out_dir, "0.proxy.jsonl")
            if os.path.exists(os.path.join(out_dir, "0.json")):
                continue
            open(log, "w").close()
            proxy = subprocess.Popen([sys.executable, "-m", "bench.outage", "proxy", "--port", str(port), "--scenario", scenario,
                                      "--after", str(after), "--log", log], cwd=os.path.dirname(HERE))
            try:
                for _ in range(50):
                    with socket.socket() as s:
                        if s.connect_ex(("127.0.0.1", port)) == 0:
                            break
                    time.sleep(0.2)
                r = route.run_one(it, f"prova-{MODEL}", 0, variant(scenario, binary, port))
            finally:
                proxy.terminate()
                proxy.wait(timeout=10)
            print(f"{scenario:9} {it['id']:28} passed={r['passed']} exit={r['olaya_exit']} steps={len(r['calls'])} {r['wall_s']}s", flush=True)


# ------------------------------------------------------------------ metrics
def analyse(record, proxy_log, events, needle):
    """One run's failover facts: whether it switched, how fast, what the fallback saw."""
    faults = [e for e in proxy_log if e.get("fault") is not None]
    b_first = next((e for e in proxy_log if e.get("provider") == "b" and "body" in e), None)
    b_answer = next((e for e in proxy_log if e.get("provider") == "b" and e.get("answered") == 200), None)
    switched = b_first is not None
    uses = [e["part"] for e in events if e.get("type") == "tool_use"]
    t_switch = b_first["t"] * 1000 if switched else None
    before = [u for u in uses if t_switch and ((u.get("state") or {}).get("time") or {}).get("end", 0) < t_switch
              and (u.get("state") or {}).get("status") == "completed"]
    after = [u for u in uses if t_switch and ((u.get("state") or {}).get("time") or {}).get("start", 0) >= t_switch]
    key = lambda u: (u.get("tool"), json.dumps((u.get("state") or {}).get("input"), sort_keys=True))
    # A file change issued again after the switch means the fallback didn't see it happen; the same
    # call executed twice would be the harness's own doing. Re-running a command (tests, ls) is
    # normal work, so shell calls don't count.
    side = {"edit", "write", "apply_patch", "multiedit"}
    ids = [u.get("callID") for u in uses if (u.get("state") or {}).get("status") == "completed" and u.get("callID")]
    # A repeat that failed changed nothing (typically "oldString not found": the change is already
    # there), so only completed repeats count.
    done = lambda u: (u.get("state") or {}).get("status") == "completed"
    duplicates = sum(1 for u in after if u.get("tool") in side and done(u) and key(u) in {key(b) for b in before}) + len(ids) - len(set(ids))
    body = json.dumps(b_first["body"]) if switched else ""
    lost = sum(1 for u in before if u.get("callID") and u["callID"] not in body) if switched else 0
    return {
        "switched": switched,
        "faulted": bool(faults),
        # an error exit; a timeout counts only when the baseline finished in time (report() pairs them)
        "stopped": record.get("olaya_exit") not in ("0", None),
        "duplicates": duplicates,
        "lost": lost,
        "needle": (needle in body) if switched else None,
        "failover_s": round(b_answer["t"] - faults[0]["t"], 1) if (faults and b_answer) else None,
    }


def lcb_paired(diffs, rng=None, n=2000):
    rng = rng or random.Random(0)
    if not diffs:
        return None
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return round(means[int(0.025 * n)], 4)


def report():
    base = os.path.join(os.path.dirname(route.RUNS), "variants")
    runs = {}
    for f in glob.glob(os.path.join(base, f"{PREFIX}*", f"prova-{MODEL}", "*", "0.json")):
        scenario = f.split(os.sep)[-4].removeprefix(PREFIX)
        d = os.path.dirname(f)
        rec = json.load(open(f))
        read = lambda name: [json.loads(l) for l in open(os.path.join(d, name))] if os.path.exists(os.path.join(d, name)) else []
        runs[(scenario, rec["item"])] = (rec, analyse(rec, read("0.proxy.jsonl"), read("0.events.jsonl"), codename(rec["item"])))
    faults = [(s, i) for (s, i) in runs if s != "none" and ("none", i) in runs]
    # A timeout is the failover's doing only if the same item's baseline didn't time out as well:
    # a slow or looping model times out with or without a fault.
    for s, i in faults:
        rec, facts = runs[(s, i)]
        facts["stopped"] = facts["stopped"] or (rec["timed_out"] and not runs[("none", i)][0]["timed_out"])
    diffs = [float(runs[(s, i)][0]["passed"]) - float(runs[("none", i)][0]["passed"]) for s, i in faults]
    facts = [runs[k][1] for k in faults]
    switched = [f for f in facts if f["switched"]]
    # A stall is detectable only once the stall limit has passed, and that window is configuration,
    # not failover: stall runs count from then. (F1 fixed 180 s assuming a 30 s limit; a local model's
    # long tool call arrives all at once, so the limit is 240 s. The raw stall times are reported.)
    detect = lambda k: STALL_S if k[0] == "stall" else 0
    times = sorted(runs[k][1]["failover_s"] - detect(k) for k in faults if runs[k][1]["failover_s"] is not None)
    stall_raw = sorted(runs[k][1]["failover_s"] for k in faults if k[0] == "stall" and runs[k][1]["failover_s"] is not None)
    per = {s: {"runs": sum(1 for k in faults if k[0] == s),
               "passed": sum(runs[k][0]["passed"] for k in faults if k[0] == s),
               "switched": sum(runs[k][1]["switched"] for k in faults if k[0] == s),
               "stopped": sum(runs[k][1]["stopped"] for k in faults if k[0] == s)} for s in SCENARIOS[1:]}
    rep = {
        "track": "E (Outage)", "generated": time.strftime("%Y-%m-%d %H:%M"), "model": MODEL,
        "providers": "two proxies in front of the same local model", "pairs": len(faults),
        "baseline_passed": sum(runs[k][0]["passed"] for k in runs if k[0] == "none"),
        "self_test": "pass" if self_test() else "fail",
        "baselines_ok": bool(faults) and all(("none", i) in runs for _, i in faults),
        "per_scenario": per,
        "metrics": {
            "session_stops": sum(f["stopped"] for f in facts),
            "success_delta_lcb95": lcb_paired(diffs),
            "duplicate_tool_calls": sum(f["duplicates"] for f in facts),
            "lost_tool_results": sum(f["lost"] for f in facts),
            "needles_after_switch": round(sum(bool(f["needle"]) for f in switched) / len(switched), 4) if switched else None,
            "failover_p95_s": round(times[int(0.95 * (len(times) - 1))], 1) if times else None,
            "stall_failover_s_raw": stall_raw,
            "stall_limit_s": STALL_S,
        },
    }
    os.makedirs(os.path.join(REPORTS, "failover"), exist_ok=True)
    path = os.path.join(REPORTS, "failover", "track-e.json")
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print("->", path)


def self_test():
    # faults: `a` from its second request; `b` untouched except in an outage
    assert fault("quota", "a", 1, 1, None) is None and fault("quota", "a", 2, 1, None)[0] == 429
    assert fault("quota", "b", 5, 1, None) is None and fault("none", "a", 9, 1, None) is None
    assert fault("throttle", "a", 2, 1, None)[2] == {"retry-after": "3600"}
    assert fault("stall", "a", 2, 1, None) == "stall" and fault("drop", "a", 2, 1, None) == "drop"
    assert fault("outage", "b", 1, 1, time.time())[0] == 503 and fault("outage", "b", 1, 1, time.time() - OUTAGE_S - 1) is None
    # analysis: a completed edit before the switch, repeated after it, and a result the fallback saw
    t0 = 1_000.0
    log = [{"t": t0, "provider": "a", "fault": None}, {"t": t0 + 5, "provider": "a", "fault": 429},
           {"t": t0 + 6, "provider": "b", "body": {"messages": ["call_1 ZX-abc"]}}, {"t": t0 + 9, "provider": "b", "answered": 200}]
    edit = {"tool": "edit", "callID": "call_1", "state": {"status": "completed", "input": {"f": 1}, "time": {"start": 1_001_000, "end": 1_002_000}}}
    again = {"tool": "edit", "callID": "call_2", "state": {"status": "completed", "input": {"f": 1}, "time": {"start": 1_007_000, "end": 1_007_500}}}
    facts = analyse({"timed_out": False, "olaya_exit": "0"}, log, [{"type": "tool_use", "part": edit}, {"type": "tool_use", "part": again}], "ZX-abc")
    assert facts == {"switched": True, "faulted": True, "stopped": False, "duplicates": 1, "lost": 0, "needle": True, "failover_s": 4.0}, facts
    test = {"tool": "bash", "callID": "call_3", "state": {"status": "completed", "input": {"c": "pytest"}, "time": {"start": 1_001_000, "end": 1_002_000}}}
    rerun = {"tool": "bash", "callID": "call_4", "state": {"status": "completed", "input": {"c": "pytest"}, "time": {"start": 1_007_000, "end": 1_007_500}}}
    failed = {"tool": "edit", "callID": "call_5", "state": {"status": "error", "input": {"f": 1}, "time": {"start": 1_007_000, "end": 1_007_500}}}
    assert analyse({"timed_out": False, "olaya_exit": "0"}, log, [{"type": "tool_use", "part": edit}, {"type": "tool_use", "part": failed}], "ZX-abc")["duplicates"] == 0
    assert analyse({"timed_out": False, "olaya_exit": "0"}, log, [{"type": "tool_use", "part": test}, {"type": "tool_use", "part": rerun}], "ZX-abc")["duplicates"] == 0
    assert lcb_paired([0.0, 0.0, 0.0]) == 0.0 and lcb_paired([]) is None
    # normalisation: durations zeroed, `b` named like `a`, tool calls numbered in order
    body = json.loads(normalise(json.dumps({"m": "12 failed in 0.03s; model ID is provb/x"}), "b"))
    assert body == {"m": "12 failed in 0.00s; model ID is prova/x", "seed": 7}, body
    ids = {}
    line = b'data: {"choices":[{"delta":{"tool_calls":[{"id":"call_zq9","index":0}]}}]}\n'
    assert b'"call_1"' in renumber(line, ids) and b'"call_1"' in renumber(line, ids)
    assert b'"call_2"' in renumber(line.replace(b"zq9", b"k2"), ids) and renumber(b": keep-alive\n", ids) == b": keep-alive\n"
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("proxy"); p.add_argument("--port", type=int, required=True); p.add_argument("--scenario", required=True)
    p.add_argument("--after", type=int, default=1); p.add_argument("--log", required=True)
    p.add_argument("--upstream", default=UPSTREAM); p.add_argument("--raw", action="store_true")
    r = sub.add_parser("run"); r.add_argument("--binary", required=True); r.add_argument("--items", type=int, default=6)
    r.add_argument("--scenarios", default=",".join(SCENARIOS))
    sub.add_parser("report")
    sub.add_parser("demo")
    a = ap.parse_args(argv)
    if a.cmd == "proxy":
        asyncio.run(serve(a.port, a.scenario, a.after, a.log, a.upstream, a.raw))
    elif a.cmd == "run":
        run(a.binary, a.items, scenarios=a.scenarios.split(","))
    elif a.cmd == "report":
        report()
    else:
        print("outage self-test", "passed" if self_test() else "FAILED")


if __name__ == "__main__":
    main()
