"""Gate checks for Laya's three roles: permissions, model routing and context retention.

    python -m bench.gates G0 --docs ~/Desktop/OLaya/docs         # research memos
    python -m bench.gates G1 --workspace ~/Desktop/OLaya         # OpenSpec designs + thresholds
    python -m bench.gates G2                                     # harness seams (runs bun tests)
    python -m bench.gates G4                                     # reads the gate's report
    python -m bench.gates demo                                   # self-test

A gate prints one line per check and exits 1 if any check fails. Its requirements live in
gates.json; thresholds are null until G1 fixes them, and a gate with null thresholds fails with
"thresholds not set" rather than passing by default.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

from .run import REPORTS

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SPEC = json.load(open(os.path.join(HERE, "gates.json")))


class Result:
    def __init__(self):
        self.lines = []
        self.ok = True

    def check(self, passed, what):
        self.lines.append(("PASS" if passed else "FAIL") + "  " + what)
        self.ok = self.ok and bool(passed)
        return passed


# ------------------------------------------------------------------ G0: research
def sections(text):
    return {m.group(1).strip() for m in re.finditer(r"^## (.+)$", text, re.M)}


def source_lines(text):
    m = re.search(r"^## Sources\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return [l for l in (m.group(1).splitlines() if m else []) if "http" in l]


def arxiv_ids(text):
    return set(re.findall(r"\b(\d{4}\.\d{4,5})\b", text))


def gate_g0(res, docs, spec=SPEC["G0"]):
    papers = os.path.join(docs, "research", "papers")
    on_disk = {f.split("-")[0] for f in os.listdir(papers)} if os.path.isdir(papers) else set()
    for rel in spec["memos"]:
        path = os.path.join(docs, rel)
        if not res.check(os.path.exists(path), f"{rel} exists"):
            continue
        text = open(path).read()
        missing = [s for s in spec["sections"] if s not in sections(text)]
        res.check(not missing, f"{rel}: required sections" + (f" (missing: {', '.join(missing)})" if missing else ""))
        n = len(source_lines(text))
        res.check(n >= spec["min_sources"], f"{rel}: {n} sources (need {spec['min_sources']})")
        cited = arxiv_ids(text) & on_disk
        res.check(len(cited) >= spec["min_pdfs_per_memo"], f"{rel}: {len(cited)} cited papers saved (need {spec['min_pdfs_per_memo']})")


# ------------------------------------------------------------------ G1: design
def gate_g1(res, workspace, spec=SPEC["G1"]):
    for change in spec["changes"]:
        root = os.path.join(workspace, "openspec", "changes", change)
        if not res.check(os.path.isdir(root), f"openspec change {change} exists"):
            continue
        p = subprocess.run(["openspec", "validate", change, "--strict"], cwd=workspace, capture_output=True, text=True)
        res.check(p.returncode == 0, f"{change}: openspec validate --strict" + ("" if p.returncode == 0 else f"\n      {p.stdout.strip() or p.stderr.strip()}"))
        design = os.path.join(root, "design.md")
        res.check(os.path.exists(design) and "## Gate tests" in open(design).read(), f"{change}: design.md has a '## Gate tests' section")
    for gate in spec["gates_needing_thresholds"]:
        res.check(SPEC[gate].get("thresholds") is not None, f"{gate}: thresholds fixed in gates.json")


# ------------------------------------------------------------------ G2+: tests and reports
def run_tests(res, spec):
    for pkg, files in spec.get("tests", {}).items():
        cwd = os.path.join(REPO, pkg)
        present = [f for f in files if os.path.exists(os.path.join(cwd, f))]
        for f in sorted(set(files) - set(present)):
            res.check(False, f"{pkg}/{f} exists")
        if present:
            p = subprocess.run(["bun", "test", *present], cwd=cwd, capture_output=True, text=True)
            tail = (p.stdout + p.stderr).strip().splitlines()[-4:]
            res.check(p.returncode == 0, f"{pkg}: bun test {' '.join(present)}" + ("" if p.returncode == 0 else "\n      " + "\n      ".join(tail)))
    for pkg in spec.get("typecheck", []):
        p = subprocess.run(["bun", "run", "typecheck"], cwd=os.path.join(REPO, pkg), capture_output=True, text=True)
        res.check(p.returncode == 0, f"{pkg}: typecheck")


def compare(value, bound):
    """A bound is {"min": x} or {"max": x}, or a bare number meaning a minimum."""
    if isinstance(bound, dict):
        return ("min" not in bound or value >= bound["min"]) and ("max" not in bound or value <= bound["max"])
    return value >= bound


def check_report(res, gate, spec):
    if "report" not in spec:
        return
    path = os.path.join(REPORTS, spec["report"])
    if not res.check(os.path.exists(path), f"{gate} report {path}"):
        return
    report = json.load(open(path))
    if "thresholds" not in spec:
        res.check(report.get("self_test") == "pass", f"{gate}: report's metrics self-test passed")
        res.check(report.get("baselines_ok", False), f"{gate}: baselines computed and in sane order")
        return
    if not res.check(spec["thresholds"] is not None, f"{gate}: thresholds set (G1)"):
        return
    for key, bound in spec["thresholds"].items():
        value = report.get("metrics", {}).get(key)
        res.check(value is not None and compare(value, bound), f"{gate}: {key} = {value} (bound {bound})")


def gate(name, args):
    res = Result()
    if name == "G0":
        gate_g0(res, os.path.expanduser(args.docs))
    elif name == "G1":
        gate_g1(res, os.path.expanduser(args.workspace))
    else:
        spec = SPEC[name]
        run_tests(res, spec)
        check_report(res, name, spec)
    return res


# ------------------------------------------------------------------ self-test
def demo():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "research", "papers"))
        for i in range(5):
            open(os.path.join(d, "research", "papers", f"2401.0000{i}-x.pdf"), "w").close()
        body = "\n".join(f"## {s}\ntext" for s in SPEC["G0"]["sections"][:-1])
        sources = "\n".join(f"- https://arxiv.org/abs/2401.0000{i % 5} paper {i}" for i in range(10))
        good = f"{body}\n## Sources\n{sources}\n"
        spec = dict(SPEC["G0"], memos=["research/a.md"])
        open(os.path.join(d, "research", "a.md"), "w").write(good)
        res = Result(); gate_g0(res, d, spec)
        assert res.ok, res.lines
        open(os.path.join(d, "research", "a.md"), "w").write(good.replace("## Risks", "## Riskz"))
        res = Result(); gate_g0(res, d, spec)
        assert not res.ok and any("missing: Risks" in l for l in res.lines), res.lines
        open(os.path.join(d, "research", "a.md"), "w").write(f"{body}\n## Sources\n- https://x.dev\n")
        res = Result(); gate_g0(res, d, spec)
        assert not res.ok
    assert compare(0.35, {"min": 0.3}) and not compare(0.35, {"max": 0.3}) and compare(3, 2)
    res = Result(); check_report(res, "GX", {"report": "no/such.json", "thresholds": None})
    assert not res.ok
    print("gates self-test passed")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", help="G0..G10, or demo")
    ap.add_argument("--docs", default="~/Desktop/OLaya/docs")
    ap.add_argument("--workspace", default="~/Desktop/OLaya")
    args = ap.parse_args(argv)
    if args.gate == "demo":
        demo()
        return 0
    res = gate(args.gate, args)
    print(f"{args.gate}:")
    for line in res.lines:
        print("  " + line)
    print(f"{args.gate}: {'PASSED' if res.ok else 'NOT PASSED'}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
