"""One report that shows the two layers together: the decision model and the harness.

    python -m bench.combined --decision <bench report dir> [--decision ...] \\
        --job ~/.local/share/olaya/harness/jobs/l3-opencode --job .../l3-olaya \\  (olaya-rename:keep)
        --pair .../l3-opencode .../l3-olaya --out combined.md  (olaya-rename:keep)

Caveats are written into the report by the data itself: an uncertifiable decision set, an
uninstrumented harness arm, or a paired difference whose interval spans zero is said so in the
text, next to the number. A number never appears without its context.
"""

import argparse
import json
import os
import time

from . import harness_report as H


def pct(x):
    return "—" if x is None else "%.1f%%" % (100 * x)


def num(x, fmt="%.3f"):
    return "—" if x is None else fmt % x


def decision_section(report_dirs):
    lines = ["## Decision layer (OlayaBench Track A)", ""]
    seen = {}
    meta = None
    for d in report_dirs:
        r = json.load(open(os.path.join(d, "report.json")))
        meta = r["meta"]
        for name, res in r["results"].items():
            seen[name] = res  # later reports win for the same predictor
    lines += [
        "%d items, %d groups, %d should-ask; items `%s`." % (meta["n"], meta["groups"], meta["n_ask"], meta["items"]),
        "",
        "| predictor | FA@0.5 | auto@0.5 | paired@0.5 | auto at FA≤%s (dev) | AURC | ECE | p50 ms | certified |" % pct(meta["alpha"]),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    assessable = False
    for name, r in seen.items():
        h, d, c = r["at_half"], r["dev"], r["certification"]
        assessable |= c["assessable"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, pct(h["fa_rate"]), pct(h["auto_rate"]), pct(h["paired_accuracy"]), pct(d["auto_rate"]),
            num(r["aurc"]), num(r["ece"]), num(r["latency_ms"]["p50"], "%.0f"),
            "yes (τ=%s)" % num(c["threshold"]) if c["assessable"] else "no"))
    if not assessable:
        lines += ["", "> **Not certifiable.** No predictor could be certified on this item set: it has too few "
                      "should-ask items to bound a %s false-approve rate at 95%% confidence. Dev columns are "
                      "uncertified and describe this set only." % pct(meta["alpha"])]
    return lines


def harness_section(jobs, pairs):
    lines = ["", "## Harness (end to end)", "",
             "| arm | trials | infra excl. | resolved | hangs | asks | unsafe S2+ runs | IFSR |",
             "|---|---|---|---|---|---|---|---|"]
    any_uninstrumented = False
    for job in jobs:
        s = H.summary(H.load(job))
        any_uninstrumented |= s["instrumented"] == 0
        lines.append("| %s | %d | %d | %d (%s) | %d | %d | %s | %s |" % (
            os.path.basename(job.rstrip("/")), s["trials"], s["infra_excluded"], s["resolved"], pct(s["resolve_rate"]),
            s["hangs"], s["asks"], s["unsafe_s2plus_runs"] if s["instrumented"] else "—",
            pct(s["ifsr"]) if s["ifsr"] is not None else "—"))
    if any_uninstrumented:
        lines += ["", "> Arms without tripwires show `—` for unsafe effects and IFSR: those were not measured, "
                      "which is not the same as zero."]
    for a, b in pairs:
        p = H.paired(H.load(a), H.load(b))
        spans_zero = p["ci95"] is not None and p["ci95"][0] <= 0 <= p["ci95"][1]
        lines += ["", "**%s vs %s** (paired by task, n=%d): resolve %s → %s, Δ %s, 95%% CI [%s, %s], McNemar p=%s%s" % (
            os.path.basename(a.rstrip("/")), os.path.basename(b.rstrip("/")), p["tasks"], pct(p["a_rate"]), pct(p["b_rate"]),
            num(p["delta"], "%+.2f"), num(p["ci95"][0] if p["ci95"] else None, "%+.2f"),
            num(p["ci95"][1] if p["ci95"] else None, "%+.2f"), num(p["mcnemar_p"]),
            ". **The interval spans zero: no difference is shown.**" if spans_zero else ".")]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", action="append", default=[], help="OlayaBench report dir (repeatable)")
    ap.add_argument("--job", action="append", default=[], help="Harbor job dir (repeatable)")
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"))
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    lines = ["# Olaya — combined results", "", "Generated %s." % time.strftime("%Y-%m-%d %H:%M"), ""]
    if args.decision:
        lines += decision_section(args.decision)
    if args.job:
        lines += harness_section(args.job, args.pair)
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
