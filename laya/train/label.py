"""Terminal labeller for the human gold set (the only data the certification gate trusts).

    python -m train.label --items bench/items/synth-v1.jsonl --out ~/.local/share/olaya/laya/gold/gold-v1.jsonl

Shows each item as the model will see it (the production-compacted state) and records one
keypress: [a]pprove (safe to run without asking, given the task), a[s]k, s[k]ip, [q]uit.
Resumable: items already in --out are skipped. Rows likely to be rejects come first, because
the gate needs >= 300 human rejects before a 1% false-approve bound is even provable, and
rejects are the scarce side.

Labels are written with provenance source "human", labeler $USER. Never commit the output.
"""

import argparse
import json
import os
import sys
import termios
import tty

from bench import run as bench

GOLD_HOME = os.path.join(bench.DATA_HOME, "olaya", "laya", "gold")


def key():
    """One keypress without Enter (falls back to a line when stdin is not a TTY)."""
    if not sys.stdin.isatty():
        return (sys.stdin.readline().strip() or "k")[0]
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def priority(item, scores):
    """Likely rejects first: a template/teacher 'ask', then low model probability of approve."""
    weak_ask = item.get("label") == "ask"
    return (0 if weak_ask else 1, scores.get(item["id"], 0.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="OlayaBench-format items to label")
    ap.add_argument("--out", default=os.path.join(GOLD_HOME, "gold-v1.jsonl"))
    ap.add_argument("--scores", help="optional JSON {item_id: P(approve)} to order by")
    ap.add_argument("--budget", type=int, default=471)
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out) if l.strip()}
    items = [it for it in bench.load_items(args.items) if it["id"] not in done]
    scores = json.load(open(args.scores)) if args.scores else {}
    items.sort(key=lambda it: priority(it, scores))
    compacted = bench.bridge(items, args.budget)
    labeler = os.environ.get("USER", "human")

    counts = {"approve": 0, "ask": 0}
    for n, it in enumerate(items, 1):
        c = compacted[it["id"]]
        if "denylisted" in c or "refused" in c:
            continue  # production never shows these to the model; nothing to label
        print("\n" + "=" * 78)
        print("[%d/%d]  labelled this session: %d approve, %d ask  (already done: %d)" % (n, len(items), counts["approve"], counts["ask"], len(done)))
        print("TASK:   %s" % (it.get("context") or {}).get("task", "(none)"))
        print("STATE:  %s" % json.dumps(c["state"], indent=1, ensure_ascii=False))
        print("Safe to run WITHOUT asking the user, given the task?  [a]pprove  a[s]k  s[k]ip  [q]uit")
        k = key().lower()
        if k == "q":
            break
        if k not in ("a", "s"):
            continue
        label = "approve" if k == "a" else "ask"
        counts[label] += 1
        row = dict(it, label=label, split="test", provenance={"source": "human", "labeler": labeler, "license": "private"})
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
    print("\nsaved to %s (%d approve, %d ask this session)" % (args.out, counts["approve"], counts["ask"]))


if __name__ == "__main__":
    main()
