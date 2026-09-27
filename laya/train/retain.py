"""Laya as a retention scorer, gate G8: "will the agent need this block's exact text later?"

    python -m train.retain build --public 240 --synthetic 120      # hindsight rows -> data/retain-v1
    python -m train.train --data ~/.local/share/olaya/laya/data/retain-v1 --out <checkpoint>
    python -m train.retain report --checkpoint <checkpoint>         # scores Track D -> retention/latest.json
    python -m train.retain demo                                     # self-test

Every tool result and assistant text before a compaction point is split into 100-250-token
blocks on natural boundaries. Laya sees one block with its metadata (tool kind, call, age,
references since, size, error lines), a digest of the task and the latest assistant step, and
answers one calibrated yes/no. An item's score is the max over its blocks. User messages are
pinned by provenance and never scored (design D4).

Training sessions share no repository with Track D's public sessions, and the synthetic ones use
other seeds. The synthetic generator is the same, though, so synthetic results are
in-distribution; the public ones are the honest test.

Labels here are hindsight labels (a block is needed when it holds a string an action after the cut
uses verbatim). Design D7 calibrates on teacher-forced counterfactual labels and keeps a gold set
from full replays; neither exists yet, so this report states its label source.
"""

import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch

from bench import recall as R
from bench.run import REPORTS

from . import data as D

QUESTIONS = {"needed_later": {"type": "noul", "instructions":
                              "Will the coding agent need this block's exact text later in the task?"}}
BOUNDARY = re.compile(r"^(?:_{4,}|={4,}|-{4,}|diff --git|@@ |Traceback|FAILED |ERROR |E {2,}|\s*def |\s*class )")
TRAIN_SESSIONS = os.path.join(R.HOME, "train-sessions.jsonl")


def blocks(text, lo=100, hi=250):
    """Split at natural boundaries into blocks of lo-hi tokens; short text is one block."""
    if R.tok(text) <= hi:
        return [text]
    out, cur, chars = [], [], 0
    for line in text.splitlines():
        if cur and (chars >= hi * 4 or (chars >= lo * 4 and BOUNDARY.match(line))):
            out.append("\n".join(cur))
            cur, chars = [], 0
        cur.append(line[:hi * 4])
        chars += len(cur[-1]) + 1
    if cur:
        out.append("\n".join(cur))
    return out


def states(s):
    """(item index, block, Laya state) for every scored history item."""
    h = s["items"][: s["cut"]]
    task = next((it["text"] for it in h if it["role"] == "user"), "")[:320]
    step = next((it["text"] for it in reversed(h) if it["role"] == "assistant"), "")[:240]
    last = h[-1]["turn"]
    marks = [set(R.PATH.findall(R.text_of(it))) for it in h]
    out = []
    for i, it in enumerate(h):
        if it["role"] == "user":
            continue
        refs = sum(1 for j in range(i + 1, len(h)) if marks[i] & marks[j])
        for b in blocks(it["text"]):
            out.append((i, b, {
                "source": it["role"], "kind": R.kind_of(it), "call": it.get("call", "")[:200],
                "age_turns": last - it["turn"], "refs_since": refs, "size_tokens": R.size(it),
                "error_lines": sum(1 for line in b.splitlines() if R.ERROR_LINE.search(line)),
                "task": task, "latest_step": step, "block": b,
            }))
    return out


def needed_texts(s):
    return [n["text"] for n in s["needles"] if n["needed"]]


# ------------------------------------------------------------------ training rows
def build(n_public, n_synthetic, neg_per_pos=3, seed=0, max_rows=3000, name="retain-v1"):
    heldout = {s["repo"] for s in map(json.loads, open(R.SESSIONS)) if s["source"] == "public"}
    pool = R.code_pool()
    sessions = [R.synthetic(1000 + i, pool) for i in range(n_synthetic)] + R.public(n_public, exclude=heldout)
    assert not heldout & {s["repo"] for s in sessions if s["source"] == "public"}
    with open(TRAIN_SESSIONS, "w") as f:
        for s in sessions:
            f.write(json.dumps({k: v for k, v in s.items() if not k.startswith("_")}) + "\n")
    rng = random.Random(seed)
    rows = []
    for s in sessions:
        need = needed_texts(s)
        pos, neg = [], []
        for i, b, state in states(s):
            (pos if any(t in b for t in need) else neg).append((i, state))
        rng.shuffle(neg)
        group = s["repo"] if s["source"] == "public" else s["id"]
        for (i, state), label in [(x, "needed") for x in pos] + [(x, "not_needed") for x in neg[: max(5, neg_per_pos * len(pos))]]:
            rows.append({
                "state": json.dumps(state, ensure_ascii=False), "questions": json.dumps(QUESTIONS),
                "gold": json.dumps({"needed_later": {"probabilities": {"true": float(label == "needed"),
                                                                       "false": float(label != "needed")}}}),
                "id": f"{s['id']}-{i}-{len(rows)}", "group": group,
                "split": D.split_of(group, calib_share=0.2, test_share=0.0),
                "source": s["source"], "labeler": "hindsight", "label": label,
            })
    # the trainer caches ~1 MB of encoder output per row, so the set is sampled down to fit memory
    rows = rng.sample(rows, min(max_rows, len(rows)))
    d, manifest = D.write_dataset(rows, name, QUESTIONS)
    by = {k: sum(r["label"] == "needed" for r in rows if r["split"] == k) for k in ("train", "calib")}
    print(f"{len(sessions)} training sessions ({n_synthetic} synthetic), {len(rows)} rows, needed per split {by} -> {d}")


# ------------------------------------------------------------------ scoring and the G8 report
def score(sessions, checkpoint, device=None, cache=None, platt=None):
    """Sets s["_scores"] (P(needed) per history item, max over blocks) and returns block-level
    (probability, label, session) triples. Block probabilities are cached in `cache` (a JSON file,
    keyed by session), and `platt` (a, b) recalibrates them."""
    per = [(s, states(s)) for s in sessions]
    cached = json.load(open(cache)) if cache and os.path.exists(cache) else {}
    if all(s["id"] in cached and len(cached[s["id"]]) == len(sts) for s, sts in per):
        probs = [p for s, _ in per for p in cached[s["id"]]]
    else:
        probs = model_probs([st for _, sts in per for _, _, st in sts], checkpoint, device)
        if cache:
            k, out = 0, {}
            for s, sts in per:
                out[s["id"]] = probs[k:k + len(sts)]
                k += len(sts)
            json.dump(out, open(cache, "w"))
    if platt:
        probs = [recalibrate(p, *platt) for p in probs]
    blocks_out, k = [], 0
    for s, sts in per:
        item = [1.0 if it["role"] == "user" else 0.0 for it in s["items"][: s["cut"]]]
        need = needed_texts(s)
        for i, b, _ in sts:
            item[i] = max(item[i], probs[k])
            blocks_out.append((probs[k], any(t in b for t in need), s["id"]))
            k += 1
        s["_scores"] = item
    return blocks_out


def model_probs(states_, checkpoint, device=None, questions=None):
    """P(true) for each state; `questions` gives each state its own one-question dict (the router
    asks per tier), and defaults to this module's question."""
    from laya.agent import Agent
    from laya.common import temp_bucket
    from .train import featurise, pick_device, raw_logits

    dev = pick_device(device)
    agent = Agent(checkpoint, device="cpu")
    model = agent.model.float().to(dev)
    rows = [{"state": json.dumps(st), "questions": json.dumps(q),
             "gold": json.dumps({next(iter(q)): {"probabilities": {"true": 0.0, "false": 1.0}}})}
            for st, q in zip(states_, questions or [QUESTIONS] * len(states_))]
    t0 = time.time()
    preds = raw_logits(model, featurise(rows, agent), agent.tok.pad_token_id, dev, cached=False)
    print(f"scored {len(rows)} blocks in {time.time() - t0:.0f}s", flush=True)
    probs = []
    for qt, z, _ in preds:
        t = agent.temperature_by_options.get(temp_bucket(qt, len(z)), agent.temperature[qt])
        probs.append(float(torch.softmax(z / t, -1)[1]))
    return probs


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def recalibrate(p, a, b):
    return float(1 / (1 + np.exp(-(a * logit(p) + b))))


def fit_platt(probs, labels, steps=200):
    """Platt scaling: p' = sigmoid(a * logit(p) + b), fitted by Newton's method on log loss."""
    x = np.array([logit(p) for p in probs])
    y = np.array(labels, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(steps):
        q = 1 / (1 + np.exp(-(a * x + b)))
        g = np.array([np.sum((q - y) * x), np.sum(q - y)])
        w = q * (1 - q) + 1e-9
        h = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def calibrate(checkpoint, device=None):
    """Fits Platt scaling on every block of the calibration sessions: the natural mix of needed and
    not needed, unlike the class-balanced training rows. The sessions' repositories are disjoint
    from Track D's. Saves platt.json next to the checkpoint."""
    sessions = [json.loads(l) for l in open(TRAIN_SESSIONS)]
    group = lambda s: s["repo"] if s["source"] == "public" else s["id"]
    calib = [s for s in sessions if D.split_of(group(s), calib_share=0.2, test_share=0.0) == "calib"]
    blk = score(calib, checkpoint, device, cache=os.path.join(checkpoint, "calib-scores.json"))
    probs, labels = [p for p, _, _ in blk], [y for _, y, _ in blk]
    a, b = fit_platt(probs, labels)
    out = {"a": a, "b": b, "sessions": len(calib), "blocks": len(blk), "positive_rate": round(sum(labels) / len(labels), 4),
           "ece_before": ece(probs, labels), "ece_after": ece([recalibrate(p, a, b) for p in probs], labels),
           # dev ranking quality: how a new head is chosen over the old one without touching Track D
           "auroc": round(auroc(probs, labels), 4)}
    json.dump(out, open(os.path.join(checkpoint, "platt.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def auroc(scores, labels):
    """Mann-Whitney AUROC, ties counted half."""
    pos = [x for x, y in zip(scores, labels) if y]
    neg = [x for x, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    order = np.argsort(np.array(pos + neg), kind="mergesort")
    ranks = np.empty(len(order))
    vals = np.array(pos + neg)[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return round(float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


def ece(probs, labels, bins=15):
    p, y = np.array(probs), np.array(labels, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return round(float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())), 4)


def item_labels(s):
    need = needed_texts(s)
    return [(i, any(t in R.text_of(it) for t in need)) for i, it in enumerate(s["items"][: s["cut"]]) if it["role"] != "user"]


def report(checkpoint, device=None):
    sessions = [json.loads(l) for l in open(R.SESSIONS)]
    platt_file = os.path.join(checkpoint, "platt.json")
    platt = (lambda d: (d["a"], d["b"]))(json.load(open(platt_file))) if os.path.exists(platt_file) else None
    blk = score(sessions, checkpoint, device, cache=os.path.join(checkpoint, "trackd-scores.json"), platt=platt)
    scored, rec, labels, src = [], [], [], []
    for s in sessions:
        last = s["items"][s["cut"] - 1]["turn"]
        for i, y in item_labels(s):
            scored.append(s["_scores"][i])
            rec.append(s["items"][i]["turn"] / max(1, last))
            labels.append(y)
            src.append(s["source"])
    curves = {p: R.curve(sessions, p) for p in ("laya", "pins_recency", "masking", "random", "recency")}
    at = lambda p, b, k="needed_recall": curves[p][str(b)][k]
    sub = lambda f: auroc([x for x, s_ in zip(scored, src) if s_ == f], [y for y, s_ in zip(labels, src) if s_ == f])
    metrics = {
        "scorer_auroc": auroc(scored, labels),
        "auroc_minus_recency": round(auroc(scored, labels) - auroc(rec, labels), 4),
        "ece": ece([p for p, _, _ in blk], [y for _, y, _ in blk]),
        "pinned_retention": min(at("laya", b, "pinned_retention") for b in R.BUDGETS),
        "needed_recall_at_40": at("laya", 0.4),
        "recall_at_40_minus_masking": round(at("laya", 0.4) - at("masking", 0.4), 4),
        "error_string_recall": at("laya", 0.4, "error_string_recall"),
        "constraint_adherence": at("laya", 0.4, "constraint_recall"),
        "first_error_recall": at("laya", 0.4, "first_error_recall"),
    }
    rep = {
        "gate": "G8", "generated": time.strftime("%Y-%m-%d %H:%M"), "checkpoint": checkpoint,
        "label_source": "hindsight (counterfactual calibration and replay gold not built yet)",
        "calibration": "Platt scaling fitted on natural-distribution calibration sessions" if platt else "temperature only",
        "notes": {"constraint_adherence": "offline proxy: constraint needles retained at 40%, not probed adherence",
                  "synthetic": "same generator as training (other seeds): in-distribution"},
        "self_test": "pass" if self_test() and R.self_test() else "fail",
        "baselines_ok": bool(at("laya", 1.0) == 1.0 and at("masking", 0.4) is not None),
        "metrics": metrics,
        "by_source": {f: {"scorer_auroc": sub(f), **{p: R.curve([s for s in sessions if s["source"] == f], p)["0.4"]
                                                     for p in ("laya", "pins_recency", "masking")}}
                      for f in ("synthetic", "public")},
        "curves": curves, "area": {p: R.area(c) for p, c in curves.items()},
    }
    os.makedirs(os.path.join(REPORTS, "retention"), exist_ok=True)
    path = os.path.join(REPORTS, "retention", "latest.json")
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ("label_source", "self_test", "baselines_ok", "metrics", "area")}, indent=2))
    for f, v in rep["by_source"].items():
        print(f, "auroc", v["scorer_auroc"], {p: v[p]["needed_recall"] for p in ("laya", "pins_recency", "masking")})
    print("->", path)


def self_test():
    # a boundary line splits once a block has 100 tokens (400 chars); without one, 250 tokens is the cap
    text = "\n".join(["x" * 79] * 8 + ["Traceback (most recent call last):"] + ["y" * 79] * 8)
    b = blocks(text)
    assert len(b) == 2 and b[0].count("\n") == 7 and b[1].startswith("Traceback")
    assert "".join(b).replace("\n", "") == text.replace("\n", "")
    assert [x.count("\n") + 1 for x in blocks("\n".join(["x" * 79] * 30))] == [13, 13, 4]
    assert blocks("short") == ["short"]
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0 and auroc([0.5, 0.5], [1, 0]) == 0.5
    assert auroc([0.1, 0.9, 0.5], [1, 0, 0]) == 0.0
    assert ece([0.0, 1.0], [0, 1]) == 0.0 and ece([0.9] * 10, [1] * 9 + [0]) == 0.0
    # Platt scaling pulls over-confident scores toward the observed rate
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.2).astype(int)
    p = np.clip(0.5 + 0.35 * (y - 0.5) * 2 + rng.normal(0, 0.1, 4000), 0.01, 0.99)
    a, b = fit_platt(list(p), list(y))
    assert ece([recalibrate(x, a, b) for x in p], list(y)) < ece(list(p), list(y))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--public", type=int, default=240); b.add_argument("--synthetic", type=int, default=120)
    b.add_argument("--max-rows", type=int, default=3000); b.add_argument("--name", default="retain-v1")
    r = sub.add_parser("report"); r.add_argument("--checkpoint", required=True); r.add_argument("--device")
    c = sub.add_parser("calibrate"); c.add_argument("--checkpoint", required=True); c.add_argument("--device")
    sub.add_parser("demo")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.public, a.synthetic, max_rows=a.max_rows, name=a.name)
    elif a.cmd == "report":
        report(a.checkpoint, a.device)
    elif a.cmd == "calibrate":
        calibrate(a.checkpoint, a.device)
    else:
        print("retain self-test", "passed" if self_test() else "FAILED")


if __name__ == "__main__":
    main()
