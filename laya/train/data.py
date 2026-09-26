"""Training rows for the coding-decision fine-tune.

A row is Laya's native schema (`state`, `questions`, `gold` as JSON strings) plus provenance
keys the trainer ignores. Rows come from two places:

- items in OlayaBench format (raw request + task context + label), compacted by the
  PRODUCTION compactor through the bun bridge, never by a Python copy of it;
- Olaya shadow logs, which are already written in this schema.

Splits are assigned per `group`, so a counterfactual group (same action, different tasks)
never straddles train and evaluation.
"""

import hashlib
import json
import os

from bench import run as bench

DATA_HOME = os.path.join(bench.DATA_HOME, "olaya", "laya", "data")
STATE_TS = os.path.join(bench.REPO, "packages", "olaya", "src", "laya", "state.ts")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def question_hash(questions):
    # Same canonical form as the sidecar (service.question_hash) and the TypeScript client.
    canon = json.dumps(questions, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def gold_from_label(label, qid, soft=None):
    """`soft` is P(approve) from a teacher; a human label is one-hot."""
    p = soft if soft is not None else (1.0 if label == "approve" else 0.0)
    return {qid: {"probabilities": {"true": p, "false": 1.0 - p}}}


def split_of(group, calib_share=0.2, test_share=0.2):
    """Stable split from the group id: the same group always lands in the same split."""
    h = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 1000 / 1000
    if h < test_share:
        return "test"
    if h < test_share + calib_share:
        return "calib"
    return "train"


def rows_from_items(items, questions, budget):
    """Compact items with the production code and turn them into training rows.

    Denylisted and refused items are dropped: production never sends them to the model, so
    training on them would teach a distribution the model is never shown.
    """
    qid = next(iter(questions))
    compacted = bench.bridge(items, budget)
    rows, dropped = [], {"denylisted": 0, "refused": 0}
    for it in items:
        c = compacted[it["id"]]
        if "denylisted" in c:
            dropped["denylisted"] += 1
            continue
        if "refused" in c:
            dropped["refused"] += 1
            continue
        rows.append({
            "state": json.dumps(c["state"], ensure_ascii=False),
            "questions": json.dumps(questions, ensure_ascii=False),
            "gold": json.dumps(gold_from_label(it.get("label"), qid, it.get("soft"))),
            "id": it["id"],
            "group": it["group"],
            "split": it.get("split") or split_of(it["group"]),
            "source": (it.get("provenance") or {}).get("source", "unknown"),
            "labeler": (it.get("provenance") or {}).get("labeler", "unknown"),
            "label": it.get("label"),
            "reply": it.get("reply"),
            "family": it.get("family"),
            "route": it.get("route"),
            "severity": it.get("severity"),
        })
    return rows, dropped


def rows_from_shadow(directory):
    """Labelled shadow decisions. Unlabelled ones (gold null) and refusals are skipped."""
    rows = []
    if not os.path.isdir(directory):
        return rows
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("shadow-") and name.endswith(".jsonl")):
            continue
        for line in open(os.path.join(directory, name)):
            rec = json.loads(line)
            if rec.get("kind") != "decision" or rec.get("gold") is None:
                continue
            gold = json.loads(rec["gold"])
            p = next(iter(gold.values()))["probabilities"]["true"]
            rows.append(dict(
                rec,
                group="shadow-" + rec["id"],
                split=split_of("shadow-" + rec["id"]),
                source="shadow",
                labeler="user",
                label="approve" if p >= 0.5 else "ask",
            ))
    return rows


def write_dataset(rows, name, questions, dropped=None, out=DATA_HOME):
    """Write rows plus a manifest that pins everything the rows depend on."""
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    with open(os.path.join(d, "rows.jsonl"), "w") as f:
        f.write(body)
    counts = {}
    for r in rows:
        s = counts.setdefault(r["split"], {"n": 0, "ask": 0, "by_source": {}})
        s["n"] += 1
        s["ask"] += r["label"] == "ask"
        s["by_source"][r["source"]] = s["by_source"].get(r["source"], 0) + 1
    manifest = {
        "name": name,
        "rows": len(rows),
        "splits": counts,
        "dropped": dropped or {},
        "question_hash": question_hash(questions),
        "compactor_sha256": sha256_file(STATE_TS),
        "content_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return d, manifest


def load_dataset(path, require_current_compactor=True):
    """Rows of a built dataset. Refuses data built by a different compactor than the one in
    the tree: such states are not what production sends any more."""
    manifest = json.load(open(os.path.join(path, "manifest.json")))
    if require_current_compactor and manifest["compactor_sha256"] != sha256_file(STATE_TS):
        raise SystemExit("dataset %s was built with a different state.ts; rebuild it" % path)
    rows = [json.loads(l) for l in open(os.path.join(path, "rows.jsonl"))]
    return rows, manifest


def demo():
    assert split_of("g1") == split_of("g1")
    shares = [split_of("g%d" % i) for i in range(4000)]
    assert 0.15 < shares.count("test") / 4000 < 0.25 and 0.15 < shares.count("calib") / 4000 < 0.25
    g = gold_from_label("ask", "q")
    assert g["q"]["probabilities"] == {"true": 0.0, "false": 1.0}
    assert gold_from_label(None, "q", soft=0.8)["q"]["probabilities"]["true"] == 0.8
    print("data self-check ok")


if __name__ == "__main__":
    demo()
