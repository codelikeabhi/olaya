"""Counterfactual labels for the retention scorer (gate G8, design D7).

    python -m train.counterfactual label --model Qwen/Qwen3-1.7B --states 250 [--device mps]
    python -m train.counterfactual merge     # retain-v1 + retain-cf -> retain-v2 for train.train
    python -m train.counterfactual demo      # mechanics self-test: a tiny random model, CPU, no download

Hindsight labels call a block "needed" when an action after the compaction point quotes it. A
counterfactual label asks instead whether the agent's actual next actions become less likely
without it: under a fixed causal LM, the drop in log-probability of the calls made after the cut
when one block is replaced by a stub. Blocks are scored in training sessions only (repositories
disjoint from Track D); the hindsight positives of each state are scored with sampled negatives.

The drop is kept on every row (`delta`), and a block is labelled "needed" when its drop is in the
top share that matches hindsight's positive rate, so the two label sets are comparable. The rows
extend retain-v1 for retraining; Track D is untouched until the scorer is evaluated once.
"""

import argparse
import glob
import json
import math
import os
import random

import torch

from . import data as D
from . import retain

STUB = "[...]"
# 4k tokens of history before the actions: attention cost grows with the square of this, and 8k made a
# forward pass tens of seconds on the laptop GPU
MAX_CTX = 4096
# room held back in every comparison for a restated block (blocks are 100-250 tokens), so all of a
# state's scores see the same stretch of history
RESERVE = 320
RESULT_CHARS = 600


def render(it):
    if it["role"] == "tool":
        return f"CALL {it.get('call', '')}\nRESULT {it['text']}\n"
    return f"{it['role'].upper()} {it['text']}\n"


def targets(s):
    """(context, scored) text pairs after the cut: each call and assistant message is scored; the
    tool results between them are context only."""
    out = []
    for it in s["items"][s["cut"]:]:
        if it["role"] == "tool":
            out.append(("", f"CALL {it.get('call', '')}\n"))
            out.append((f"RESULT {it['text'][:RESULT_CHARS]}\n", ""))
        elif it["role"] == "assistant":
            out.append(("", f"ASSISTANT {it['text']}\n"))
    return out


@torch.no_grad()
@torch.no_grad()  # scoring only: with gradients on, each 4k pass held its activations (24 GB, swapping)
def logprob(model, tok, history, segments, device, keep="", reserve=0):
    """Sum of log p over the scored segments, given `keep` and then the history, left-truncated to
    fit MAX_CTX (`keep` itself is never cut). `reserve` holds back room as if `keep` were there, so
    a comparison with and without it sees the same stretch of history."""
    ids, scored = [], []
    for context, target in segments:
        for text, score in ((context, False), (target, True)):
            if text:
                piece = tok(text, add_special_tokens=False)["input_ids"]
                ids += piece
                scored += [score] * len(piece)
    # the scored actions alone may not fit: keep their first half-window, so history still has room
    ids, scored = ids[: MAX_CTX // 2], scored[: MAX_CTX // 2]
    if not any(scored):
        return 0.0
    kept = tok(keep, add_special_tokens=False)["input_ids"] if keep else []
    room = MAX_CTX - len(ids) - max(len(kept), reserve)
    past = tok(history, add_special_tokens=False)["input_ids"]
    # `past[-0:]` is all of it: with no room, no history at all
    prefix = kept + (past[-room:] if room > 0 else [])
    ids, scored = prefix + ids, [False] * len(prefix) + scored
    x = torch.tensor([ids], device=device)
    # only the scored positions go through the output head: a full 8k x vocab logit matrix is
    # gigabytes, and all but a few hundred rows of it would be thrown away
    at = torch.tensor([i for i in range(len(ids) - 1) if scored[i + 1]], device=device)
    hidden = model.base_model(x).last_hidden_state[0, at]
    lp = torch.log_softmax(model.get_output_embeddings()(hidden).float(), -1)
    return float(lp.gather(1, x[0, at + 1, None])[:, 0].sum())


def deltas(model, tok, s, candidates, device):
    """Log-probability drop of the post-cut actions without each candidate block.

    The history is left-truncated to fit the window, and most blocks of a long session lie before it,
    where stubbing them changes nothing. One baseline per state (the history as it was); then a block
    inside the kept window is scored as baseline minus the history with it stubbed, and a block
    before the window as the history with the block restated at the front minus the baseline. The
    same room is held back in every pass, so all of a state's scores see the same stretch."""
    h = s["items"][: s["cut"]]
    segments = targets(s)
    rendered = [render(it) for it in h]
    history = "".join(rendered)
    base = logprob(model, tok, history, segments, device, reserve=RESERVE)
    start = window_start(tok, history, segments)
    out = []
    for i, block, state in candidates:
        at = sum(len(r) for r in rendered[:i]) + rendered[i].find(block)
        if at >= start:  # in the window: take it out
            without = "".join(render(dict(it, text=it["text"].replace(block, STUB, 1))) if j == i else r
                              for j, (it, r) in enumerate(zip(h, rendered)))
            d = base - logprob(model, tok, without, segments, device, reserve=RESERVE)
        else:  # before the window: bring it into view
            restated = f"EARLIER OUTPUT ({h[i].get('call') or h[i]['role']}):\n{block}\n"
            d = logprob(model, tok, history, segments, device, keep=restated, reserve=RESERVE) - base
        out.append((i, block, state, d))
    return out


def window_start(tok, history, segments):
    """The character offset where the kept part of the history begins."""
    target = sum(len(tok(c + t, add_special_tokens=False)["input_ids"]) for c, t in segments)
    enc = tok(history, add_special_tokens=False, return_offsets_mapping=True)
    first = len(enc["input_ids"]) - max(0, MAX_CTX - target - RESERVE)
    if first >= len(enc["input_ids"]):
        return len(history)  # no room left: every block lies before the window
    return enc["offset_mapping"][first][0] if first > 0 else 0


def label(model_name, n_states, device, per_state=12, seed=0, name="retain-cf", positive_rate=0.078):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device != "cpu" else torch.float32)
    model.to(device).eval()
    return run(model, tok, n_states, device, per_state, seed, name, positive_rate)


def run(model, tok, n_states, device, per_state=12, seed=0, name="retain-cf", positive_rate=0.078, write=True):
    rng = random.Random(seed)
    sessions = [json.loads(l) for l in open(retain.TRAIN_SESSIONS)]
    rng.shuffle(sessions)
    scored = []
    for s in sessions[:n_states]:
        if not targets(s):
            continue
        need = retain.needed_texts(s)
        cands = retain.states(s)
        pos = [c for c in cands if any(t in c[1] for t in need)]
        neg = [c for c in cands if c not in pos]
        rng.shuffle(neg)
        pick = pos[: per_state // 2] + neg[: per_state - min(len(pos), per_state // 2)]
        scored += [(s, i, block, state, d, any(t in block for t in need)) for i, block, state, d in deltas(model, tok, s, pick, device)]
        print(f"{s['id']}: {len(pick)} blocks, max drop {max((x[4] for x in scored[-len(pick):]), default=0):.2f}", flush=True)
    if not scored:
        return []
    cut = sorted((x[4] for x in scored), reverse=True)[max(0, math.ceil(positive_rate * len(scored)) - 1)]
    rows = []
    for s, i, block, state, d, hindsight in scored:
        needed = d >= cut and d > 0
        group = s["repo"] if s["source"] == "public" else s["id"]
        rows.append({
            "state": json.dumps(state, ensure_ascii=False), "questions": json.dumps(retain.QUESTIONS),
            "gold": json.dumps({"needed_later": {"probabilities": {"true": float(needed), "false": float(not needed)}}}),
            "id": f"{s['id']}-{i}-cf{len(rows)}", "group": group,
            "split": D.split_of(group, calib_share=0.2, test_share=0.0),
            "source": s["source"], "labeler": "counterfactual", "label": "needed" if needed else "not_needed",
            "delta": round(d, 4), "hindsight": "needed" if hindsight else "not_needed",
        })
    agree = sum((r["label"] == r["hindsight"]) for r in rows) / len(rows)
    print(f"{len(rows)} rows, drop threshold {cut:.3f}, {sum(r['label'] == 'needed' for r in rows)} needed, agreement with hindsight {agree:.3f}")
    if write:
        d, _ = D.write_dataset(rows, name, retain.QUESTIONS)
        print("->", d)
    return rows


def merge(n_each=1500, cf_weight=2.0, seed=0, name="retain-v2"):
    """retain-v2: hindsight rows plus counterfactual rows, at most ~3,000 (the trainer caches ~1 MB
    per row). Where both label the same block, the counterfactual label wins; its rows count double."""
    rng = random.Random(seed)
    read = lambda d: [json.loads(l) for l in open(os.path.join(D.DATA_HOME, d, "rows.jsonl"))]
    cf = read("retain-cf")
    rng.shuffle(cf)
    cf = [dict(r, weight=cf_weight) for r in cf[:n_each]]
    taken = {r["state"] for r in cf}
    hindsight = [r for r in read("retain-v1") if r["state"] not in taken]
    rng.shuffle(hindsight)
    rows = hindsight[:n_each] + cf
    d, _ = D.write_dataset(rows, name, retain.QUESTIONS)
    print(f"{len(rows)} rows ({len(rows) - len(cf)} hindsight, {len(cf)} counterfactual at weight {cf_weight}) -> {d}")
    return rows


def self_test():
    """Mechanics on a tiny random model: a stubbed block changes the score, rows are well formed."""
    global MAX_CTX
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    # any tokenizer will do for mechanics; Laya's is already cached (in an older snapshot)
    path = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/*/tokenizer"))[0]
    tok = AutoTokenizer.from_pretrained(path)
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                         num_key_value_heads=2, intermediate_size=64, max_position_embeddings=MAX_CTX)).eval()
    s = {"items": [{"role": "user", "text": "fix the failing test", "turn": 0},
                   {"role": "tool", "call": "bash(pytest)", "text": "E KeyError: 'tenant_d477'", "turn": 1},
                   {"role": "tool", "call": "read(a.py)", "text": "def f(): pass", "turn": 2},
                   {"role": "tool", "call": "edit(a.py, 'tenant_d477')", "text": "Edit applied.", "turn": 3}], "cut": 3}
    segments = targets(s)
    assert segments == [("", "CALL edit(a.py, 'tenant_d477')\n"), ("RESULT Edit applied.\n", "")], segments
    out = deltas(model, tok, s, [(1, "E KeyError: 'tenant_d477'", {}), (2, "def f(): pass", {})], "cpu")
    assert len(out) == 2 and all(math.isfinite(d) for *_, d in out) and any(d != 0 for *_, d in out), out
    # a block far before the kept window still counts (the first run scored every such block 0)
    long = dict(s, items=[s["items"][0], s["items"][1], *[{"role": "tool", "call": f"read(f{n}.py)", "text": "x = 1\n" * 400, "turn": 2}
                                                          for n in range(8)], *s["items"][2:]], cut=s["cut"] + 8)
    MAX_CTX, saved = 256, MAX_CTX
    try:
        far = deltas(model, tok, long, [(1, "E KeyError: 'tenant_d477'", {})], "cpu")
    finally:
        MAX_CTX = saved
    assert far[0][3] != 0, far
    # scored actions longer than the window: every pass stays inside it (a `-0` slice once took the
    # whole history), and no gradients are kept
    seen = []
    forward = model.base_model.forward
    model.base_model.forward = lambda x, *a, **k: (seen.append((x.shape[1], torch.is_grad_enabled())), forward(x, *a, **k))[1]
    MAX_CTX, saved = 256, MAX_CTX
    try:
        assert math.isfinite(logprob(model, tok, "y " * 2000, [("", "z " * 600)], "cpu"))
    finally:
        MAX_CTX = saved
        model.base_model.forward = forward
    assert seen and all(n <= 256 and not grad for n, grad in seen), seen
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("label"); l.add_argument("--model", default="Qwen/Qwen3-1.7B"); l.add_argument("--states", type=int, default=250)
    l.add_argument("--device", default="mps"); l.add_argument("--per-state", type=int, default=12)
    sub.add_parser("merge")
    sub.add_parser("demo")
    a = ap.parse_args(argv)
    if a.cmd == "label":
        label(a.model, a.states, a.device, a.per_state)
    elif a.cmd == "merge":
        merge()
    else:
        print("counterfactual self-test", "passed" if self_test() else "FAILED")


if __name__ == "__main__":
    main()
