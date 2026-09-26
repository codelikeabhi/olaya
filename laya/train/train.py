"""Fine-tune a Laya checkpoint on coding decisions. Runs on Apple Silicon (MPS), CUDA or CPU.

    python -m train.train --data ~/.local/share/olaya/laya/data/<name> --out <dir>
    python -m train.train --data ... --unfreeze-encoder      # second move, not the first

Port of Laya's 2xT4 notebook (RLCD: Gaussian-perturbed logits scored by proper_reward, a
group-normalised policy gradient, plus soft cross-entropy to the gold distribution) with the
CUDA-only parts removed: one process, fp32, no GradScaler or autocast.

Default is head-only. A frozen encoder's output is fixed per input, so it is computed ONCE and
the head trains over the cached states for as many epochs as needed; that turns an epoch into
seconds and makes iterating on data cheap.

After training, temperatures are fitted per (question type, option count) bucket on the
calibration split and written to `temperature_by_options`, REPLACING the inherited dict. The
shipped notebook left the inherited dict in place, and since it wins at inference the fitted
values were silently discarded (defect D1). Our only bucket, noul:2, is in the inherited dict.
"""

import argparse
import json
import math
import os
import random
import shutil
import time

import torch

from laya.agent import Agent
from laya.common import QTYPES, build_sequence, clamp_temperature, proper_reward, temp_bucket

from . import data as D

HERE = os.path.dirname(os.path.abspath(__file__))


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def base_dir(model_id):
    """Local directory of a checkpoint (hub ids resolve from the cache, as laya.Agent does)."""
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])


def featurise(rows, agent):
    """Rows -> token sequences, option markers and target distributions (noul: [false, true])."""
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    items = []
    for row in rows:
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, qdef in questions.items():
            q = Agent._to_internal(qdef)
            seq, markers = build_sequence(agent.tok, json.loads(row["state"]), q, max_len, head_max_len)
            probs = gold[qid]["probabilities"]
            if q["t"] != "noul":
                raise ValueError("only noul questions are trained here; got %r" % q["t"])
            target = [float(probs.get("false", 0.0)), float(probs.get("true", 0.0))]
            s = sum(target) or 1.0
            items.append({
                "ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "target": [t / s for t in target],
                "label": row.get("label"), "reply": row.get("reply"), "id": row.get("id"),
            })
    return items


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    k = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, k), dtype=torch.long)
    mmask = torch.zeros((n, k), dtype=torch.bool)
    target = torch.zeros((n, k))
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        mpos[i, : len(it["markers"])] = torch.tensor(it["markers"])
        mmask[i, : len(it["markers"])] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"])
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "qtype": torch.tensor([it["qtype"] for it in items])}


@torch.no_grad()
def encode(model, items, pad_id, device, batch=16):
    """Cache the frozen encoder's output per item (fp16 on CPU)."""
    model.eval()
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        b = collate(chunk, pad_id)
        h = model.encoder(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).last_hidden_state
        for j, it in enumerate(chunk):
            it["hidden"] = h[j, : len(it["ids"])].to("cpu", torch.float16)


def hidden_batch(items, device):
    L = max(it["hidden"].shape[0] for it in items)
    d = items[0]["hidden"].shape[1]
    h = torch.zeros((len(items), L, d), dtype=torch.float32)
    for i, it in enumerate(items):
        h[i, : it["hidden"].shape[0]] = it["hidden"].float()
    return h.to(device)


def forward(model, items, pad_id, device, cached):
    b = collate(items, pad_id)
    logits, act = model(
        b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
        b["marker_mask"].to(device), b["qtype"].to(device),
        hidden=hidden_batch(items, device) if cached else None,
    )
    return logits.float(), act, b


def rlcd_loss(logits, act, b, device, sigma, group=4, w_ce=1.0, weights=None):
    """The notebook's objective: policy gradient against proper_reward + soft cross-entropy."""
    mask = b["marker_mask"].to(device)
    target = b["target"].to(device)
    qtype = b["qtype"].to(device)
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group,) + logits.shape, device=device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        adv = (r - r.mean(0, keepdim=True)) / (r.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    per_rl = -(adv * logp).mean(0)
    per_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)
    per = per_rl + w_ce * per_ce
    if weights is not None:
        per = per * weights.to(device)
    # act_head stays in the graph at zero weight so the state dict is the stock layout.
    return per.mean() + 0.0 * act.sum()


def sample_weights(items, always_weight):
    """Class-balanced: each label (approve/ask, or needed/not_needed) carries equal total weight;
    `always` replies count more."""
    counts = {}
    for it in items:
        counts[it["label"]] = counts.get(it["label"], 0) + 1
    w = [len(items) / (len(counts) * counts[it["label"]]) * (always_weight if it.get("reply") == "always" else 1.0)
         for it in items]
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def raw_logits(model, items, pad_id, device, cached, batch=32):
    model.eval()
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        logits, _, _ = forward(model, chunk, pad_id, device, cached)
        for j, it in enumerate(chunk):
            out.append((it["qtype"], logits[j, : len(it["markers"])].cpu(), torch.tensor(it["target"])))
    return out


def fit_temperatures(preds):
    """One temperature per bucket by LBFGS on soft cross-entropy, clamped to Laya's range."""
    buckets = {}
    for qt, z, t in preds:
        buckets.setdefault(temp_bucket(qt, len(z)), []).append((z, t))
    fitted = {}
    for name, sel in buckets.items():
        if len(sel) < 10:
            continue  # too few to fit; the bucket then falls back to cfg["temperature"]
        Z = torch.stack([z for z, _ in sel])
        T = torch.stack([t for _, t in sel])
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
            loss.backward()
            return loss

        opt.step(closure)
        fitted[name] = round(clamp_temperature(float(log_t.exp())), 4)
    return fitted


def save(model, agent_cfg, src_dir, out, manifest, temperatures):
    from safetensors.torch import save_file

    os.makedirs(out, exist_ok=True)
    state = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(out, "model.safetensors"))
    cfg = dict(agent_cfg)
    # D1: replace, never merge. An inherited bucket would override the fitted one at inference.
    cfg["temperature_by_options"] = temperatures
    with open(os.path.join(out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    for sub in ("tokenizer", "encoder"):
        dst = os.path.join(out, sub)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(src_dir, sub), dst)
    with open(os.path.join(out, "olaya_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir from train.data (rows.jsonl + manifest.json)")
    ap.add_argument("--base", default="convaiinnovations/laya")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--unfreeze-encoder", action="store_true")
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    ap.add_argument("--always-weight", type=float, default=1.5)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimiser steps (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-stale-data", action="store_true")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    rows, data_manifest = D.load_dataset(args.data, require_current_compactor=not args.allow_stale_data)
    train_rows = [r for r in rows if r["split"] == "train"]
    calib_rows = [r for r in rows if r["split"] == "calib"]
    if not train_rows:
        raise SystemExit("no training rows")

    src = base_dir(args.base)
    agent = Agent(src, device="cpu")
    model = agent.model.float().to(device)
    pad = agent.tok.pad_token_id
    train_items = featurise(train_rows, agent)
    calib_items = featurise(calib_rows, agent)
    cached = not args.unfreeze_encoder

    if cached:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        t0 = time.time()
        encode(model, train_items + calib_items, pad, device)
        print("encoded %d items in %.1fs" % (len(train_items) + len(calib_items), time.time() - t0), flush=True)

    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    groups = [{"params": head, "lr": args.lr_head}]
    if not cached:
        groups.append({"params": list(model.encoder.parameters()), "lr": args.lr_encoder})
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_items) / args.batch)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total), eta_min=1e-6)
    weights = sample_weights(train_items, args.always_weight)

    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train_items)))
        random.shuffle(order)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * epoch / max(1, args.epochs - 1)
        running = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            chunk = [train_items[j] for j in idx]
            logits, act, b = forward(model, chunk, pad, device, cached)
            loss = rlcd_loss(logits, act, b, device, sigma, weights=weights[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            running += float(loss.detach())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        print("epoch %d/%d loss %.4f (%.1fs)" % (epoch + 1, args.epochs, running / max(1, steps_per_epoch), time.time() - t0), flush=True)
        if args.max_steps and step >= args.max_steps:
            break

    temperatures = fit_temperatures(raw_logits(model, calib_items, pad, device, cached)) if calib_items else {}
    print("fitted temperatures:", temperatures, flush=True)

    questions = json.loads(train_rows[0]["questions"])
    manifest = {
        "base": args.base,
        "dataset": {"name": data_manifest["name"], "content_sha256": data_manifest["content_sha256"]},
        "compactor_sha256": data_manifest["compactor_sha256"],
        "questions": questions,
        "question_hash": D.question_hash(questions),
        "state_budget_at_training": None,
        "encoder": "frozen" if cached else "trained",
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("data", "out")},
        "temperature_by_options": temperatures,
        "train_items": len(train_items),
        "calib_items": len(calib_items),
        "steps": step,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save(model, agent.cfg, src, args.out, manifest, temperatures)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
