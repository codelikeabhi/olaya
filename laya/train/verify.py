"""Verify a trained checkpoint before anyone serves it.

    python -m train.verify <checkpoint> [--base convaiinnovations/laya]

Checks, each of which has failed silently in the past or would:
1. stock `laya.load` accepts it (no patched library needed to serve it);
2. the fitted temperatures are the ones applied at inference (defect D1: inherited
   `temperature_by_options` used to override them);
3. no stale inherited bucket survives (e.g. choice:11+ = 0.1006);
4. a frozen-encoder run left the encoder bit-identical to the base;
5. the manifest pins the production question set by the same hash the sidecar computes.
"""

import argparse
import json
import os
import sys

import torch


def check(ckpt, base="convaiinnovations/laya"):
    import laya
    from safetensors.torch import load_file

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import service
    from train.train import base_dir

    failures = []
    manifest = json.load(open(os.path.join(ckpt, "olaya_manifest.json")))
    cfg = json.load(open(os.path.join(ckpt, "rl_agent_config.json")))

    agent = laya.load(ckpt, device="cpu")  # 1
    fitted = manifest["temperature_by_options"]
    for bucket, t in fitted.items():  # 2
        if abs(agent.temperature_by_options.get(bucket, -1) - t) > 1e-6:
            failures.append("bucket %s applies %s, fitted %s" % (bucket, agent.temperature_by_options.get(bucket), t))
    stale = set(cfg.get("temperature_by_options", {})) - set(fitted)  # 3
    if stale:
        failures.append("inherited buckets survived: %s" % sorted(stale))

    if manifest["encoder"] == "frozen":  # 4
        mine = load_file(os.path.join(ckpt, "model.safetensors"))
        theirs = load_file(os.path.join(base_dir(base), "model.safetensors"))
        moved = [k for k in mine if k.startswith("encoder.") and not torch.equal(mine[k].float(), theirs[k].float())]
        if moved:
            failures.append("frozen run changed %d encoder tensors, e.g. %s" % (len(moved), moved[0]))

    if service.question_hash(manifest["questions"]) != manifest["question_hash"]:  # 5
        failures.append("manifest question_hash does not match its questions")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--base", default="convaiinnovations/laya")
    args = ap.parse_args(argv)
    failures = check(args.checkpoint, args.base)
    for f in failures:
        print("FAIL", f)
    print("verify: %s" % ("ok" if not failures else "%d failure(s)" % len(failures)))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
