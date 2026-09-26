"""Real coding-agent actions from public trajectories, as unlabelled OlayaBench-format items.

    python -m train.replay --out ~/.local/share/olaya/laya/data/replay-swe-agent.jsonl --trajectories 300

Source: nebius/SWE-agent-trajectories (CC-BY-4.0). Each trajectory is an agent solving a real
GitHub issue in a real repository. Its shell commands, paired with the issue as the task, are
the realistic distribution the approve decision has to handle; synthetic data cannot provide
that. Attribution is recorded on every item; the derived data stays local and is never
committed.

Items are written WITHOUT a label. The teacher (train.teacher) labels them, and they must
never enter a certification set, which is human-only.
"""

import argparse
import json
import re

SOURCE = "nebius/SWE-agent-trajectories"
LICENSE = "CC-BY-4.0"
# SWE-agent's own interface commands. In Olaya these are read-only tools, not permission
# requests, so they carry no approve decision.
INTERFACE = {"open", "goto", "scroll_up", "scroll_down", "create", "edit", "search_file", "search_dir",
             "find_file", "submit", "end_of_edit", "exit", "exit_forfeit"}
FENCE = re.compile(r"```(?:[a-z]*\n)?(.*?)```", re.S)


def task_of(user_text):
    """The issue's opening lines: enough to judge scope, short enough to leave room in the state."""
    text = user_text.split("ISSUE:", 1)[-1]
    text = text.split("INSTRUCTIONS:", 1)[0]
    lines = [l.strip() for l in text.strip().splitlines() if l.strip() and not l.strip().startswith("```")]
    return " ".join(lines)[:400]


def commands_of(ai_text):
    blocks = FENCE.findall(ai_text or "")
    if not blocks:
        return []
    cmd = blocks[-1].strip()
    if not cmd or cmd.split()[0] in INTERFACE:
        return []
    return [cmd]


def items_from(row, per_trajectory):
    traj = row.get("trajectory") or []
    users = [m for m in traj if m.get("role") == "user"]
    if not users:
        return []
    task = task_of(users[0].get("text") or "")
    out = []
    seen = set()
    for m in traj:
        if m.get("role") != "ai":
            continue
        for cmd in commands_of(m.get("text") or ""):
            if cmd in seen or len(cmd) > 600:
                continue
            seen.add(cmd)
            head = cmd.split()[0]
            out.append({
                "id": "R-%s-%d" % (row["instance_id"], len(out)),
                "track": "approve",
                "group": "replay-" + row["instance_id"],
                "family": "replay",
                "route": "shell",
                "request": {"permission": "bash", "patterns": ["%s *" % head], "metadata": {"command": cmd}},
                "context": {"task": task, "cwd": "/work/repo"},
                "label": None,
                "severity": None,
                "provenance": {"source": "replay:" + SOURCE, "license": LICENSE, "labeler": "none",
                               "instance_id": row["instance_id"], "model": row.get("model_name")},
            })
            if len(out) >= per_trajectory:
                return out
    return out


def main(argv=None):
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trajectories", type=int, default=300)
    ap.add_argument("--per-trajectory", type=int, default=4)
    args = ap.parse_args(argv)
    ds = load_dataset(SOURCE, split="train", streaming=True)
    repos, n, items = set(), 0, []
    for row in ds:
        repo = row["instance_id"].rsplit("-", 1)[0]
        if repo in repos and len(repos) < args.trajectories // 2:
            continue  # favour breadth across repositories early on
        repos.add(repo)
        items += items_from(row, args.per_trajectory)
        n += 1
        if n >= args.trajectories:
            break
    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print("%d actions from %d trajectories across %d repositories -> %s" % (len(items), n, len(repos), args.out))


def demo():
    row = {"instance_id": "acme__widgets-12", "model_name": "m", "trajectory": [
        {"role": "system", "text": "sys"},
        {"role": "user", "text": "ISSUE:\nWidget crashes on empty input\nsteps...\nINSTRUCTIONS:\nfix it"},
        {"role": "ai", "text": "Let me look.\n```\nfind_file \"w.py\"\n```"},
        {"role": "ai", "text": "Run it.\n```\npython reproduce.py\n```"},
        {"role": "ai", "text": "Again.\n```\npython reproduce.py\n```"},
        {"role": "ai", "text": "Clean.\n```\nrm reproduce.py\n```"},
    ]}
    items = items_from(row, 4)
    assert [i["request"]["metadata"]["command"] for i in items] == ["python reproduce.py", "rm reproduce.py"], items
    assert items[0]["context"]["task"].startswith("Widget crashes on empty input")
    assert all(i["label"] is None and i["group"] == "replay-acme__widgets-12" for i in items)
    print("replay self-check ok")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--demo"]:
        demo()
    else:
        main()
