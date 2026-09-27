"""Track C items from EvalPlus (HumanEval+ 164, MBPP+ 378): many small tasks with strong tests, for
the per-tier outcomes the routing head needs (gate G4 was short of data at 32-80 Exercism items).

    python -m bench.evalplus build            # -> track-c/evalplus/<id>/, track-c/items-evalplus.jsonl
    python -m bench.evalplus image            # the sandbox image with numpy, which the tests import
    python -m bench.evalplus demo             # self-test: a canonical solution passes, the stub fails

Each item is a folder with a stub `solution.py` (HumanEval+: the prompt, signature and docstring;
MBPP+: the reference signature with no body), the suite's own test code in `check_data.py` (plus
inputs and expected results), and `test_solution.py`, which runs that code against the solution.
Items carry their own sandbox image; Exercism items keep the plain one.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

from . import route

HOME = os.path.join(route.HOME, "evalplus")
ITEMS = os.path.join(route.HOME, "items-evalplus.jsonl")
IMAGE = "olaya-trackc:py312-np"
# the reference solution fails its own tests in this sandbox (found by checking all 542): left out
BROKEN = {"he-32", "mbpp-590"}
HERE = os.path.dirname(os.path.abspath(__file__))

RUNNER = '''import pathlib

import solution


def test_solution():
    # the suite's own checks, run with the solution's names in scope
    code = pathlib.Path(__file__).with_name("check_data.py").read_text()
    exec(compile(code, "check_data.py", "exec"), {"__name__": "check_data", **vars(solution)})
'''


def humaneval(x):
    name = x["entry_point"]
    return {
        "id": "he-" + x["task_id"].split("/")[1], "difficulty": 1,
        "files": {"solution.py": x["prompt"], "check_data.py": x["test"] + f"\ncheck({name})\n"},
        "instructions": f"Complete the function `{name}` in solution.py so it does what its docstring says.",
    }


def mbpp(x):
    name = re.search(r"assertion\((\w+)\(\*inp\)", x["test"]).group(1)
    # tolerant of `def f (x) :`, annotations and a return type
    signature = re.search(rf"^def {name}\s*\(.*?\)\s*(?:->\s*[^:]+)?\s*:", x["code"], re.M | re.S).group(0)
    return {
        "id": f"mbpp-{x['task_id']}", "difficulty": 1,
        "files": {"solution.py": "\n".join(x.get("test_imports") or []) + f"\n\n{signature}\n    raise NotImplementedError\n",
                  "check_data.py": x["test"]},
        "instructions": f"{x['prompt']}\n\nImplement it as `{name}` in solution.py. For example: {x['test_list'][0]}",
    }


def build():
    from datasets import load_dataset

    specs = [humaneval(x) for x in load_dataset("evalplus/humanevalplus", split="test")] + \
            [mbpp(x) for x in load_dataset("evalplus/mbppplus", split="test")]
    specs = [spec for spec in specs if spec["id"] not in BROKEN]
    with open(ITEMS, "w") as out:
        for spec in specs:
            d = os.path.join(HOME, spec["id"])
            os.makedirs(d, exist_ok=True)
            for name, text in {**spec["files"], "test_solution.py": RUNNER}.items():
                open(os.path.join(d, name), "w").write(text)
            out.write(json.dumps({"id": spec["id"], "difficulty": spec["difficulty"], "dir": d, "image": IMAGE,
                                  "solution": ["solution.py"], "test": ["test_solution.py", "check_data.py"],
                                  "instructions": spec["instructions"]}) + "\n")
    print(f"{len(specs)} items -> {ITEMS}")


def image():
    dockerfile = f"FROM {route.IMAGE}\nUSER root\nRUN pip install --no-cache-dir numpy==2.1.2\nUSER agent\n"
    subprocess.run(["docker", "build", "-t", IMAGE, "-"], input=dockerfile.encode(), check=True)


def self_test():
    """A canonical solution passes its item's tests and the stub fails them, for one item of each suite."""
    from datasets import load_dataset

    ok = True
    for x, spec, answer in [
        (x := load_dataset("evalplus/humanevalplus", split="test")[0], humaneval(x), x["prompt"] + x["canonical_solution"]),
        (y := load_dataset("evalplus/mbppplus", split="test")[0], mbpp(y), y["code"]),
    ]:
        for body, want in ((spec["files"]["solution.py"], 1), (answer, 0)):
            with tempfile.TemporaryDirectory() as d:
                for name, text in {**spec["files"], "solution.py": body, "test_solution.py": RUNNER}.items():
                    open(os.path.join(d, name), "w").write(text)
                code = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=d, capture_output=True).returncode
                ok &= (code == 0) == (want == 0)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "image", "demo"):
        sub.add_parser(c)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build()
    elif a.cmd == "image":
        image()
    else:
        print("evalplus self-test", "passed" if self_test() else "FAILED")


if __name__ == "__main__":
    main()
