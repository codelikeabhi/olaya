"""Harbor agent adapter for Olaya.

olaya-rename:keep-file: Harbor's own agent is named OpenCode and runs a command called `opencode`;
those names are Harbor's contract. Olaya's build paths below use Olaya's names.

    PYTHONPATH=<olaya>/laya harbor run ... -a bench.harbor_olaya:Olaya \\
        --ae OLAYA_LAYA_ENABLED=1 --ae OLAYA_LAYA_SHADOW=1 \\
        --ae OLAYA_LAYA_URL=http://host.docker.internal:8731

Olaya is OpenCode plus a decision layer, so the adapter reuses Harbor's OpenCode agent
wholesale and changes one thing: after the stock install, it overwrites the installed
`opencode` with our own build. Run command, flags, trajectory parsing and timeouts are
inherited unchanged, which is what makes an Olaya-vs-OpenCode comparison a clean ablation.

The binary comes from `packages/olaya/dist` (built with `bun run script/build.ts
--skip-embed-web-ui`), or from $OLAYA_DIST. The decision layer is configured purely through
agent env (`--ae`), exactly as a user would configure it.
"""

import os
from pathlib import Path
from typing import override

from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment

REPO = Path(__file__).resolve().parents[2]
DIST = Path(os.environ.get("OLAYA_DIST", REPO / "packages" / "olaya" / "dist"))


def binary_for(libc: str, arch: str) -> Path:
    """Pick the build matching the container.

    x64 uses the baseline (no-AVX2) build: task images run under x86 emulation on Apple
    Silicon, where AVX2 support is not guaranteed, and speed is irrelevant next to an LLM step.
    """
    arch = "arm64" if arch in ("aarch64", "arm64") else "x64"
    parts = ["olaya", "linux", arch]
    if arch == "x64":
        parts.append("baseline")
    if libc == "musl":
        parts.append("musl")
    path = DIST / "-".join(parts) / "bin" / "olaya"
    if not path.exists():
        raise FileNotFoundError("no Olaya build at %s; run script/build.ts first" % path)
    return path


class Olaya(OpenCode):
    @staticmethod
    @override
    def name() -> str:
        return "olaya"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        # Run the stock install first, then swap the binary. The stock install also puts Node
        # on PATH; skipping it would give the agent a different toolbox than the OpenCode arm
        # and turn the comparison into an environment ablation instead of a harness one.
        await super().install(environment)
        probe = await self.exec_as_agent(
            environment,
            command=(
                "[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; "
                "(ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]) && echo musl || echo glibc; "
                "uname -m; readlink -f $(command -v opencode)"
            ),
        )
        libc, arch, target = (probe.stdout or "").split()[:3]
        # npm's `opencode` is a node launcher script around a platform package; replacing the
        # resolved file keeps every PATH lookup pointing at our build.
        await environment.upload_file(binary_for(libc, arch), "/tmp/olaya-opencode")
        await self.exec_as_root(
            environment,
            command="install -m 755 /tmp/olaya-opencode %s && rm /tmp/olaya-opencode" % target,
        )
        check = await self.exec_as_agent(
            environment, command="[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; opencode --version"
        )
        if check.return_code != 0:
            raise RuntimeError("Olaya binary failed to start: %s" % (check.stdout or check.stderr))

    @override
    def get_version_command(self) -> str | None:
        return "[ -f ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; opencode --version"
