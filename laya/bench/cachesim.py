"""Cache-aware bill for an agent trajectory: what a run would cost on real model tiers.

Local proxy tiers (Ollama) are free, so benchmark runs are billed by replaying their calls through
this simulator at real prices. Each model has its own prefix cache, and a switch between models or
a context rewrite pays a cache write again. That is the effect a model router has to price in
(docs research/08, research/10).

    python -m bench.cachesim demo        # self-test against hand-computed bills

A call is billed as: cache read for the part of its prompt that model still has cached, cache
write for the rest of the prompt (the harness puts a breakpoint at the end of every request), and
output. A cache entry lives for the TTL after its last read or write.
"""

import json
import sys
from dataclasses import dataclass, field

# $ per million tokens: base input, 5-minute write, 1-hour write, cache read, output.
# Source: platform.claude.com/docs/en/about-claude/pricing, read 2026-09-26.
PRICES = {
    "claude-fable-5-1": {"input": 10.0, "write_5m": 12.5, "write_1h": 20.0, "read": 0.25, "output": 50.0},
    "claude-opus-5-5": {"input": 4.0, "write_5m": 5.0, "write_1h": 8.0, "read": 0.20, "output": 20.0},
    "claude-opus-5": {"input": 5.0, "write_5m": 6.25, "write_1h": 10.0, "read": 0.50, "output": 25.0},
    "claude-sonnet-5": {"input": 2.0, "write_5m": 2.5, "write_1h": 4.0, "read": 0.20, "output": 10.0},
    "claude-haiku-4-5": {"input": 1.0, "write_5m": 1.25, "write_1h": 2.0, "read": 0.10, "output": 5.0},
}

# The local proxy ladder: which real tier each local model stands in for (design, laya-model-routing D7).
PROXY = {
    "qwen3:0.6b": "claude-haiku-4-5",
    "qwen3:4b": "claude-sonnet-5",
    "qwen3:8b": "claude-opus-5-5",
    "qwen3:14b": "claude-fable-5-1",
}

TTL_S = {"5m": 300, "1h": 3600}


@dataclass
class Call:
    """One model request. `prompt` is its total input tokens; `stable` is how many leading tokens
    are unchanged from the same model's previous request (the whole previous prompt when history
    is append-only; less after a compaction rewrote it)."""
    model: str
    t: float
    prompt: int
    output: int
    stable: int | None = None


@dataclass
class Bill:
    read: float = 0.0
    write: float = 0.0
    uncached: float = 0.0
    output: float = 0.0
    tokens: dict = field(default_factory=lambda: {"read": 0, "write": 0, "uncached": 0, "output": 0})

    @property
    def total(self):
        return self.read + self.write + self.uncached + self.output

    def as_dict(self):
        return {"total": round(self.total, 6), "read": round(self.read, 6), "write": round(self.write, 6),
                "uncached": round(self.uncached, 6), "output": round(self.output, 6), "tokens": self.tokens}


def price_of(model, prices=PRICES):
    return prices[PROXY.get(model, model)]


def bill(calls, ttl="5m", min_cacheable=1024, prices=PRICES):
    """Bill a sequence of calls. Caches are per real tier, keyed by the tier a proxy stands in for."""
    cache = {}  # tier -> (cached prefix length, expires at)
    last_prompt = {}  # tier -> previous prompt length, to default `stable` for append-only history
    out = Bill()
    for c in sorted(calls, key=lambda c: c.t):
        tier = PROXY.get(c.model, c.model)
        p = prices[tier]
        cached, expires = cache.get(tier, (0, -1.0))
        stable = c.stable if c.stable is not None else last_prompt.get(tier, 0)
        hit = min(cached, stable, c.prompt) if c.t <= expires else 0
        rest = c.prompt - hit
        if c.prompt >= min_cacheable:
            write, uncached = rest, 0
            cache[tier] = (c.prompt, c.t + TTL_S[ttl])
        else:
            write, uncached = 0, rest
        write_price = p["write_5m"] if ttl == "5m" else p["write_1h"]
        out.read += hit * p["read"] / 1e6
        out.write += write * write_price / 1e6
        out.uncached += uncached * p["input"] / 1e6
        out.output += c.output * p["output"] / 1e6
        for k, v in (("read", hit), ("write", write), ("uncached", uncached), ("output", c.output)):
            out.tokens[k] += v
        last_prompt[tier] = c.prompt
    return out


def demo():
    # 1. Append-only session on one tier: the first call writes, later calls read what is cached.
    calls = [Call("claude-sonnet-5", 0, 10_000, 500), Call("claude-sonnet-5", 30, 12_000, 400)]
    b = bill(calls)
    # call 1: write 10k at $2.50; call 2: read 10k at $0.20, write 2k at $2.50; outputs 900 at $10
    expect = 10_000 * 2.5e-6 + 10_000 * 0.2e-6 + 2_000 * 2.5e-6 + 900 * 10e-6
    assert abs(b.total - expect) < 1e-9, (b.total, expect)

    # 2. Switching tiers mid-session pays the whole context again on the new tier.
    stay = bill([Call("claude-opus-5-5", 0, 100_000, 300), Call("claude-opus-5-5", 20, 102_000, 300)])
    switch = bill([Call("claude-opus-5-5", 0, 100_000, 300), Call("claude-sonnet-5", 20, 102_000, 300)])
    assert switch.tokens["write"] == 202_000 and stay.tokens["write"] == 102_000
    # Opus 5.5 and Sonnet 5 read at the same $0.20, so the switch costs more than staying
    assert switch.total > stay.total, (switch.total, stay.total)

    # 3. The cache expires after the TTL: a call after 6 minutes writes again.
    cold = bill([Call("claude-haiku-4-5", 0, 5_000, 0), Call("claude-haiku-4-5", 400, 5_000, 0)])
    assert cold.tokens["read"] == 0 and cold.tokens["write"] == 10_000
    warm = bill([Call("claude-haiku-4-5", 0, 5_000, 0), Call("claude-haiku-4-5", 200, 5_000, 0)])
    assert warm.tokens["read"] == 5_000

    # 4. A compaction rewrite keeps only the stable prefix (system prompt and tools) cached.
    comp = bill([Call("claude-sonnet-5", 0, 80_000, 0), Call("claude-sonnet-5", 10, 20_000, 0, stable=4_000)])
    assert comp.tokens["read"] == 4_000 and comp.tokens["write"] == 80_000 + 16_000

    # 5. Proxies bill at their mapped real tier, and share that tier's cache.
    proxy = bill([Call("qwen3:4b", 0, 10_000, 500), Call("qwen3:4b", 30, 12_000, 400)])
    assert abs(proxy.total - b.total) < 1e-12

    # 6. Below the minimum cacheable length nothing is cached.
    tiny = bill([Call("claude-sonnet-5", 0, 500, 0), Call("claude-sonnet-5", 5, 600, 0)])
    assert tiny.tokens["uncached"] == 1_100 and tiny.tokens["read"] == 0
    print("cachesim self-test passed")


if __name__ == "__main__":
    if sys.argv[1:2] == ["demo"]:
        demo()
    else:
        calls = [Call(**c) for c in json.load(sys.stdin)]
        print(json.dumps(bill(calls).as_dict(), indent=2))
