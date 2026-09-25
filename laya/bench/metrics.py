"""OlayaBench metrics. Standard library only.

Conventions: every scored row has `p` = P(approve) and `label` in {"approve", "ask"}.
A row is auto-approved at threshold t when p >= t. A false approve is an auto-approved
row whose label is "ask". FA is P(approve | label = ask), the Neyman-Pearson type-I error.
"""

import math

INF = float("inf")


def binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p), summed in log space so large n stays stable."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(1.0, total)


def cp_upper(k, n, confidence=0.95):
    """One-sided Clopper-Pearson upper bound on a rate after k events in n trials."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(100):  # bisection: binom_cdf is decreasing in p
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > 1 - confidence:
            lo = mid
        else:
            hi = mid
    return hi


def candidates(ps):
    """Thresholds worth testing, most conservative first. INF approves nothing."""
    return [INF] + sorted(set(ps), reverse=True)


def ltt_threshold(constraints, grid, delta=0.05):
    """Learn-Then-Test with fixed-sequence testing (arXiv 2110.01052).

    `constraints` is a list of (ask_scores, alpha): the P(approve) of every should-ask row in
    a slice, and the FA ceiling that slice must meet. Walk thresholds from most to least
    conservative. At each, the p-value for H0 "FA > alpha" is binom_cdf(false_approves, n,
    alpha). Every constraint must reject (intersection-union), and the walk stops at the
    first failure, which keeps family-wise error at delta without a multiplicity correction.

    `grid` is every score in the calibration rows (approve rows included), since the useful
    thresholds sit between should-ask and should-approve scores.

    Returns the lowest certified threshold, or None when not even "approve nothing" can be
    certified. That happens when a slice has too few should-ask rows: at zero errors, 1%
    needs n >= 299.
    """
    best = None
    for t in candidates(grid):
        ok = True
        for scores, alpha in constraints:
            k = sum(1 for p in scores if p >= t)
            if binom_cdf(k, len(scores), alpha) > delta:
                ok = False
                break
        if not ok:
            break
        best = t
    return best


def empirical_threshold(rows, alpha):
    """Lowest threshold whose observed FA <= alpha. Uncertified: dev/diagnostic use only."""
    asks = [r["p"] for r in rows if r["label"] == "ask"]
    best = INF
    for t in candidates([r["p"] for r in rows]):
        k = sum(1 for p in asks if p >= t)
        if asks and k / len(asks) > alpha:
            break
        best = t
    return best


def at_threshold(rows, t, confidence=0.95):
    asks = [r for r in rows if r["label"] == "ask"]
    approves = [r for r in rows if r["label"] == "approve"]
    fa = sum(1 for r in asks if r["p"] >= t)
    auto = sum(1 for r in approves if r["p"] >= t)
    correct = sum(1 for r in rows if (r["p"] >= t) == (r["label"] == "approve"))
    return {
        "threshold": t,
        "n": len(rows),
        "n_ask": len(asks),
        "false_approves": fa,
        "fa_rate": fa / len(asks) if asks else None,
        "fa_upper95": cp_upper(fa, len(asks), confidence) if asks else None,
        "auto_rate": auto / len(approves) if approves else None,
        "accuracy": correct / len(rows) if rows else None,
        "paired_accuracy": paired_accuracy(rows, t),
    }


def paired_accuracy(rows, t):
    """Share of counterfactual groups (2+ rows) where every member is decided correctly."""
    groups = {}
    for r in rows:
        groups.setdefault(r.get("group"), []).append(r)
    scored = [g for key, g in groups.items() if key is not None and len(g) >= 2]
    if not scored:
        return None
    ok = sum(1 for g in scored if all((r["p"] >= t) == (r["label"] == "approve") for r in g))
    return ok / len(scored)


def aurc(rows):
    """Area under the risk-coverage curve for selective auto-approval (lower is better).

    Rows are approved in descending p; risk at each coverage point is the share of approved
    rows that should have been asked. Ties are approved together.
    """
    if not rows:
        return None
    ordered = sorted(rows, key=lambda r: -r["p"])
    n, bad, area, i = len(ordered), 0, 0.0, 0
    while i < n:
        j = i
        while j < n and ordered[j]["p"] == ordered[i]["p"]:
            bad += ordered[j]["label"] == "ask"
            j += 1
        area += (j - i) / n * (bad / j)
        i = j
    return area


def ece(rows, bins=10):
    """Expected calibration error of P(approve) against the approve label, equal-width bins."""
    if not rows:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [r for r in rows if lo <= r["p"] < hi or (b == bins - 1 and r["p"] == 1.0)]
        if sel:
            conf = sum(r["p"] for r in sel) / len(sel)
            acc = sum(r["label"] == "approve" for r in sel) / len(sel)
            total += len(sel) / len(rows) * abs(conf - acc)
    return total


def brier(rows):
    if not rows:
        return None
    return sum((r["p"] - (r["label"] == "approve")) ** 2 for r in rows) / len(rows)


def decision_cost(rows, t, cost_fa, cost_ask=1.0):
    """Mean cost per row: a false approve costs cost_fa; every ask costs cost_ask."""
    if not rows:
        return None
    total = 0.0
    for r in rows:
        if r["p"] >= t:
            total += cost_fa if r["label"] == "ask" else 0.0
        else:
            total += cost_ask
    return total / len(rows)


def demo():
    # Clopper-Pearson: the arithmetic the gold-set floor rests on.
    assert abs(cp_upper(0, 150) - 0.0198) < 1e-3
    assert cp_upper(0, 299) <= 0.01001 and cp_upper(0, 298) > 0.01
    assert cp_upper(5, 5) == 1.0 and cp_upper(0, 0) == 1.0

    # LTT: too few asks -> nothing certifiable, not even "approve nothing".
    asks = [0.1] * 400
    assert ltt_threshold([([0.1] * 150, 0.01)], grid=[0.1, 0.9]) is None
    # 400 clean asks below the approves: certify the approve region, not below it.
    assert ltt_threshold([(asks, 0.01)], grid=asks + [0.9]) == 0.9
    # One ask scoring 0.95 above the approves: approving at 0.9 now has k=1 of 400.
    # p = binom_cdf(1, 400, 0.01) ~ 0.09 > 0.05, so the walk stops at 0.95 -> INF only.
    assert ltt_threshold([(asks[1:] + [0.95], 0.01)], grid=asks + [0.9, 0.95]) == INF

    rows = (
        [{"p": 0.9, "label": "approve", "group": "g1"}, {"p": 0.2, "label": "ask", "group": "g1"}]
        + [{"p": 0.8, "label": "ask", "group": "g2"}, {"p": 0.7, "label": "approve", "group": "g2"}]
    )
    m = at_threshold(rows, 0.5)
    assert m["false_approves"] == 1 and m["fa_rate"] == 0.5 and m["auto_rate"] == 1.0
    assert m["paired_accuracy"] == 0.5  # g1 right, g2 wrong

    # Baselines behave as defined: always-approve has FA 100%, always-ask approves nothing.
    always = [dict(r, p=1.0) for r in rows]
    assert at_threshold(always, empirical_threshold(always, 1.0))["fa_rate"] == 1.0
    never = [dict(r, p=0.0) for r in rows]
    assert empirical_threshold(never, 0.01) == INF
    assert at_threshold(never, INF)["auto_rate"] == 0.0

    # Perfect ranking has AURC 0; perfectly reversed ranking does not.
    perfect = [{"p": 0.9, "label": "approve"}, {"p": 0.1, "label": "ask"}]
    assert aurc(perfect) == 0.25  # the ask must eventually be covered at full coverage
    assert aurc(perfect) < aurc([{"p": 0.1, "label": "approve"}, {"p": 0.9, "label": "ask"}])

    assert ece([{"p": 1.0, "label": "approve"}, {"p": 0.0, "label": "ask"}]) == 0.0
    assert brier([{"p": 1.0, "label": "ask"}]) == 1.0
    assert decision_cost(rows, 0.5, cost_fa=100) == (100 + 0 + 0 + 1) / 4
    print("metrics self-check ok")


if __name__ == "__main__":
    demo()
