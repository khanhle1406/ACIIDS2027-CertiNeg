from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.stats import hypergeom


@lru_cache(maxsize=200000)
def hypergeom_upper_count(N: int, n: int, x: int, alpha: float) -> int:
    """Exact one-sided (1-alpha) upper confidence bound for M defects in N.

    A simple random sample without replacement of n units yields x defects.
    The bound is obtained by inverting the lower-tail hypergeometric test:
    retain M whenever P_M[X <= x] > alpha and return the largest retained M.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    if N < 0 or n < 0 or n > N or x < 0 or x > n:
        raise ValueError("invalid N,n,x")
    if N == 0:
        return 0
    if n == 0:
        return N

    # M must be large enough to permit x successes and small enough to permit
    # n-x sampled nondefects.
    lo = x
    hi = N - (n - x)
    best = lo
    # hypergeom.cdf(x; N,M,n) is nonincreasing in M.
    while lo <= hi:
        mid = (lo + hi) // 2
        p = hypergeom.cdf(x, N, mid, n)
        if p > alpha:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return int(best)


def hypergeom_upper_rate(N: int, n: int, x: int, alpha: float) -> float:
    if N == 0:
        return 0.0
    return hypergeom_upper_count(N, n, x, alpha) / N


def exact_coverage_min(
    N: int, n: int, alpha: float
) -> tuple[float, tuple[int, int] | None]:
    """Enumerate all M and x to compute minimum exact coverage P_M(M<=U(X))."""
    min_cov = 1.0
    arg = None
    for M in range(N + 1):
        xmin = max(0, n - (N - M))
        xmax = min(n, M)
        cov = 0.0
        for x in range(xmin, xmax + 1):
            u = hypergeom_upper_count(N, n, x, alpha)
            if M <= u:
                cov += hypergeom.pmf(x, N, M, n)
        if cov < min_cov:
            min_cov = cov
            arg = (M, n)
    return float(min_cov), arg


@dataclass(frozen=True)
class NestedCertificate:
    chosen_index: int | None
    chosen_size: int
    chosen_upper_rate: float | None
    upper_rates: tuple[float, ...]
    sample_counts: tuple[int, ...]
    defect_counts: tuple[int, ...]


def certify_nested_from_largest_sample(
    sets: Sequence[np.ndarray],
    largest_sample: np.ndarray,
    z: np.ndarray,
    epsilon: float,
    delta: float,
) -> NestedCertificate:
    """Simultaneous Bonferroni certificate for fixed nested candidate sets.

    `sets` must be nested subsets of the largest set. `largest_sample` is an
    SRSWOR from the largest set. Conditional on its random intersection size,
    the intersection with each fixed subset is SRSWOR from that subset.
    """
    K = len(sets)
    if K == 0:
        raise ValueError("sets must be nonempty")
    alpha = delta / K
    sample_set = set(map(int, largest_sample.tolist()))
    upper, ns, xs = [], [], []
    for A in sets:
        idx = np.asarray(A, dtype=int)
        observed = np.array([g for g in idx if int(g) in sample_set], dtype=int)
        n = len(observed)
        x = int(z[observed].sum()) if n else 0
        u = hypergeom_upper_rate(len(idx), n, x, alpha) if len(idx) else 0.0
        upper.append(float(u))
        ns.append(n)
        xs.append(x)

    certifiable = [k for k, u in enumerate(upper) if u <= epsilon]
    if not certifiable:
        return NestedCertificate(None, 0, None, tuple(upper), tuple(ns), tuple(xs))
    k = max(certifiable)
    return NestedCertificate(
        k, len(sets[k]), upper[k], tuple(upper), tuple(ns), tuple(xs)
    )


def certify_nested_weighted(
    sets: Sequence[np.ndarray],
    largest_sample: np.ndarray,
    z: np.ndarray,
    epsilon: float,
    delta: float,
    weights: Sequence[float],
) -> NestedCertificate:
    """Simultaneous weighted-Bonferroni certificate for fixed candidate sets.

    The nonnegative weights are fixed before audit labels are revealed and
    normalized to sum to one. Set k receives level delta * weight[k].
    """
    K = len(sets)
    w = np.asarray(weights, dtype=float)
    if K == 0 or len(w) != K or np.any(w <= 0):
        raise ValueError("weights must be positive and match sets")
    w = w / w.sum()
    sample_set = set(map(int, largest_sample.tolist()))
    upper, ns, xs = [], [], []
    for A, wk in zip(sets, w):
        idx = np.asarray(A, dtype=int)
        observed = np.array([g for g in idx if int(g) in sample_set], dtype=int)
        n = len(observed)
        x = int(z[observed].sum()) if n else 0
        u = hypergeom_upper_rate(len(idx), n, x, float(delta * wk)) if len(idx) else 0.0
        upper.append(float(u))
        ns.append(n)
        xs.append(x)
    certifiable = [k for k, u in enumerate(upper) if u <= epsilon]
    if not certifiable:
        return NestedCertificate(None, 0, None, tuple(upper), tuple(ns), tuple(xs))
    k = max(certifiable)
    return NestedCertificate(
        k, len(sets[k]), upper[k], tuple(upper), tuple(ns), tuple(xs)
    )


def certify_nested_fixed_sequence(
    sets: Sequence[np.ndarray],
    largest_sample: np.ndarray,
    z: np.ndarray,
    epsilon: float,
    delta: float,
) -> NestedCertificate:
    """Ordered fixed-sequence certificate at per-test level delta.

    Tests are attempted in the predeclared order. The procedure stops at the
    first set that cannot be certified and returns the largest consecutively
    certified prefix. Fixed-sequence testing controls the probability of any
    false certification at delta without a Bonferroni delta/K penalty.
    """
    if not sets:
        raise ValueError("sets must be nonempty")
    sample_set = set(map(int, largest_sample.tolist()))
    upper, ns, xs = [], [], []
    chosen = None
    stopped = False
    for k, A in enumerate(sets):
        idx = np.asarray(A, dtype=int)
        observed = np.array([g for g in idx if int(g) in sample_set], dtype=int)
        n = len(observed)
        x = int(z[observed].sum()) if n else 0
        u = hypergeom_upper_rate(len(idx), n, x, delta) if len(idx) else 0.0
        upper.append(float(u))
        ns.append(n)
        xs.append(x)
        if not stopped and u <= epsilon:
            chosen = k
        elif not stopped:
            stopped = True
    if chosen is None:
        return NestedCertificate(None, 0, None, tuple(upper), tuple(ns), tuple(xs))
    return NestedCertificate(
        chosen, len(sets[chosen]), upper[chosen], tuple(upper), tuple(ns), tuple(xs)
    )


def stratified_upper_rate(
    N_h: Sequence[int], n_h: Sequence[int], x_h: Sequence[int], alpha: float
) -> float:
    """Conservative simultaneous stratum-wise finite-population upper bound."""
    H = len(N_h)
    if not (len(n_h) == len(x_h) == H) or H == 0:
        raise ValueError("stratum vectors must have equal positive length")
    totalN = sum(N_h)
    if totalN == 0:
        return 0.0
    a = alpha / H
    upper_count = 0
    for N, n, x in zip(N_h, n_h, x_h):
        upper_count += hypergeom_upper_count(int(N), int(n), int(x), a)
    return upper_count / totalN


def group_inspection_cost(
    labels: np.ndarray,
    proxy_scores: np.ndarray | None,
    rng: np.random.Generator | None = None,
) -> int:
    """Trusted-label cost for exact group truth.

    Stop at the first positive witness. If no positive exists, all records must
    be inspected. If proxy_scores is None, use a random order.
    """
    labels = np.asarray(labels, dtype=int)
    m = len(labels)
    if m == 0:
        return 0
    if proxy_scores is None:
        if rng is None:
            rng = np.random.default_rng()
        order = rng.permutation(m)
    else:
        order = np.argsort(-np.asarray(proxy_scores, dtype=float), kind="mergesort")
    for rank, j in enumerate(order, start=1):
        if labels[j] == 1:
            return rank
    return m


def make_nested_sets(
    risk_score: np.ndarray, fractions: Iterable[float]
) -> list[np.ndarray]:
    order = np.argsort(np.asarray(risk_score), kind="mergesort")
    N = len(order)
    sets = []
    prev = 0
    for f in fractions:
        k = max(prev, min(N, round(float(f) * N)))
        sets.append(np.sort(order[:k]))
        prev = k
    return sets
