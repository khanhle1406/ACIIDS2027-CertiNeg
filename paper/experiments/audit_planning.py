from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from certineg import hypergeom_upper_rate


def cert_power(N, M, n, epsilon=0.10, delta=0.05):
    lo = max(0, n - (N - M))
    hi = min(n, M)
    xs = np.arange(lo, hi + 1)
    pmf = hypergeom.pmf(xs, N, M, n)
    ok = np.array([hypergeom_upper_rate(N, n, int(x), delta) <= epsilon for x in xs])
    return float(pmf[ok].sum())


def min_n(N, q_design, power_target, epsilon=0.10, delta=0.05):
    M = int(np.floor(q_design * N))
    for n in range(1, N + 1):
        p = cert_power(N, M, n, epsilon, delta)
        if p >= power_target:
            return n, p, M
    return N, cert_power(N, M, N, epsilon, delta), M


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Prospective fixed-set certification power."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT.parent / "data/results",
        help="Directory to save output results",
    )
    args = parser.parse_args()
    OUT = args.results_dir
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    # AIR BAGS original 80% prefix has 874 groups. q=.07 is a prospective design point,
    # deliberately below epsilon rather than using held-out labels as an input to planning.
    N = 874
    for q in [0.05, 0.06, 0.07, 0.08]:
        for target in [0.50, 0.80, 0.90]:
            n, p, M = min_n(N, q, target)
            rows.append(
                {
                    "N": N,
                    "epsilon": 0.10,
                    "delta": 0.05,
                    "q_design": q,
                    "power_target": target,
                    "M_design": M,
                    "min_n": n,
                    "achieved_power": p,
                    "sample_fraction": n / N,
                }
            )
    out_file = OUT / "audit_planning_airbags.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
