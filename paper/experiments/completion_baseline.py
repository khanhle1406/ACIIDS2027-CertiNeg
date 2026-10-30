"""A preregistered 80-percent output, using the same outer sample as v8."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certineg import group_inspection_cost, hypergeom_upper_rate, make_nested_sets
from run_experiments import make_candidate_population


def main() -> None:
    """Measure a fixed output without post-selection on audit labels."""
    _, z, risk, groups = make_candidate_population()
    sets = make_nested_sets(risk, [0.3, 0.45, 0.6, 0.8, 1.0])
    target = sets[3]
    costs = np.array([group_inspection_cost(y, p) for y, p in groups])
    rng = np.random.default_rng(20261026 + 99)
    rows = []
    for rep in range(2000):
        s = np.sort(rng.choice(sets[-1], size=800, replace=False))
        overlap = np.intersect1d(s, target)
        passed = (
            hypergeom_upper_rate(len(target), len(overlap), int(z[overlap].sum()), 0.05)
            <= 0.1
        )
        rows.append(
            {
                "rep": rep,
                "retention": 0.8 if passed else 0.0,
                "labels": int(costs[s].sum()),
                "violation": bool(passed and z[target].mean() > 0.1),
            }
        )
    out = ROOT.parent / "data/results/completion"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "fixed80_audits.csv", index=False)
    print(df.mean().to_string())


if __name__ == "__main__":
    main()
