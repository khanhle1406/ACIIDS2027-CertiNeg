import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from certineg import hypergeom_upper_rate

FRACS = np.array([0.30, 0.45, 0.60, 0.80, 1.00])
R = 2000
SEED = 20261026 + 50000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run epsilon-delta sensitivity sweep on AIR BAGS."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT.parent / "data/results",
        help="Directory containing sembench group data and to save sweep results",
    )
    args = parser.parse_args()
    OUT = args.results_dir
    OUT.mkdir(parents=True, exist_ok=True)

    D = np.load(OUT / "sembench_air_bags_groups.npz")
    z = D["z"].astype(np.int8)
    N = len(z)
    n = round(0.35 * N)
    ps = []
    prev = 0
    for f in FRACS:
        m = max(prev, min(N, round(float(f) * N)))
        ps.append(m)
        prev = m
    ps = np.array(ps, dtype=int)
    fdps = np.array([z[:m].mean() for m in ps])
    rng = np.random.default_rng(SEED)
    samples = [rng.choice(N, size=n, replace=False) for _ in range(R)]
    # sufficient stats per sample/prefix
    NS = np.zeros((R, len(ps)), dtype=int)
    XS = np.zeros_like(NS)
    for r, s in enumerate(samples):
        mark = np.zeros(N, np.int8)
        mark[s] = 1
        cn = np.cumsum(mark)
        cx = np.cumsum(mark * z)
        NS[r] = cn[ps - 1]
        XS[r] = cx[ps - 1]
    rows = []
    for eps in [0.05, 0.10, 0.20]:
        safe = np.where(fdps <= eps)[0]
        oracle = ps[safe[-1]] / N if len(safe) else 0.0
        for delta in [0.01, 0.05, 0.10]:
            rets = np.zeros(R)
            chosen = np.full(R, -1, dtype=int)
            vio = np.zeros(R, dtype=int)
            for r in range(R):
                k = -1
                for j, m in enumerate(ps):
                    u = hypergeom_upper_rate(
                        int(m), int(NS[r, j]), int(XS[r, j]), float(delta)
                    )
                    if u <= eps:
                        k = j
                    else:
                        break
                chosen[r] = k
                if k >= 0:
                    rets[r] = ps[k] / N
                    vio[r] = int(fdps[k] > eps)
            row = {
                "epsilon": eps,
                "delta": delta,
                "audit_n": n,
                "audit_fraction": n / N,
                "oracle_K5": oracle,
                "mean_retention": float(rets.mean()),
                "sd_retention": float(rets.std(ddof=1)),
                "violation_count": int(vio.sum()),
                "violation_rate": float(vio.mean()),
                "chosen_0": int((chosen < 0).sum()),
            }
            for j, f in enumerate(FRACS):
                row[f"chosen_{int(f * 100)}"] = int((chosen == j).sum())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "epsilon_delta_sweep_airbags.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
