import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from certineg import hypergeom_upper_rate

EPS = 0.10
DELTA = 0.05
R = 2000
SEED = 20261026
FRACS = np.array([0.30, 0.45, 0.60, 0.80, 1.0])


def safe_name(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def prefix_sizes(N):
    out = []
    prev = 0
    for f in FRACS:
        m = max(1, min(N, round(float(f) * N)))
        m = max(prev, m)
        out.append(m)
        prev = m
    return np.array(out, dtype=int)


def select_fixed(z, sample):
    N = len(z)
    ps = prefix_sizes(N)
    mark = np.zeros(N, np.int8)
    mark[sample] = 1
    cn = np.cumsum(mark)
    cx = np.cumsum(mark * z)
    k = -1
    for i, m in enumerate(ps):
        n = int(cn[m - 1])
        x = int(cx[m - 1])
        if hypergeom_upper_rate(int(m), n, x, DELTA) <= EPS:
            k = i
        else:
            break
    return k, ps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-sequence audits for all SemBench screened predicates."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT.parent / "data/results",
        help="Directory containing screened predicates and to save audit results",
    )
    args = parser.parse_args()
    OUT = args.results_dir
    OUT.mkdir(parents=True, exist_ok=True)

    screen = pd.read_csv(OUT / "sembench_predicate_screen_r8.csv")
    rows = []
    for j, row in screen.iterrows():
        t = row.predicate
        a = np.load(OUT / f"sembench_screen_{safe_name(t)}_groups.npz")
        z = a["z"].astype(np.int8)
        N = len(z)
        n = max(1, round(0.35 * N))
        ps = prefix_sizes(N)
        fdps = np.array([z[:m].mean() for m in ps])
        seed_map = {
            "AIR BAGS": SEED + 50000,
            "ENGINE AND ENGINE COOLING": SEED + 50001,
            "VISIBILITY/WIPER": SEED + 50002,
            "EXTERIOR LIGHTING": SEED + 50003,
        }
        rng = np.random.default_rng(seed_map.get(t, SEED + 70000 + j))
        rets = np.zeros(R)
        nonempty = np.zeros(R, dtype=np.int8)
        vio = np.zeros(R, dtype=np.int8)
        for r in range(R):
            s = rng.choice(N, size=min(n, N), replace=False)
            k, _ = select_fixed(z, s)
            if k >= 0:
                nonempty[r] = 1
                rets[r] = ps[k] / N
                vio[r] = int(fdps[k] > EPS)
        rows.append(
            {
                "predicate": t,
                "candidate_groups": N,
                "audit_n": n,
                "audit_fraction": n / N,
                "candidate_fdp": float(z.mean()),
                "oracle_K5": float(row.oracle_K5),
                "nonempty_cert_rate": float(nonempty.mean()),
                "mean_retention": float(rets.mean()),
                "sd_retention": float(rets.std(ddof=1)),
                "unsafe_return_rate": float(vio.mean()),
                "unsafe_return_count": int(vio.sum()),
            }
        )
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "sembench_all_predicate_audits_r8.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
