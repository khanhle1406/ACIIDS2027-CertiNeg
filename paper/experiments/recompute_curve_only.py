import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "data/results"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from certineg import hypergeom_upper_rate
from run_experiments import make_candidate_population

EPS = 0.10
DELTA = 0.05
R = 2000
FR = np.array([0.30, 0.45, 0.60, 0.80, 1.0])
RATES = [0.05, 0.10, 0.20, 0.35, 0.50, 2 / 3]


def ps(N):
    o = []
    p = 0
    for f in FR:
        m = max(p, min(N, round(f * N)))
        o.append(m)
        p = m
    return np.array(o)


def sim(z, n, seed):
    p = ps(len(z))
    fd = np.array([z[:m].mean() for m in p])
    rng = np.random.default_rng(seed)
    ret = np.zeros(R)
    v = 0
    for r in range(R):
        s = rng.choice(len(z), n, replace=False)
        mark = np.zeros(len(z), np.int8)
        mark[s] = 1
        cn = np.cumsum(mark)
        cx = np.cumsum(mark * z)
        k = -1
        for i, m in enumerate(p):
            if (
                hypergeom_upper_rate(int(m), int(cn[m - 1]), int(cx[m - 1]), DELTA)
                <= EPS
            ):
                k = i
            else:
                break
        if k >= 0:
            ret[r] = p[k] / len(z)
            v += int(fd[k] > EPS)
    return ret.mean(), ret.std(ddof=1), v


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _sizes, z, risk, _groups = make_candidate_population()
    zc = z[np.argsort(risk, kind="mergesort")].astype(np.int8)
    za = np.load(OUT / "sembench_air_bags_groups.npz")["z"].astype(np.int8)
    rows = []
    for name, zv, seed in [
        ("Controlled", zc, 20261026 + 99),
        ("AIR BAGS", za, 20261026 + 50000),
    ]:
        for rate in RATES:
            n = max(1, min(len(zv), round(rate * len(zv))))
            m, sd, v = sim(zv, n, seed)
            p = ps(len(zv))
            og = max(
                [x / len(zv) for i, x in enumerate(p) if zv[:x].mean() <= EPS],
                default=0.0,
            )
            cs = np.cumsum(zv) / np.arange(1, len(zv) + 1)
            ok = np.where(
                (np.arange(1, len(zv) + 1) >= int(np.ceil(0.3 * len(zv)))) & (cs <= EPS)
            )[0]
            oc = (ok[-1] + 1) / len(zv) if len(ok) else 0.0
            rows.append((name, n, n / len(zv), m, sd, v, og, oc))
    df = pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "audit_n",
            "audit_fraction",
            "mean_retention",
            "sd_retention",
            "violation_count",
            "oracle_K5",
            "oracle_continuous",
        ],
    )
    df.to_csv(OUT / "retention_audit_curve_r6.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
