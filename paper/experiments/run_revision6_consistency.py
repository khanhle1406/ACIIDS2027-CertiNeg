import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "data/results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from certineg import hypergeom_upper_rate
from run_experiments import make_candidate_population

EPS = 0.10
DELTA = 0.05
R = 2000
BASE = 20311026
FR5 = np.array([0.30, 0.45, 0.60, 0.80, 1.0])
RATES = np.array([0.05, 0.10, 0.20, 0.35, 0.50, 2 / 3])


def psizes(N, fr):
    out = []
    p = 0
    for f in fr:
        m = max(p, min(N, round(float(f) * N)))
        out.append(m)
        p = m
    return np.array(out, int)


def fixed(s, z, ps):
    mark = np.zeros(len(z), np.int8)
    mark[s] = 1
    cn = np.cumsum(mark)
    cx = np.cumsum(mark * z)
    k = -1
    for i, m in enumerate(ps):
        if hypergeom_upper_rate(int(m), int(cn[m - 1]), int(cx[m - 1]), DELTA) <= EPS:
            k = i
        else:
            break
    return k


def sim(z, fr, n, seed, R=R):
    ps = psizes(len(z), fr)
    fd = np.array([z[:m].mean() for m in ps])
    rng = np.random.default_rng(seed)
    ret = np.zeros(R)
    vio = 0
    for r in range(R):
        s = rng.choice(len(z), size=n, replace=False)
        k = fixed(s, z, ps)
        if k >= 0:
            ret[r] = ps[k] / len(z)
            vio += int(fd[k] > EPS)
    return ret.mean(), ret.std(ddof=1), vio


def oracle(z, minf=0.30):
    N = len(z)
    cs = np.cumsum(z) / np.arange(1, N + 1)
    ok = np.where((np.arange(1, N + 1) >= int(np.ceil(minf * N))) & (cs <= EPS))[0]
    return (ok[-1] + 1) / N if len(ok) else 0.0


def gor(z, fr):
    ps = psizes(len(z), fr)
    v = [m / len(z) for i, m in enumerate(ps) if z[:m].mean() <= EPS]
    return max(v, default=0.0)


def main() -> None:
    _sizes, z, risk, _groups = make_candidate_population()
    order = np.argsort(risk, kind="mergesort")
    zc = z[order].astype(np.int8)
    a = np.load(OUT / "sembench_air_bags_groups.npz")
    za = a["z"].astype(np.int8)
    rr = a["risk"].astype(float)
    rows = []
    for di, (name, zv) in enumerate([("Controlled", zc), ("AIR BAGS", za)]):
        for rate in RATES:
            n = max(1, min(len(zv), round(rate * len(zv))))
            mr, sd, v = sim(
                zv,
                FR5,
                n,
                (20261026 + 99) if name == "Controlled" else (20261026 + 50000),
            )
            rows.append(
                {
                    "dataset": name,
                    "audit_n": n,
                    "audit_fraction": n / len(zv),
                    "mean_retention": mr,
                    "sd_retention": sd,
                    "violation_count": v,
                    "oracle_K5": gor(zv, FR5),
                    "oracle_continuous": oracle(zv),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "retention_audit_curve_r6.csv", index=False)
    # K sweep same seed at primary budgets
    krows = []
    for di, (name, zv, n) in enumerate(
        [("Controlled", zc, 800), ("AIR BAGS", za, round(0.35 * len(za)))]
    ):
        for K in [5, 10, 20, 50]:
            fr = np.linspace(0.30, 1.0, K)
            mr, sd, v = sim(zv, fr, n, BASE + di)
            krows.append(
                {
                    "dataset": name,
                    "K": K,
                    "audit_n": n,
                    "audit_fraction": n / len(zv),
                    "mean_retention": mr,
                    "sd_retention": sd,
                    "violation_count": v,
                    "oracle_grid": gor(zv, fr),
                    "oracle_continuous": oracle(zv),
                }
            )
    pd.DataFrame(krows).to_csv(OUT / "k_granularity_r6.csv", index=False)
    # risk ranking: semantic exact same primary seed; noisy rankings 10 each
    rrows = []
    n = round(0.35 * len(za))
    mr, sd, v = sim(za, FR5, n, BASE + 1)
    rrows.append(
        {
            "ranking": "semantic",
            "mean_retention": mr,
            "sd_across_rankings": 0.0,
            "violation_count": v,
        }
    )
    sdr = float(np.std(rr)) or 1
    for ci, sig in enumerate([1.0, 3.0]):
        vals = []
        for q in range(10):
            rng = np.random.default_rng(25000000 + ci * 100 + q)
            ordr = np.argsort(rr + rng.normal(0, sig * sdr, len(rr)), kind="mergesort")
            m, _, _ = sim(za[ordr], FR5, n, 50000000 + ci * 100 + q, R=500)
            vals.append(m)
        rrows.append(
            {
                "ranking": f"noise_{int(sig)}sd",
                "mean_retention": float(np.mean(vals)),
                "sd_across_rankings": float(np.std(vals, ddof=1)),
                "violation_count": None,
            }
        )
    vals = []
    for q in range(10):
        rng = np.random.default_rng(26000000 + q)
        m, _, _ = sim(za[rng.permutation(len(za))], FR5, n, 60000000 + q, R=500)
        vals.append(m)
    rrows.append(
        {
            "ranking": "random",
            "mean_retention": float(np.mean(vals)),
            "sd_across_rankings": float(np.std(vals, ddof=1)),
            "violation_count": None,
        }
    )
    pd.DataFrame(rrows).to_csv(OUT / "risk_ranking_ablation_r6.csv", index=False)
    print("CURVE\n", pd.DataFrame(rows).to_string(index=False))
    print("\nK\n", pd.DataFrame(krows).to_string(index=False))
    print("\nRISK\n", pd.DataFrame(rrows).to_string(index=False))


if __name__ == "__main__":
    main()
