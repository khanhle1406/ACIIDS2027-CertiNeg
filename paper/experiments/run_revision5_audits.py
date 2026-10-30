from __future__ import annotations

import json
import math
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

SEED = 20261026
EPS = 0.10
DELTA = 0.05
MAIN_FRACS = np.array([0.30, 0.45, 0.60, 0.80, 1.00])
AUDIT_RATES = np.array([0.05, 0.10, 0.20, 0.35, 0.50, 2 / 3])
TARGETS = ["AIR BAGS", "ENGINE", "SERVICE BRAKES"]


def sizes_for_fracs(N, fracs):
    out = []
    prev = 0
    for f in fracs:
        k = max(prev, min(N, round(float(f) * N)))
        out.append(k)
        prev = k
    return np.array(out, dtype=int)


def fixed_sequence(sample_idx, z, prefix_sizes):
    N = len(z)
    mark = np.zeros(N, dtype=np.int8)
    mark[np.asarray(sample_idx, dtype=int)] = 1
    cn = np.cumsum(mark)
    cx = np.cumsum(mark * z)
    chosen = -1
    for k, m in enumerate(prefix_sizes):
        n = int(cn[m - 1]) if m else 0
        x = int(cx[m - 1]) if m else 0
        if hypergeom_upper_rate(int(m), n, x, DELTA) <= EPS:
            chosen = k
        else:
            break
    return chosen


def simulate(z, fracs, n, R, seed):
    N = len(z)
    ps = sizes_for_fracs(N, fracs)
    fdps = np.array([z[:m].mean() if m else 0 for m in ps])
    rng = np.random.default_rng(seed)
    ret = np.empty(R)
    vio = np.zeros(R, dtype=int)
    for r in range(R):
        s = rng.choice(N, size=min(n, N), replace=False)
        k = fixed_sequence(s, z, ps)
        if k < 0:
            ret[r] = 0
        else:
            ret[r] = ps[k] / N
            vio[r] = int(fdps[k] > EPS)
    return {
        "mean_retention": float(ret.mean()),
        "sd_retention": float(ret.std(ddof=1)),
        "median_retention": float(np.median(ret)),
        "violation_count": int(vio.sum()),
        "true_fdps": fdps.tolist(),
    }


def oracle_cont(z, min_frac=0.30):
    N = len(z)
    m = np.arange(1, N + 1)
    f = np.cumsum(z) / m
    ok = (m >= math.ceil(min_frac * N)) & (f <= EPS)
    return float(m[ok].max() / N) if ok.any() else 0.0


def grid_oracle(z, fracs):
    N = len(z)
    ps = sizes_for_fracs(N, fracs)
    vals = [m / N for i, m in enumerate(ps) if z[:m].mean() <= EPS]
    return max(vals, default=0.0)


def controlled():
    _sizes, z, risk, _groups = make_candidate_population()
    order = np.argsort(risk, kind="mergesort")
    return {"name": "Controlled", "z": z[order].astype(np.int8), "N": len(z)}


def load_target(t):
    safe = t.lower().replace(" ", "_")
    a = np.load(OUT / f"sembench_{safe}_groups.npz")
    return {
        "name": t,
        "z": a["z"].astype(np.int8),
        "risk": a["risk"],
        "cost": a["cost"],
        "N": len(a["z"]),
    }


def main() -> None:
    ctr = controlled()
    sem = {t: load_target(t) for t in TARGETS}
    meta = json.loads((OUT / "sembench_predicate_cache_meta.json").read_text())[
        "targets"
    ]
    curve = []
    for di, d in enumerate([ctr, sem["AIR BAGS"]]):
        for ri, rate in enumerate(AUDIT_RATES):
            n = max(1, min(d["N"], round(rate * d["N"])))
            s = simulate(d["z"], MAIN_FRACS, n, 1000, SEED + 1000 * di + ri)
            curve.append(
                {
                    "dataset": d["name"],
                    "audit_fraction": n / d["N"],
                    "audit_n": n,
                    "mean_retention": s["mean_retention"],
                    "sd_retention": s["sd_retention"],
                    "oracle_continuous": oracle_cont(d["z"]),
                    "oracle_K5": grid_oracle(d["z"], MAIN_FRACS),
                    "violation_count": s["violation_count"],
                    "expected_label_cost_vs_census": n / d["N"],
                }
            )
    curve = pd.DataFrame(curve)
    curve.to_csv(OUT / "retention_audit_curve.csv", index=False)

    krows = []
    for di, d in enumerate([ctr, sem["AIR BAGS"]]):
        n = 800 if d["name"] == "Controlled" else round(0.35 * d["N"])
        for K in [5, 10, 20, 50]:
            fr = np.linspace(0.30, 1.0, K)
            s = simulate(d["z"], fr, n, 1000, SEED + 3000 + 100 * di + K)
            krows.append(
                {
                    "dataset": d["name"],
                    "K": K,
                    "audit_n": n,
                    "audit_fraction": n / d["N"],
                    "mean_retention": s["mean_retention"],
                    "sd_retention": s["sd_retention"],
                    "oracle_grid": grid_oracle(d["z"], fr),
                    "oracle_continuous": oracle_cont(d["z"]),
                    "violation_count": s["violation_count"],
                }
            )
    kdf = pd.DataFrame(krows)
    kdf.to_csv(OUT / "k_granularity.csv", index=False)

    prows = []
    for ti, t in enumerate(TARGETS):
        d = sem[t]
        n = max(1, round(0.35 * d["N"]))
        s = simulate(d["z"], MAIN_FRACS, n, 2000, SEED + 5000 + ti)
        m = meta[t]
        prows.append(
            {
                "predicate": t,
                "test_precision": m["precision"],
                "test_recall": m["recall"],
                "test_f1": m["f1"],
                "test_auc": m["auc"],
                "candidate_groups": d["N"],
                "candidate_fdp": float(d["z"].mean()),
                "oracle_K5": grid_oracle(d["z"], MAIN_FRACS),
                "oracle_continuous": oracle_cont(d["z"]),
                "mean_retention": s["mean_retention"],
                "sd_retention": s["sd_retention"],
                "median_retention": s["median_retention"],
                "audit_n": n,
                "audit_fraction": n / d["N"],
                "label_cost_vs_census_expected": n / d["N"],
                "violation_count": s["violation_count"],
            }
        )
    mp = pd.DataFrame(prows)
    mp.to_csv(OUT / "sembench_multi_predicate.csv", index=False)

    # Rank-quality ablation on AIR BAGS. Arrays are in semantic-risk order.
    base = sem["AIR BAGS"]
    N = base["N"]
    item_z = base["z"].copy()
    item_risk = base["risk"].copy()
    sd = float(np.std(item_risk)) or 1.0
    rrows = []
    for ci, (label, sigma) in enumerate(
        [("semantic", 0.0), ("noise_1sd", 1.0), ("noise_3sd", 3.0)]
    ):
        vals = []
        ors = []
        reps = 1 if sigma == 0 else 10
        for q in range(reps):
            rng = np.random.default_rng(SEED + 7000 + ci * 100 + q)
            score = (
                item_risk if sigma == 0 else item_risk + rng.normal(0, sigma * sd, N)
            )
            z = item_z[np.argsort(score, kind="mergesort")]
            ss = simulate(
                z,
                MAIN_FRACS,
                round(0.35 * N),
                500 if sigma else 1000,
                SEED + 8000 + ci * 100 + q,
            )
            vals.append(ss["mean_retention"])
            ors.append(oracle_cont(z))
        rrows.append(
            {
                "ranking": label,
                "mean_retention": float(np.mean(vals)),
                "sd_across_rankings": (
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                ),
                "mean_oracle_continuous": float(np.mean(ors)),
            }
        )
    vals = []
    ors = []
    for q in range(10):
        rng = np.random.default_rng(SEED + 9000 + q)
        z = item_z[rng.permutation(N)]
        ss = simulate(z, MAIN_FRACS, round(0.35 * N), 500, SEED + 10000 + q)
        vals.append(ss["mean_retention"])
        ors.append(oracle_cont(z))
    rrows.append(
        {
            "ranking": "random",
            "mean_retention": float(np.mean(vals)),
            "sd_across_rankings": float(np.std(vals, ddof=1)),
            "mean_oracle_continuous": float(np.mean(ors)),
        }
    )
    rdf = pd.DataFrame(rrows)
    rdf.to_csv(OUT / "risk_ranking_ablation.csv", index=False)

    summ = {
        "controlled_oracle_continuous": oracle_cont(ctr["z"]),
        "airbags_oracle_continuous": oracle_cont(sem["AIR BAGS"]["z"]),
        "multi_predicate_mean_retention": float(mp.mean_retention.mean()),
        "multi_predicate_sd_across_predicates": float(mp.mean_retention.std(ddof=1)),
    }
    (OUT / "revision5_summary.json").write_text(json.dumps(summ, indent=2))
    print("CURVE\n", curve.to_string(index=False))
    print("\nK\n", kdf.to_string(index=False))
    print("\nPREDICATES\n", mp.to_string(index=False))
    print("\nRISK\n", rdf.to_string(index=False))
    print("\n", summ)


if __name__ == "__main__":
    main()
