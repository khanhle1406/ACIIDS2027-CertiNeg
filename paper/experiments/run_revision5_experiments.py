from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "data/results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from certineg import group_inspection_cost, hypergeom_upper_rate
from run_experiments import make_candidate_population

SEED = 20261026
EPS = 0.10
DELTA = 0.05
MAIN_FRACS = np.array([0.30, 0.45, 0.60, 0.80, 1.00])
AUDIT_RATES = np.array([0.05, 0.10, 0.20, 0.35, 0.50, 2 / 3])
TARGETS = ["AIR BAGS", "ENGINE", "SERVICE BRAKES"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    type=Path,
    default=ROOT.parent / "data/raw/full_data",
    help="Path to full_data directory containing car_data_full.csv and text_complaints_data_full.csv",
)
args, _ = parser.parse_known_args()
DATA = args.data_dir


def sizes_for_fracs(N, fracs):
    # Mirror make_nested_sets rounding while deduplicating only if needed.
    out = []
    prev = 0
    for f in fracs:
        k = max(prev, min(N, round(float(f) * N)))
        out.append(k)
        prev = k
    return np.array(out, dtype=int)


def fixed_sequence_from_sample(
    sample_idx, z_order, prefix_sizes, epsilon=EPS, delta=DELTA
):
    N = len(z_order)
    mark = np.zeros(N, dtype=np.int8)
    mark[np.asarray(sample_idx, dtype=int)] = 1
    cum_n = np.cumsum(mark)
    cum_x = np.cumsum(mark * z_order)
    chosen = -1
    for k, m in enumerate(prefix_sizes):
        n = int(cum_n[m - 1]) if m else 0
        x = int(cum_x[m - 1]) if m else 0
        u = hypergeom_upper_rate(int(m), n, x, delta) if m else 0.0
        if u <= epsilon:
            chosen = k
        else:
            break
    return chosen


def simulate_fixed(z_order, fracs, audit_n, R, seed):
    N = len(z_order)
    ps = sizes_for_fracs(N, fracs)
    true_fdps = np.array([z_order[:m].mean() if m else 0.0 for m in ps])
    rng = np.random.default_rng(seed)
    rets = np.empty(R)
    violations = np.zeros(R, dtype=int)
    for r in range(R):
        s = rng.choice(N, size=min(audit_n, N), replace=False)
        k = fixed_sequence_from_sample(s, z_order, ps)
        if k < 0:
            rets[r] = 0.0
        else:
            rets[r] = ps[k] / N
            violations[r] = int(true_fdps[k] > EPS)
    return {
        "mean_retention": float(rets.mean()),
        "sd_retention": float(rets.std(ddof=1)),
        "median_retention": float(np.median(rets)),
        "violation_count": int(violations.sum()),
        "true_fdps": true_fdps.tolist(),
        "prefix_sizes": ps.tolist(),
    }


def oracle_prefix(z_order, min_frac=0.30):
    N = len(z_order)
    cs = np.cumsum(z_order)
    m = np.arange(1, N + 1)
    fdp = cs / m
    eligible = (m >= math.ceil(min_frac * N)) & (fdp <= EPS)
    if not eligible.any():
        return 0.0
    return float(m[eligible].max() / N)


def build_controlled():
    _sizes, z, risk, groups = make_candidate_population()
    order = np.argsort(risk, kind="mergesort")
    zord = z[order].astype(int)
    proxy_cost = np.array(
        [group_inspection_cost(labels, scores) for labels, scores in groups],
        dtype=float,
    )[order]
    return {
        "name": "Controlled",
        "z": zord,
        "risk": np.asarray(risk)[order],
        "cost": proxy_cost,
        "N": len(zord),
        "census_cost": float(proxy_cost.sum()),
    }


def load_sembench_base():
    text = pd.read_csv(
        DATA / "text_complaints_data_full.csv",
        usecols=["car_id", "summary", "component_class"],
    )
    car = pd.read_csv(
        DATA / "car_data_full.csv", usecols=["car_id", "country", "year", "fuel_type"]
    )
    df = text.merge(car, on="car_id", how="inner")
    df["summary"] = df.summary.fillna("").astype(str)
    keys = ["country", "year", "fuel_type"]
    gid = df[keys].astype(str).agg("||".join, axis=1)
    # Same group-disjoint split for all predicates.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=SEED)
    tr, te = next(gss.split(df, np.zeros(len(df)), groups=gid))
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=5,
        max_df=0.995,
        max_features=20000,
        sublinear_tf=True,
    )
    Xtr = vec.fit_transform(df.iloc[tr].summary)
    Xte = vec.transform(df.iloc[te].summary)
    return df, keys, tr, te, Xtr, Xte


def build_sembench_targets():
    df, keys, tr, te, Xtr, Xte = load_sembench_base()
    out = {}
    for ti, target in enumerate(TARGETS):
        yall = df.component_class.eq(target).astype(int).to_numpy()
        ytr = yall[tr]
        yte = yall[te]
        clf = SGDClassifier(
            loss="log_loss",
            alpha=2e-6,
            class_weight="balanced",
            max_iter=12,
            tol=1e-3,
            random_state=SEED,
        )
        clf.fit(Xtr, ytr)
        ptr = clf.predict_proba(Xtr)[:, 1]
        p = clf.predict_proba(Xte)[:, 1]
        pos = ptr[ytr == 1]
        thr = float(np.quantile(pos, 0.08))
        pred = (p >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            yte, pred, average="binary", zero_division=0
        )
        auc = float(roc_auc_score(yte, p))
        test = df.iloc[te][keys + ["component_class"]].copy().reset_index(drop=True)
        test["score"] = p
        test["pred"] = pred
        test["y"] = yte
        groups = []
        for key, g in test.groupby(keys, sort=True):
            if bool(g.pred.any()):
                continue
            labels = g.y.to_numpy(dtype=int)
            scores = g.score.to_numpy(dtype=float)
            groups.append(
                (key, labels, scores, int(labels.any()), float(scores.max()), len(g))
            )
        groups.sort(key=lambda x: (x[4], x[0]))
        z = np.array([g[3] for g in groups], dtype=int)
        risk = np.array([g[4] for g in groups], dtype=float)
        cost = np.array(
            [group_inspection_cost(g[1], g[2]) for g in groups], dtype=float
        )
        out[target] = {
            "name": target,
            "z": z,
            "risk": risk,
            "cost": cost,
            "N": len(z),
            "census_cost": float(cost.sum()),
            "test_rows": len(te),
            "threshold": thr,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": auc,
            "candidate_fdp": float(z.mean()) if len(z) else 0.0,
        }
    return out


def main():
    controlled = build_controlled()
    sembench = build_sembench_targets()

    # 1) Retention vs audit budget on controlled + AIR BAGS, original K=5 family.
    curve = []
    for di, d in enumerate([controlled, sembench["AIR BAGS"]]):
        N = d["N"]
        z = d["z"]
        oracle = oracle_prefix(z, 0.30)
        # grid oracle for exact K=5 family
        ps = sizes_for_fracs(N, MAIN_FRACS)
        fdps = [float(z[:m].mean()) for m in ps]
        grid_oracle = max(
            [ps[i] / N for i, v in enumerate(fdps) if v <= EPS], default=0.0
        )
        for ri, rate in enumerate(AUDIT_RATES):
            n = max(1, min(N, round(rate * N)))
            sim = simulate_fixed(z, MAIN_FRACS, n, R=1000, seed=SEED + 1000 * di + ri)
            curve.append(
                {
                    "dataset": d["name"],
                    "audit_fraction": n / N,
                    "audit_n": n,
                    "mean_retention": sim["mean_retention"],
                    "sd_retention": sim["sd_retention"],
                    "oracle_continuous": oracle,
                    "oracle_K5": grid_oracle,
                    "violation_count": sim["violation_count"],
                    "expected_label_cost_vs_census": n / N,
                }
            )
    pd.DataFrame(curve).to_csv(OUT / "retention_audit_curve.csv", index=False)

    # 2) K granularity: same fraction range [0.30,1.0], fixed sequence, current audit budgets.
    krows = []
    for di, d in enumerate([controlled, sembench["AIR BAGS"]]):
        n = 800 if d["name"] == "Controlled" else round(0.35 * d["N"])
        for K in [5, 10, 20, 50]:
            fr = np.linspace(0.30, 1.0, K)
            sim = simulate_fixed(d["z"], fr, n, R=1000, seed=SEED + 3000 + 100 * di + K)
            ps = sizes_for_fracs(d["N"], fr)
            fdps = [float(d["z"][:m].mean()) for m in ps]
            grid_oracle = max(
                [ps[i] / d["N"] for i, v in enumerate(fdps) if v <= EPS], default=0.0
            )
            krows.append(
                {
                    "dataset": d["name"],
                    "K": K,
                    "audit_n": n,
                    "audit_fraction": n / d["N"],
                    "mean_retention": sim["mean_retention"],
                    "sd_retention": sim["sd_retention"],
                    "oracle_grid": grid_oracle,
                    "oracle_continuous": oracle_prefix(d["z"], 0.30),
                    "violation_count": sim["violation_count"],
                }
            )
    pd.DataFrame(krows).to_csv(OUT / "k_granularity.csv", index=False)

    # 3) Multiple predicates at 35% group audit and original K=5 family.
    prows = []
    for ti, target in enumerate(TARGETS):
        d = sembench[target]
        n = max(1, round(0.35 * d["N"]))
        sim = simulate_fixed(d["z"], MAIN_FRACS, n, R=2000, seed=SEED + 5000 + ti)
        ps = sizes_for_fracs(d["N"], MAIN_FRACS)
        fdps = [float(d["z"][:m].mean()) for m in ps]
        grid_oracle = max(
            [ps[i] / d["N"] for i, v in enumerate(fdps) if v <= EPS], default=0.0
        )
        prows.append(
            {
                "predicate": target,
                "test_precision": d["precision"],
                "test_recall": d["recall"],
                "test_f1": d["f1"],
                "test_auc": d["auc"],
                "candidate_groups": d["N"],
                "candidate_fdp": d["candidate_fdp"],
                "oracle_K5": grid_oracle,
                "mean_retention": sim["mean_retention"],
                "sd_retention": sim["sd_retention"],
                "median_retention": sim["median_retention"],
                "audit_n": n,
                "audit_fraction": n / d["N"],
                "label_cost_vs_census_expected": n / d["N"],
                "violation_count": sim["violation_count"],
            }
        )
    mp = pd.DataFrame(prows)
    mp.to_csv(OUT / "sembench_multi_predicate.csv", index=False)

    # 4) Risk-ranking quality on AIR BAGS: score noise and complete shuffle.
    base = sembench["AIR BAGS"]
    N = base["N"]
    base_risk = base["risk"]
    z_orig = base["z"]
    riskrows = []
    # Since base arrays already sorted by original risk, recover an item table and reorder by perturbed scores.
    item_z = z_orig.copy()
    item_risk = base_risk.copy()
    std = float(np.std(item_risk)) or 1.0
    configs = [("semantic", 0.0), ("noise_1sd", 1.0), ("noise_3sd", 3.0)]
    for ci, (label, sigma) in enumerate(configs):
        vals = []
        oracles = []
        repscores = 1 if sigma == 0 else 10
        for q in range(repscores):
            rr = np.random.default_rng(SEED + 7000 + ci * 100 + q)
            score = (
                item_risk.copy()
                if sigma == 0
                else item_risk + rr.normal(0, sigma * std, size=N)
            )
            order = np.argsort(score, kind="mergesort")
            z = item_z[order]
            n = round(0.35 * N)
            sim = simulate_fixed(
                z,
                MAIN_FRACS,
                n,
                R=500 if sigma else 1000,
                seed=SEED + 8000 + ci * 100 + q,
            )
            vals.append(sim["mean_retention"])
            oracles.append(oracle_prefix(z, 0.30))
        riskrows.append(
            {
                "ranking": label,
                "score_noise_sd": sigma,
                "mean_retention": float(np.mean(vals)),
                "sd_across_rankings": float(np.std(vals, ddof=1))
                if len(vals) > 1
                else 0.0,
                "mean_oracle_continuous": float(np.mean(oracles)),
            }
        )
    # Fully random rankings across 10 seeds.
    vals = []
    oracles = []
    for q in range(10):
        rr = np.random.default_rng(SEED + 9000 + q)
        order = rr.permutation(N)
        z = item_z[order]
        sim = simulate_fixed(
            z, MAIN_FRACS, round(0.35 * N), R=500, seed=SEED + 10000 + q
        )
        vals.append(sim["mean_retention"])
        oracles.append(oracle_prefix(z, 0.30))
    riskrows.append(
        {
            "ranking": "random",
            "score_noise_sd": np.nan,
            "mean_retention": float(np.mean(vals)),
            "sd_across_rankings": float(np.std(vals, ddof=1)),
            "mean_oracle_continuous": float(np.mean(oracles)),
        }
    )
    pd.DataFrame(riskrows).to_csv(OUT / "risk_ranking_ablation.csv", index=False)

    summary = {
        "targets": TARGETS,
        "audit_rates": AUDIT_RATES.tolist(),
        "main_fractions": MAIN_FRACS.tolist(),
        "controlled_oracle_continuous": oracle_prefix(controlled["z"], 0.30),
        "airbags_oracle_continuous": oracle_prefix(sembench["AIR BAGS"]["z"], 0.30),
        "multi_predicate_mean_retention": float(mp.mean_retention.mean()),
        "multi_predicate_sd_retention_across_predicates": float(
            mp.mean_retention.std(ddof=1)
        ),
    }
    (OUT / "revision5_summary.json").write_text(json.dumps(summary, indent=2))
    print("RETENTION CURVE")
    print(pd.DataFrame(curve).to_string(index=False))
    print("\nK GRANULARITY")
    print(pd.DataFrame(krows).to_string(index=False))
    print("\nMULTI PREDICATE")
    print(mp.to_string(index=False))
    print("\nRISK ABLATION")
    print(pd.DataFrame(riskrows).to_string(index=False))
    print("\nSUMMARY", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
