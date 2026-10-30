from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certineg import (
    certify_nested_from_largest_sample,
    exact_coverage_min,
    group_inspection_cost,
    hypergeom_upper_rate,
    make_nested_sets,
    stratified_upper_rate,
)

OUT = ROOT.parent / "data/results"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(exist_ok=True)
SEED = 20261026


def make_candidate_population(seed=SEED, N=1200):
    rng = np.random.default_rng(seed)
    sizes = np.clip(np.rint(rng.lognormal(3.3, 1.0, N)).astype(int), 2, 500)
    latent = rng.beta(2, 6, N)
    p_def = expit(-5.0 + 8.0 * latent + 0.10 * np.log1p(sizes))
    # Independent stream makes the exact dataset frozen/reproducible.
    rz = np.random.default_rng(seed + 1)
    z = rz.binomial(1, p_def).astype(int)
    risk = np.clip(latent + 0.15 * rz.normal(size=N), 0, 1)
    groups = []
    for g, (m, defect, r) in enumerate(zip(sizes, z, risk)):
        labels = np.zeros(int(m), dtype=int)
        if defect:
            # Mostly one witness; sometimes 2-4. This makes existential error brittle.
            k = 1 + int(rng.binomial(3, 0.18))
            k = min(k, int(m))
            pos = rng.choice(int(m), size=k, replace=False)
            labels[pos] = 1
        # Proxy is deliberately imperfect but informative; never used for validity.
        scores = (
            rng.normal(0, 1, int(m)) + 2.0 * labels + 0.15 * rng.normal(size=int(m))
        )
        groups.append((labels, scores))
    return sizes, z, risk, groups


def row_vs_group_mismatch(seed=9917, N=1500, recall=0.92):
    rng = np.random.default_rng(seed)
    sizes = np.clip(np.rint(rng.lognormal(3.0, 0.9, N)).astype(int), 2, 250)
    # prevalence of groups with at least one true positive
    group_pos = rng.binomial(1, 0.62, N)
    total_tp = 0
    caught_tp = 0
    candidate = []
    candidate_true = []
    for g, m in enumerate(sizes):
        y = np.zeros(m, dtype=int)
        if group_pos[g]:
            k = 1 + rng.binomial(2, 0.15)
            y[rng.choice(m, size=min(k, m), replace=False)] = 1
        # semantic predicate with high positive-row recall and a tiny FP rate
        pred = np.zeros(m, dtype=int)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        if len(pos):
            pred[pos] = rng.binomial(1, recall, len(pos))
        if len(neg):
            pred[neg] = rng.binomial(1, 0.002, len(neg))
        total_tp += int(y.sum())
        caught_tp += int((pred * y).sum())
        if pred.sum() == 0:
            candidate.append(g)
            candidate_true.append(int(y.sum() > 0))
    rr = caught_tp / max(1, total_tp)
    fdp = np.mean(candidate_true) if candidate_true else 0.0
    return {
        "row_recall": rr,
        "candidate_groups": len(candidate),
        "group_false_absence_fdp": fdp,
        "positive_groups": int(group_pos.sum()),
        "total_groups": N,
    }


def clopper_pearson_upper(k, n, alpha=0.05):
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def run_nested(R=2000, n_outer=800, epsilon=0.10, delta=0.05):
    sizes, z, risk, groups = make_candidate_population()
    fracs = [0.30, 0.45, 0.60, 0.80, 1.00]
    sets = make_nested_sets(risk, fracs)
    largest = sets[-1]
    true_fdp = [float(z[A].mean()) for A in sets]
    rng = np.random.default_rng(SEED + 99)
    proxy_cost = np.array([group_inspection_cost(l, s) for l, s in groups], dtype=int)
    random_expected = np.array(
        [
            len(l) if l.sum() == 0 else (len(l) + 1) / (int(l.sum()) + 1)
            for l, s in groups
        ],
        dtype=float,
    )
    rows = []
    for rep in range(R):
        sample = np.sort(rng.choice(largest, size=n_outer, replace=False))
        cert = certify_nested_from_largest_sample(sets, sample, z, epsilon, delta)
        if cert.chosen_index is None:
            fdp = 0.0
            size = 0
            frac = 0.0
            violated = False
        else:
            A = sets[cert.chosen_index]
            fdp = float(z[A].mean())
            size = len(A)
            frac = fracs[cert.chosen_index]
            violated = fdp > epsilon
        pcost = int(proxy_cost[sample].sum())
        rcost = float(random_expected[sample].sum())
        rows.append(
            (
                rep,
                cert.chosen_index if cert.chosen_index is not None else -1,
                size,
                frac,
                fdp,
                violated,
                pcost,
                rcost,
                cert.chosen_upper_rate
                if cert.chosen_upper_rate is not None
                else np.nan,
            )
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "rep",
            "chosen_index",
            "chosen_size",
            "chosen_fraction",
            "true_fdp",
            "violation",
            "proxy_label_cost",
            "random_label_cost",
            "chosen_upper",
        ],
    )
    return df, fracs, true_fdp, sizes, z, risk, groups, sets


def fixed_set_cost_curve(z, groups, A, epsilon=0.10, delta=0.05, R=500):
    rng = np.random.default_rng(SEED + 277)
    proxy_cost_all = np.array(
        [group_inspection_cost(l, s) for l, s in groups], dtype=int
    )
    random_expected_all = np.array(
        [
            len(l) if l.sum() == 0 else (len(l) + 1) / (int(l.sum()) + 1)
            for l, s in groups
        ],
        dtype=float,
    )
    rec = []
    for n in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
        if n > len(A):
            continue
        for rep in range(R):
            s = np.sort(rng.choice(A, size=n, replace=False))
            x = int(z[s].sum())
            u = hypergeom_upper_rate(len(A), n, x, delta)
            proxy = int(proxy_cost_all[s].sum())
            rand = float(random_expected_all[s].sum())
            rec.append((n, rep, x, u, u <= epsilon, proxy, rand))
    return pd.DataFrame(
        rec,
        columns=[
            "n_groups",
            "rep",
            "defects_seen",
            "upper_fdp",
            "certified",
            "proxy_label_cost",
            "random_label_cost",
        ],
    )


def stratified_exploratory(
    z, groups, sizes, A, epsilon=0.10, delta=0.05, R=1000, total_n=600
):
    # Four size strata fixed before audit. Compare proportional allocation with a
    # cost-aware allocation that shifts sample toward cheap groups. Both remain
    # valid because each stratum is SRSWOR and bounds are Bonferroni-combined.
    a_sizes = sizes[A]
    edges = np.quantile(a_sizes, [0, 0.25, 0.5, 0.75, 1])
    sid = np.digitize(a_sizes, edges[1:-1], right=True)
    strata = [A[sid == h] for h in range(4)]
    Nh = np.array([len(s) for s in strata])
    avg_cost = np.array([np.mean(sizes[s]) for s in strata])
    prop = np.maximum(5, np.floor(total_n * Nh / Nh.sum()).astype(int))
    while prop.sum() < total_n:
        prop[np.argmax(Nh - prop)] += 1
    while prop.sum() > total_n:
        prop[np.argmax(prop)] -= 1
    # heuristic cost-aware allocation, NOT claimed optimal: N_h/sqrt(c_h)
    w = Nh / np.sqrt(avg_cost)
    ca = np.maximum(5, np.floor(total_n * w / w.sum()).astype(int))
    while ca.sum() < total_n:
        ca[np.argmax(w - ca / total_n)] += 1
    while ca.sum() > total_n:
        ca[np.argmax(ca)] -= 1
    ca = np.minimum(ca, Nh)
    rng = np.random.default_rng(SEED + 333)
    proxy_cost_all = np.array(
        [group_inspection_cost(l, s) for l, s in groups], dtype=int
    )
    rec = []
    for design, nh in [("proportional", prop), ("cost_aware_heuristic", ca)]:
        for rep in range(R):
            xs = []
            costs = []
            actual_n = []
            for s, n in zip(strata, nh):
                n = int(min(n, len(s)))
                sample = rng.choice(s, size=n, replace=False)
                actual_n.append(n)
                xs.append(int(z[sample].sum()))
                costs.append(int(proxy_cost_all[sample].sum()))
            u = stratified_upper_rate(Nh, actual_n, xs, delta)
            rec.append((design, rep, u, u <= epsilon, sum(actual_n), sum(costs)))
    meta = {
        "N_h": Nh.tolist(),
        "avg_group_size_h": [float(x) for x in avg_cost],
        "n_proportional": prop.tolist(),
        "n_cost_aware": ca.tolist(),
    }
    return pd.DataFrame(
        rec,
        columns=[
            "design",
            "rep",
            "upper_fdp",
            "certified",
            "n_groups",
            "proxy_label_cost",
        ],
    ), meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Main CertiNeg experiment pipeline.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT.parent / "data/results",
        help="Output directory for generated results",
    )
    args = parser.parse_args()
    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    # 1) exhaustive proof-oriented coverage audit for manageable finite populations.
    cov_rows = []
    for N in [10, 20, 30, 40]:
        for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
            n = max(0, min(N, round(frac * N)))
            for alpha in [0.10, 0.05, 0.01]:
                cov, _ = exact_coverage_min(N, n, alpha)
                cov_rows.append((N, n, alpha, cov))
    cov = pd.DataFrame(cov_rows, columns=["N", "n", "alpha", "min_exact_coverage"])
    cov.to_csv(out / "exact_coverage_grid.csv", index=False)

    # 2) motivating mismatch.
    mismatch = row_vs_group_mismatch()
    (out / "row_vs_group_mismatch.json").write_text(json.dumps(mismatch, indent=2))

    # 3) nested selection and inner-order cost.
    nested, fracs, true_fdp, sizes, z, _risk, groups, sets = run_nested()
    nested.to_csv(out / "nested_audit_repetitions.csv", index=False)
    summ = {
        "R": len(nested),
        "epsilon": 0.10,
        "delta": 0.05,
        "outer_sample_n": 800,
        "fractions": fracs,
        "true_fdp_by_fraction": true_fdp,
        "violation_count": int(nested.violation.sum()),
        "violation_rate": float(nested.violation.mean()),
        "violation_cp95_upper": clopper_pearson_upper(
            int(nested.violation.sum()), len(nested), 0.05
        ),
        "median_chosen_fraction": float(nested.chosen_fraction.median()),
        "mean_chosen_fraction": float(nested.chosen_fraction.mean()),
        "proxy_cost_mean": float(nested.proxy_label_cost.mean()),
        "random_cost_mean": float(nested.random_label_cost.mean()),
        "proxy_cost_saving_pct": float(
            100 * (1 - nested.proxy_label_cost.mean() / nested.random_label_cost.mean())
        ),
        "candidate_N": len(z),
        "candidate_true_fdp": float(z.mean()),
        "group_size_median": float(np.median(sizes)),
        "group_size_p95": float(np.percentile(sizes, 95)),
    }
    (out / "nested_summary.json").write_text(json.dumps(summ, indent=2))

    # 4) fixed-set certification-cost curve on the largest truly safe nested set (80%).
    A = sets[3]
    curve = fixed_set_cost_curve(z, groups, A, R=250)
    curve.to_csv(out / "fixed_set_cost_curve.csv", index=False)

    # 5) exploratory valid stratification on same set.
    strat, meta = stratified_exploratory(z, groups, sizes, A, R=500, total_n=600)
    strat.to_csv(out / "stratified_exploratory.csv", index=False)
    (out / "stratified_meta.json").write_text(json.dumps(meta, indent=2))

    # Summary table for manuscript.
    rows = []
    rows.append(
        {
            "experiment": "Exact hypergeometric coverage",
            "metric": "minimum enumerated coverage",
            "value": cov.min_exact_coverage.min(),
            "note": "All N<=60 grid cases at/above nominal",
        }
    )
    rows.append(
        {
            "experiment": "Row-to-group mismatch",
            "metric": "row recall",
            "value": mismatch["row_recall"],
            "note": "controlled semantic-predicate simulation",
        }
    )
    rows.append(
        {
            "experiment": "Row-to-group mismatch",
            "metric": "group false-absence FDP",
            "value": mismatch["group_false_absence_fdp"],
            "note": f"{mismatch['candidate_groups']} candidate-negative groups",
        }
    )
    rows.append(
        {
            "experiment": "Nested certification",
            "metric": "empirical violation rate",
            "value": summ["violation_rate"],
            "note": f"R={summ['R']}, target delta={summ['delta']}",
        }
    )
    rows.append(
        {
            "experiment": "Nested certification",
            "metric": "95% CP upper on violation rate",
            "value": summ["violation_cp95_upper"],
            "note": "one-sided exact binomial upper",
        }
    )
    rows.append(
        {
            "experiment": "Inner proxy order",
            "metric": "trusted-label savings",
            "value": summ["proxy_cost_saving_pct"] / 100,
            "note": "relative to random within-group order",
        }
    )
    pd.DataFrame(rows).to_csv(out / "paper_summary.csv", index=False)

    # Figures. Use default matplotlib colors/styles only.
    g = (
        curve.groupby("n_groups")
        .agg(
            cert_rate=("certified", "mean"),
            proxy_cost=("proxy_label_cost", "mean"),
            random_cost=("random_label_cost", "mean"),
        )
        .reset_index()
    )
    plt.figure(figsize=(5.4, 3.3))
    plt.plot(g.proxy_cost, g.cert_rate, marker="o", label="Proxy-ranked")
    plt.plot(g.random_cost, g.cert_rate, marker="s", label="Random order")
    plt.xlabel("Mean trusted record labels")
    plt.ylabel("Certification probability")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "cost_utility.pdf", bbox_inches="tight")
    plt.savefig(FIG / "cost_utility.png", dpi=220, bbox_inches="tight")
    plt.close()

    counts = nested.chosen_fraction.value_counts(normalize=True).sort_index()
    plt.figure(figsize=(5.2, 3.2))
    plt.bar([str(x) for x in counts.index], counts.values)
    plt.xlabel("Selected fraction of candidate output")
    plt.ylabel("Frequency over audits")
    plt.tight_layout()
    plt.savefig(FIG / "nested_selection.pdf", bbox_inches="tight")
    plt.savefig(FIG / "nested_selection.png", dpi=220, bbox_inches="tight")
    plt.close()

    st = (
        strat.groupby("design")
        .agg(
            cert_rate=("certified", "mean"),
            cost=("proxy_label_cost", "mean"),
            upper=("upper_fdp", "mean"),
        )
        .reset_index()
    )
    plt.figure(figsize=(5.2, 3.2))
    plt.bar(st.design, st.cost)
    plt.ylabel("Mean trusted record labels")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "stratified_cost.pdf", bbox_inches="tight")
    plt.savefig(FIG / "stratified_cost.png", dpi=220, bbox_inches="tight")
    plt.close()

    print(
        json.dumps(
            {
                "nested_summary": summ,
                "mismatch": mismatch,
                "stratified": st.to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
