import argparse
import json
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
parser = argparse.ArgumentParser(
    description="Run the real SemBench Cars text workload for CertiNeg."
)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=ROOT.parent / "data/raw/full_data",
    help="Directory containing text_complaints_data_full.csv and car_data_full.csv",
)
args = parser.parse_args()
DATA = args.data_dir
sys.path.insert(0, str(ROOT / "src"))
from certineg import (
    certify_nested_fixed_sequence,
    certify_nested_from_largest_sample,
    certify_nested_weighted,
    group_inspection_cost,
    make_nested_sets,
)

SEED = 20261026
rng = np.random.default_rng(SEED)
TARGET = "AIR BAGS"
EPS = 0.10
DELTA = 0.05
R = 2000
FRACS = [0.3, 0.45, 0.6, 0.8, 1.0]

text = pd.read_csv(
    DATA / "text_complaints_data_full.csv",
    usecols=["car_id", "summary", "component_class"],
)
car = pd.read_csv(
    DATA / "car_data_full.csv", usecols=["car_id", "country", "year", "fuel_type"]
)
df = text.merge(car, on="car_id", how="inner")
df["summary"] = df["summary"].fillna("").astype(str)
df["y"] = df["component_class"].eq(TARGET).astype(int)

# Deterministic group-disjoint split by the eventual NOT EXISTS grouping key.
keys = ["country", "year", "fuel_type"]
group_id = df[keys].astype(str).agg("||".join, axis=1)
gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=SEED)
tr, te = next(gss.split(df, df["y"], groups=group_id))
vec = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    min_df=5,
    max_df=0.995,
    max_features=20000,
    sublinear_tf=True,
)
Xtr = vec.fit_transform(df.iloc[tr]["summary"])
Xte = vec.transform(df.iloc[te]["summary"])
clf = SGDClassifier(
    loss="log_loss",
    alpha=2e-6,
    class_weight="balanced",
    max_iter=12,
    tol=1e-3,
    random_state=SEED,
)
clf.fit(Xtr, df.iloc[tr]["y"])
p = clf.predict_proba(Xte)[:, 1]
y = df.iloc[te]["y"].to_numpy()
# Select threshold using training scores to target high recall without test-label tuning.
ptr = clf.predict_proba(Xtr)[:, 1]
ytr = df.iloc[tr]["y"].to_numpy()
pos = ptr[ytr == 1]
thr = float(np.quantile(pos, 0.08))  # approx 92% training positive recall
pred = (p >= thr).astype(int)
prec, rec, f1, _ = precision_recall_fscore_support(
    y, pred, average="binary", zero_division=0
)
auc = roc_auc_score(y, p)

test = df.iloc[te].copy().reset_index(drop=True)
test["score"] = p
test["pred"] = pred
# Every held-out row is one complaint record. Candidate groups are predicted NOT EXISTS.

grouped = []
for key, g in test.groupby(keys, sort=True):
    labels = g["y"].to_numpy(dtype=int)
    scores = g["score"].to_numpy(dtype=float)
    pred_any = bool(g["pred"].any())
    true_any = bool(labels.any())
    if not pred_any:
        grouped.append(
            (key, labels, scores, int(true_any), float(scores.max()), len(g))
        )

# Lower max score = safer predicted-negative group.
grouped.sort(key=lambda x: (x[4], x[0]))
N = len(grouped)
z = np.array([x[3] for x in grouped], dtype=int)
risks = np.array([x[4] for x in grouped], dtype=float)
sizes = np.array([x[5] for x in grouped], dtype=int)
sets = make_nested_sets(risks, FRACS)
true_fdps = [float(z[s].mean()) if len(s) else 0.0 for s in sets]

# Inner trusted-label cost: inspect highest semantic score first; negative groups exhaust.
cost = np.array([group_inspection_cost(x[1], x[2]) for x in grouped], dtype=int)


# Expected random-order cost for exact group truth.
def random_expected(labels):
    m = len(labels)
    r = int(labels.sum())
    if r == 0:
        return float(m)
    return float((m + 1) / (r + 1))


rand_cost = np.array([random_expected(x[1]) for x in grouped])

# Audit size: cap at 800 or 35% candidate population, whichever is smaller, but at least 100.
n = min(800, max(100, round(0.35 * N)))
rows = []
weights = np.asarray(FRACS, dtype=float)
weights = weights / weights.sum()
for rep in range(R):
    s = rng.choice(N, size=n, replace=False)
    certs = [
        (
            "equal_simultaneous",
            certify_nested_from_largest_sample(sets, s, z, epsilon=EPS, delta=DELTA),
        ),
        (
            "retention_weighted",
            certify_nested_weighted(
                sets, s, z, epsilon=EPS, delta=DELTA, weights=weights
            ),
        ),
        (
            "fixed_sequence",
            certify_nested_fixed_sequence(sets, s, z, epsilon=EPS, delta=DELTA),
        ),
    ]
    for method, cert in certs:
        ci = cert.chosen_index
        if ci is None:
            frac = 0.0
            tf = 0.0
        else:
            frac = len(sets[ci]) / N
            tf = true_fdps[ci]
        rows.append(
            (
                rep,
                method,
                ci if ci is not None else -1,
                frac,
                tf,
                int(tf > EPS),
                int(cost[s].sum()),
                float(rand_cost[s].sum()),
            )
        )
res = pd.DataFrame(
    rows,
    columns=[
        "rep",
        "method",
        "chosen_index",
        "chosen_fraction",
        "true_fdp",
        "violation",
        "proxy_label_cost",
        "random_expected_cost",
    ],
)
res.to_csv(OUT / "sembench_text_audit.csv", index=False)
mainres = res[res.method.eq("fixed_sequence")].copy()
method_summary = (
    res.groupby("method")
    .agg(
        mean_selected_fraction=("chosen_fraction", "mean"),
        median_selected_fraction=("chosen_fraction", "median"),
        violation_count=("violation", "sum"),
        mean_proxy_label_cost=("proxy_label_cost", "mean"),
    )
    .reset_index()
)
method_summary.to_csv(OUT / "sembench_method_comparison.csv", index=False)
summary = {
    "dataset_rows_total": len(df),
    "test_rows": len(test),
    "target": TARGET,
    "classifier": "TF-IDF unigram (20k features) + SGD logistic loss",
    "threshold": thr,
    "test_precision": float(prec),
    "test_recall": float(rec),
    "test_f1": float(f1),
    "test_auc": float(auc),
    "group_key": "country, year, fuel_type",
    "group_disjoint_split": True,
    "candidate_negative_groups": N,
    "candidate_true_fdp": float(z.mean()) if N else 0.0,
    "candidate_group_size_median": float(np.median(sizes)) if N else 0.0,
    "candidate_group_size_p95": float(np.quantile(sizes, 0.95)) if N else 0.0,
    "nested_fractions": FRACS,
    "nested_true_fdps": true_fdps,
    "audit_n": n,
    "R": R,
    "epsilon": EPS,
    "delta": DELTA,
    "primary_selection_method": "fixed_sequence",
    "mean_selected_fraction": float(mainres.chosen_fraction.mean()),
    "median_selected_fraction": float(mainres.chosen_fraction.median()),
    "violation_count": int(mainres.violation.sum()),
    "violation_rate": float(mainres.violation.mean()),
    "mean_proxy_label_cost": float(mainres.proxy_label_cost.mean()),
    "mean_random_expected_cost": float(mainres.random_expected_cost.mean()),
    "inner_cost_saving_pct": float(
        1 - mainres.proxy_label_cost.mean() / mainres.random_expected_cost.mean()
    ),
    "selected_fraction_counts": {
        str(k): int(v)
        for k, v in mainres.chosen_fraction.value_counts().sort_index().items()
    },
    "census_proxy_label_cost": int(cost.sum()),
    "audit_proxy_cost_vs_census": float(mainres.proxy_label_cost.mean() / cost.sum()),
    "method_mean_selected_fraction": {
        r["method"]: float(r["mean_selected_fraction"])
        for _, r in method_summary.iterrows()
    },
    "weighted_alpha_weights": weights.tolist(),
}
(OUT / "sembench_text_summary.json").write_text(json.dumps(summary, indent=2))
pd.DataFrame(
    {"fraction": FRACS, "true_fdp": true_fdps, "size": [len(s) for s in sets]}
).to_csv(OUT / "sembench_text_nested_sets.csv", index=False)
print(json.dumps(summary, indent=2))
