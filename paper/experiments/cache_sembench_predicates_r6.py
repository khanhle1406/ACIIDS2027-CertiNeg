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
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    type=Path,
    default=ROOT.parent / "data/raw/full_data",
)
args = parser.parse_args()
DATA = args.data_dir
sys.path.insert(0, str(ROOT / "src"))
from certineg import group_inspection_cost

SEED = 20261026
TARGETS = [
    "AIR BAGS",
    "ENGINE AND ENGINE COOLING",
    "VISIBILITY/WIPER",
    "EXTERIOR LIGHTING",
]
keys = ["country", "year", "fuel_type"]
text = pd.read_csv(
    DATA / "text_complaints_data_full.csv",
    usecols=["car_id", "summary", "component_class"],
)
car = pd.read_csv(
    DATA / "car_data_full.csv", usecols=["car_id", "country", "year", "fuel_type"]
)
df = text.merge(car, on="car_id", how="inner")
df["summary"] = df.summary.fillna("").astype(str)
gid = df[keys].astype(str).agg("||".join, axis=1)
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
meta = {"dataset_rows_total": len(df), "test_rows": len(te), "targets": {}}
for target in TARGETS:
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
    thr = float(np.quantile(ptr[ytr == 1], 0.08))
    pred = (p >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        yte, pred, average="binary", zero_division=0
    )
    auc = float(roc_auc_score(yte, p))
    test = df.iloc[te][keys].copy().reset_index(drop=True)
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
    z = np.array([g[3] for g in groups], dtype=np.int8)
    risk = np.array([g[4] for g in groups], dtype=float)
    cost = np.array([group_inspection_cost(g[1], g[2]) for g in groups], dtype=np.int32)
    sizes = np.array([g[5] for g in groups], dtype=np.int32)
    safe = "".join(c if c.isalnum() else "_" for c in target.lower()).strip("_")
    np.savez_compressed(
        OUT / f"sembench_{safe}_groups.npz", z=z, risk=risk, cost=cost, sizes=sizes
    )
    meta["targets"][target] = {
        "threshold": thr,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": auc,
        "candidate_groups": len(z),
        "candidate_fdp": float(z.mean()),
        "census_cost": int(cost.sum()),
        "group_size_median": float(np.median(sizes)),
        "group_size_p95": float(np.quantile(sizes, 0.95)),
    }
    print(target, meta["targets"][target], flush=True)
(OUT / "sembench_predicate_cache_meta.json").write_text(json.dumps(meta, indent=2))
