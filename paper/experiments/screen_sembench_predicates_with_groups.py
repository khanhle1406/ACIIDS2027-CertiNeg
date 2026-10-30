import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Screen SemBench predicates and persist group data."
)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=ROOT.parent / "data/raw/full_data",
    help="Path to full_data directory containing car_data_full.csv and text_complaints_data_full.csv",
)
parser.add_argument(
    "--results-dir",
    type=Path,
    default=ROOT.parent / "data/results",
    help="Directory to save output results",
)
args = parser.parse_args()
DATA = args.data_dir
OUT = args.results_dir
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261026
keys = ["country", "year", "fuel_type"]


def safe_name(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


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
targets = df.iloc[tr].component_class.value_counts().head(15).index.tolist()
rows = []
for target in targets:
    yall = df.component_class.eq(target).astype(np.int8).to_numpy()
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
    pred = (p >= thr).astype(np.int8)
    prec, rec, f1, _ = precision_recall_fscore_support(
        yte, pred, average="binary", zero_division=0
    )
    auc = float(roc_auc_score(yte, p))
    test = df.iloc[te][keys].copy().reset_index(drop=True)
    test["score"] = p
    test["pred"] = pred
    test["y"] = yte
    cand = []
    for key, g in test.groupby(keys, sort=True):
        if bool(g.pred.any()):
            continue
        # Candidate group risk order = ascending max semantic score, matching v7 screen.
        cand.append((float(g.score.max()), int(g.y.any()), len(g)))
    cand.sort(key=lambda x: x[0])
    z = np.array([x[1] for x in cand], dtype=np.int8)
    sizes = np.array([x[2] for x in cand], dtype=np.int32)
    scores = np.array([x[0] for x in cand], dtype=np.float64)
    np.savez_compressed(
        OUT / f"sembench_screen_{safe_name(target)}_groups.npz",
        z=z,
        size=sizes,
        score=scores,
    )
    N = len(z)
    fdp = float(z.mean()) if N else float("nan")
    oracle = 0.0
    cont = 0.0
    fracs = [0.30, 0.45, 0.60, 0.80, 1.0]
    if N:
        for f in fracs:
            m = max(1, min(N, round(f * N)))
            if z[:m].mean() <= 0.10:
                oracle = max(oracle, m / N)
        cs = np.cumsum(z) / np.arange(1, N + 1)
        ok = np.where((np.arange(1, N + 1) >= int(np.ceil(0.30 * N))) & (cs <= 0.10))[0]
        cont = (ok[-1] + 1) / N if len(ok) else 0.0
    row = {
        "predicate": target,
        "train_pos": int(ytr.sum()),
        "test_pos": int(yte.sum()),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": auc,
        "candidate_groups": N,
        "candidate_fdp": fdp,
        "oracle_K5": oracle,
        "oracle_continuous": cont,
    }
    rows.append(row)
    print(target, N, fdp, oracle, flush=True)
pd.DataFrame(rows).to_csv(OUT / "sembench_predicate_screen_r8.csv", index=False)
(OUT / "sembench_predicate_screen_r8_meta.json").write_text(
    json.dumps({"seed": SEED, "targets": targets, "data_dir": str(DATA)}, indent=2)
)
