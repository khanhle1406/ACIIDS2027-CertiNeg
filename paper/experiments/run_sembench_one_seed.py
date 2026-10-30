import argparse
import json
import os
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

from certineg import hypergeom_upper_rate

EPS = 0.10
DELTA = 0.05
TARGET = "AIR BAGS"
FRACS = np.array([0.30, 0.45, 0.60, 0.80, 1.00])
R = 2000
seeds = [int(os.environ.get("ONE_SEED", "20261026"))]
keys = ["country", "year", "fuel_type"]


def sizes(N):
    out = []
    p = 0
    for f in FRACS:
        m = max(p, min(N, round(float(f) * N)))
        out.append(m)
        p = m
    return np.array(out, dtype=int)


def fixed(sample, z, ps):
    N = len(z)
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
    return k


def simulate(z, n, seed):
    N = len(z)
    ps = sizes(N)
    fdps = np.array([z[:m].mean() for m in ps])
    rng = np.random.default_rng(seed)
    ret = []
    vio = 0
    for _ in range(R):
        s = rng.choice(N, size=min(n, N), replace=False)
        k = fixed(s, z, ps)
        rr = 0 if k < 0 else ps[k] / N
        ret.append(rr)
        vio += int(k >= 0 and fdps[k] > EPS)
    return np.mean(ret), np.std(ret, ddof=1), vio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SemBench evaluation for one seed on AIR BAGS."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT.parent / "data/raw/full_data",
        help="Path to full_data directory containing car_data_full.csv and text_complaints_data_full.csv",
    )
    args = parser.parse_args()
    DATA = args.data_dir

    text = pd.read_csv(
        DATA / "text_complaints_data_full.csv",
        usecols=["car_id", "summary", "component_class"],
    )
    car = pd.read_csv(
        DATA / "car_data_full.csv",
        usecols=["car_id", "country", "year", "fuel_type"],
    )
    df = text.merge(car, on="car_id", how="inner")
    df["summary"] = df.summary.fillna("").astype(str)
    gid = df[keys].astype(str).agg("||".join, axis=1)
    yall = df.component_class.eq(TARGET).astype(np.int8).to_numpy()

    rows = []
    for seed in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=seed)
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
        ytr = yall[tr]
        yte = yall[te]
        clf = SGDClassifier(
            loss="log_loss",
            alpha=2e-6,
            class_weight="balanced",
            max_iter=12,
            tol=1e-3,
            random_state=seed,
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
            cand.append((float(g.score.max()), int(g.y.any())))
        cand.sort(key=lambda x: x[0])
        z = np.array([x[1] for x in cand], dtype=np.int8)
        N = len(z)
        ps = sizes(N)
        fdps = [float(z[:m].mean()) for m in ps]
        oracle = max([ps[i] / N for i, v in enumerate(fdps) if v <= EPS], default=0.0)
        cs = np.cumsum(z) / np.arange(1, N + 1)
        ok = np.where((np.arange(1, N + 1) >= int(np.ceil(0.30 * N))) & (cs <= EPS))[0]
        cont = (ok[-1] + 1) / N if len(ok) else 0.0
        n = round(0.35 * N)
        mr, sr, vio = simulate(z, n, seed + 50000)
        row = {
            "seed": seed,
            "test_rows": len(te),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": auc,
            "candidate_groups": N,
            "candidate_fdp": float(z.mean()),
            "oracle_K5": oracle,
            "oracle_continuous": cont,
            "audit_n": n,
            "audit_fraction": n / N,
            "mean_retention": float(mr),
            "sd_retention_audit": float(sr),
            "violation_count": vio,
            "violation_rate": vio / R,
        }
        rows.append(row)
        print(row, flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"sembench_airbags_split_{seeds[0]}.csv", index=False)
    summary = {}
    for c in [
        "precision",
        "recall",
        "f1",
        "auc",
        "candidate_groups",
        "candidate_fdp",
        "oracle_K5",
        "oracle_continuous",
        "audit_fraction",
        "mean_retention",
        "violation_rate",
    ]:
        summary[c + "_mean"] = float(out[c].mean())
        summary[c + "_sd_across_splits"] = float(out[c].std(ddof=1))
    summary["total_false_certificates"] = int(out.violation_count.sum())
    summary["total_audit_randomizations"] = int(R * len(out))
    (OUT / f"sembench_airbags_split_{seeds[0]}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("\nSUMMARY", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
