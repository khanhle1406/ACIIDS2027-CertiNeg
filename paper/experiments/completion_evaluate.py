"""Matched grouped workloads; all audit repetitions use cached predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from completion_data import KEYS, ROOT, TARGETS, load_data, selected_rows
from scipy.stats import beta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from certineg import hypergeom_upper_rate

OUT = ROOT / "data/results/completion"


def evaluate(
    frame: pd.DataFrame,
    keys: tuple[str, ...],
    threshold: float,
    method: str,
    target: str,
    corpus: str,
) -> list[dict]:
    """Freeze candidate groups before simulating randomized audits."""
    candidates = []
    for key, g in frame.groupby(list(keys), sort=True):
        if (g.score >= threshold).any():
            continue
        ordered = g.sort_values("score", ascending=False, kind="stable")
        labels = ordered.component_class.eq(target).to_numpy()
        cost = int(np.flatnonzero(labels)[0] + 1) if labels.any() else len(labels)
        candidates.append((float(g.score.max()), str(key), int(labels.any()), cost))
    candidates.sort()
    N = len(candidates)
    z = np.array([g[2] for g in candidates], dtype=int)
    costs = np.array([g[3] for g in candidates], dtype=int)
    prefix = (
        sorted({max(1, round(f * N)) for f in (0.3, 0.45, 0.6, 0.8, 1.0)}) if N else []
    )
    results = []
    for fraction in (0.35, 2 / 3):
        rng = np.random.default_rng(20261026)
        n = round(fraction * N)
        retention, bad, label_cost = [], 0, []
        for _ in range(2000):
            sample = rng.choice(N, size=n, replace=False)
            seen = np.zeros(N, dtype=int)
            seen[sample] = 1
            ns, xs = np.cumsum(seen), np.cumsum(seen * z)
            selected = 0
            for m in prefix:
                if hypergeom_upper_rate(m, int(ns[m - 1]), int(xs[m - 1]), 0.05) > 0.1:
                    break
                selected = m
            retention.append(selected / N if N else 0.0)
            bad += int(selected > 0 and z[:selected].mean() > 0.1)
            label_cost.append(int(costs[sample].sum()))
        results.append(
            {
                "corpus": corpus,
                "model": method,
                "predicate": target,
                "grouping": "/".join(keys),
                "candidate_groups": N,
                "candidate_fdp": float(z.mean()) if N else None,
                "threshold": threshold,
                "row_f1": f1_score(
                    frame.component_class.eq(target),
                    frame.score >= threshold,
                    zero_division=0,
                ),
                "audit_fraction": n / N if N else 0.0,
                "retention": float(np.mean(retention)),
                "retention_sd": float(np.std(retention, ddof=1)),
                "nonempty": float(np.mean(np.array(retention) > 0)),
                "violations": bad,
                "repetitions": 2000,
                "violation_upper95": float(beta.ppf(0.95, bad + 1, 2000 - bad))
                if bad < 2000
                else 1.0,
                "census_labels": int(costs.sum()),
                "audit_labels_mean": float(np.mean(label_cost)),
            }
        )
    return results


def main() -> None:
    """Train only on parent-disjoint data and evaluate every declared workload."""
    OUT.mkdir(exist_ok=True)
    df, tr, te = load_data()
    cal, test = selected_rows(df, tr, 16), selected_rows(df, te, 64)
    vec = TfidfVectorizer(max_features=20000, min_df=5, max_df=0.995, sublinear_tf=True)
    xtr = vec.fit_transform(df.iloc[tr].summary)
    xte = vec.transform(df.iloc[te].summary)
    rows = []
    for target in (
        "AIR BAGS",
        "ENGINE AND ENGINE COOLING",
        "VISIBILITY/WIPER",
        "EXTERIOR LIGHTING",
    ):
        model = SGDClassifier(
            loss="log_loss",
            alpha=2e-6,
            class_weight="balanced",
            max_iter=12,
            tol=1e-3,
            random_state=20261026,
        )
        model.fit(xtr, df.iloc[tr].component_class.eq(target))
        ptr = model.predict_proba(xtr)[:, 1]
        threshold = float(
            np.quantile(ptr[df.iloc[tr].component_class.eq(target)], 0.08)
        )
        full = df.iloc[te].copy().assign(score=model.predict_proba(xte)[:, 1])
        for keys in KEYS:
            rows.extend(
                evaluate(
                    full, keys, threshold, "TF-IDF/SGD", target, "full-parent-split"
                )
            )
        if target in TARGETS:
            pc = model.predict_proba(vec.transform(cal.summary))[:, 1]
            matched_threshold = float(
                np.quantile(pc[cal.component_class.eq(target)], 0.08)
            )
            small = full.loc[full.complaint_id.isin(test.complaint_id)]
            for keys in KEYS:
                rows.extend(
                    evaluate(
                        small,
                        keys,
                        matched_threshold,
                        "TF-IDF/SGD",
                        target,
                        "matched-subset",
                    )
                )
        print("CPU completed", target, flush=True)
    cache = OUT / "llm_scores.jsonl"
    if cache.exists():
        blocks = [json.loads(line) for line in cache.read_text().splitlines()]
        scores = pd.DataFrame([s for b in blocks for s in b.get("scores", [])])
        expected = set(pd.concat([cal, test]).complaint_id)
        if not scores.empty and all(
            set(scores.loc[scores.target.eq(t)].complaint_id) == expected
            for t in TARGETS
        ):
            for target in TARGETS:
                joined = df.merge(
                    scores.loc[scores.target.eq(target)],
                    on="complaint_id",
                    validate="one_to_one",
                )
                calibration = joined.loc[joined.complaint_id.isin(cal.complaint_id)]
                threshold = float(
                    np.quantile(
                        calibration.loc[
                            calibration.component_class.eq(target), "score"
                        ],
                        0.08,
                    )
                )
                heldout = joined.loc[joined.complaint_id.isin(test.complaint_id)]
                for keys in KEYS:
                    rows.extend(
                        evaluate(
                            heldout,
                            keys,
                            threshold,
                            "Qwen3-0.6B",
                            target,
                            "matched-subset",
                        )
                    )
        else:
            print(
                "LLM cache incomplete: no partial-corpus inference reported", flush=True
            )
    pd.DataFrame(rows).to_csv(OUT / "workloads.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
