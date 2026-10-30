"""Validate cache completeness, input isolation and numerical result provenance."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from completion_data import ROOT, TARGETS, load_data, selected_rows


def main() -> None:
    """Fail explicitly rather than silently report an incomplete LLM workload."""
    out = ROOT / "data/results/completion"
    df, tr, te = load_data()
    calibration, test = selected_rows(df, tr, 16), selected_rows(df, te, 64)
    membership = pd.read_csv(out / "llm_membership.csv")
    assert set(membership.loc[membership.part.eq("test"), "complaint_id"]) == set(
        test.complaint_id
    )
    assert set(
        membership.loc[membership.part.eq("calibration"), "complaint_id"]
    ) == set(calibration.complaint_id)
    blocks = [
        json.loads(line) for line in (out / "llm_scores.jsonl").read_text().splitlines()
    ]
    assert len(blocks) == 80 and all("scores" in b for b in blocks)
    scores = pd.DataFrame([s for b in blocks for s in b["scores"]])
    assert not scores.duplicated(["complaint_id", "target"]).any()
    assert np.isfinite(scores.score).all() and scores.score.between(0, 1).all()
    expected = set(membership.complaint_id)
    for target in TARGETS:
        assert set(scores.loc[scores.target.eq(target), "complaint_id"]) == expected
    results = pd.read_csv(out / "workloads.csv")
    assert len(results) == 32
    assert len(results.loc[results.model.eq("Qwen3-0.6B")]) == 8
    assert results.retention.between(0, 1).all()
    assert results.violations.between(0, results.repetitions).all()
    report = {
        "parent_blocks": len(blocks),
        "input_rows": len(expected),
        "scores": len(scores),
        "workload_cells": len(results),
        "truncated_fraction": float(scores.truncated.mean()),
        "remote_elapsed_seconds": max(b["elapsed_seconds"] for b in blocks),
        "status": "verified",
    }
    (out / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
