"""Frozen, label-blind protocol for the completion experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("AIR BAGS", "EXTERIOR LIGHTING")
KEYS = [("country", "year", "fuel_type"), ("country", "year", "transmission")]


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Split on parent groups so both grouping schemes remain disjoint."""
    folder = ROOT / "data/raw/full_data"
    text = pd.read_csv(folder / "text_complaints_data_full.csv")
    cars = pd.read_csv(folder / "car_data_full.csv")
    df = text.merge(cars, on="car_id", validate="many_to_one")
    df["summary"] = df.summary.fillna("").astype(str)
    df["parent"] = df[["country", "year"]].astype(str).agg("||".join, axis=1)
    tr, te = next(
        GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=20261026).split(
            df, groups=df.parent
        )
    )
    assert set(df.iloc[tr].parent).isdisjoint(df.iloc[te].parent)
    return df, tr, te


def selected_rows(df: pd.DataFrame, indices: np.ndarray, count: int) -> pd.DataFrame:
    """Select complete parent groups by a fixed hash without inspecting labels."""
    parents = sorted(
        df.iloc[indices].parent.unique(),
        key=lambda s: hashlib.sha256(("certineg-v9|" + s).encode()).hexdigest(),
    )[:count]
    return df.iloc[indices].loc[lambda x: x.parent.isin(parents)].copy()
