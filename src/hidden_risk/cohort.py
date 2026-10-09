"""Study population and outcome (docs/project_description.md, section 3)."""
from __future__ import annotations

import pandas as pd

PHQ_ITEMS = [f"phq_{i}" for i in range(1, 10)]


def select_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Adults (18+) who answered all nine PHQ-9 items."""
    keep = (df["age"] >= 18) & df[PHQ_ITEMS].notna().all(axis=1)
    return df[keep.fillna(False)].reset_index(drop=True)


def add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """PHQ scores, depression screen and the suicidal-ideation outcome (needs a complete PHQ-9)."""
    df = df.copy()
    df["phq8_score"] = df[PHQ_ITEMS[:8]].sum(axis=1, skipna=False).astype("Int16")
    df["phq9_score"] = df[PHQ_ITEMS].sum(axis=1, skipna=False).astype("Int16")
    df["phq8_positive"] = (df["phq8_score"] >= 10).astype("Int16")
    df["si"] = (df["phq_9"] >= 1).astype("Int16")
    df["si_strict"] = (df["phq_9"] >= 2).astype("Int16")
    return df
