"""Study population and outcome (docs/project_description.md, section 3)."""
from __future__ import annotations

import pandas as pd

PHQ_ITEMS = [f"phq_{i}" for i in range(1, 10)]
SCREEN_CUTOFF = 10  # a PHQ score of 10 or more is a positive depression screen


def _is_adult(df: pd.DataFrame) -> pd.Series:
    return (df["age"] >= 18).fillna(False).astype(bool)


def select_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Adults (18+) who answered all nine PHQ-9 items."""
    keep = _is_adult(df) & df[PHQ_ITEMS].notna().all(axis=1)
    return df[keep].reset_index(drop=True)


def cohort_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Participants left after each selection step, per cycle and in total.

    Takes the harmonised table of all NHANES participants, before select_cohort.
    """
    adult = _is_adult(df)
    # The PHQ-9 is administered in the mobile examination centre (MEC); people who
    # were only interviewed at home have an examination weight of 0.
    examined = adult & (df["weight_mec"] > 0)
    answered = df[PHQ_ITEMS].notna()
    steps = pd.DataFrame({
        "participants": True,
        "adults": adult,
        "examined": examined,
        "phq9_started": examined & answered.any(axis=1),
        "phq9_complete": adult & answered.all(axis=1),
    })
    flow = steps.groupby(df["cycle"].to_numpy()).sum()
    flow.loc["total"] = steps.sum()
    return flow.astype("int64").rename_axis("cycle").reset_index()


def add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """PHQ scores, depression screens and the suicidal-ideation outcome (needs a complete PHQ-9)."""
    df = df.copy()
    df["phq8_score"] = df[PHQ_ITEMS[:8]].sum(axis=1, skipna=False).astype("Int16")
    df["phq9_score"] = df[PHQ_ITEMS].sum(axis=1, skipna=False).astype("Int16")
    df["phq8_positive"] = (df["phq8_score"] >= SCREEN_CUTOFF).astype("Int16")
    df["phq9_positive"] = (df["phq9_score"] >= SCREEN_CUTOFF).astype("Int16")
    df["si"] = (df["phq_9"] >= 1).astype("Int16")
    df["si_strict"] = (df["phq_9"] >= 2).astype("Int16")
    return df
