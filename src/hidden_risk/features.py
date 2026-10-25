"""Step 2: medication classes, multimorbidity, social flags, missingness indicators."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW
from .io import read_xpt

DRUG_FLAGS = [
    "rx_antidepressant",
    "rx_opioid",
    "rx_benzo_z",
    "rx_antipsychotic_mood",
]
HEART = ["heart_failure", "coronary_heart_disease", "angina", "heart_attack"]
LUNG = ["emphysema", "chronic_bronchitis"]
CONDITIONS = [
    "arthritis", "heart_disease", "stroke", "lung_disease", "cancer",
    "diabetes", "hypertension", "asthma", "liver_condition", "thyroid_problem",
]
COPIED = [
    "seqn", "cycle", "split", "age", "sex", "ethnicity",
    "hospital_stay", "mental_health_visit", "healthcare_visits", "self_rated_health",
    "income_poverty_ratio", "education", "marital_status",
    "sleep_trouble", "sleep_hours", "bmi", "high_cholesterol",
    "smoked_100_cigarettes",
    "ever_marijuana", "ever_hard_drugs", "ever_injected_drugs",
]
Z_DRUGS = "ZOLPIDEM|ZALEPLON|ESZOPICLONE"
MOOD_NAMES = r"LITHIUM|VALPRO|DIVALPROEX|LAMOTRIGINE|CARBAMAZEPINE|OXCARBAZEPINE"


def _category_text(lookup: pd.DataFrame) -> pd.Series:
    cols = [c for c in lookup.columns if c.startswith(("RXDDCN", "RXDICN"))]
    text = lookup[cols[0]].astype("string").str.upper().fillna("")
    for col in cols[1:]:
        text = text + " | " + lookup[col].astype("string").str.upper().fillna("")
    return text


def class_by_drug_id(lookup: pd.DataFrame) -> pd.DataFrame:
    """One row per drug id: True if any category name puts it in that class."""
    text = _category_text(lookup)
    name = lookup["RXDDRUG"].astype("string").str.upper().fillna("")
    return pd.DataFrame({
        "drug_id": lookup["RXDDRGID"],
        "rx_antidepressant": text.str.contains("ANTIDEPRESSANT", regex=False),
        "rx_opioid": text.str.contains("NARCOTIC ANALGESIC", regex=False),
        "rx_benzo_z": text.str.contains("BENZODIAZEPINE", regex=False) | name.str.contains(Z_DRUGS),
        "rx_antipsychotic_mood": (
            text.str.contains("ANTIPSYCHOTIC|ANTIMANIC")
            | name.str.contains(MOOD_NAMES)
        ),
    })


def _yes(block: pd.DataFrame) -> pd.Series:
    """1 if any value is 1, 0 if something is known and nothing is 1, NA if all are missing."""
    yes = block.eq(1).any(axis=1)
    known = block.notna().any(axis=1)
    out = pd.Series(pd.NA, index=block.index, dtype="Int16")
    return out.mask(known, 0).mask(yes, 1).astype("Int16")


def _flag(series: pd.Series, when_true: pd.Series) -> pd.Series:
    """1 where when_true, 0 where series is known, NA where series is missing."""
    out = pd.Series(pd.NA, index=series.index, dtype="Int16")
    # pandas treats NA in a mask condition as True, so missing answers must be False explicitly
    return out.mask(series.notna(), 0).mask(when_true.fillna(False).astype(bool), 1).astype("Int16")


def medication_features(persons: pd.DataFrame, prescriptions: pd.DataFrame,
                        lookup: pd.DataFrame) -> pd.DataFrame:
    classes = class_by_drug_id(lookup)
    tagged = prescriptions.merge(classes, on="drug_id", how="left")
    per_person = tagged.groupby("seqn", as_index=False)[DRUG_FLAGS].max()
    n_drugs = (prescriptions.dropna(subset=["drug_id"])
               .groupby("seqn")["drug_id"].nunique()
               .rename("n_drugs"))
    out = (persons[["seqn", "rx_any"]]
           .merge(per_person, on="seqn", how="left")
           .merge(n_drugs, on="seqn", how="left"))
    none = out["rx_any"].eq(0)
    unknown = out["rx_any"].isna()
    for col in DRUG_FLAGS:
        out[col] = out[col].astype("Float64").fillna(0).where(~unknown)
        out.loc[none, col] = 0
        out[col] = out[col].astype("Int16")
    out["n_drugs"] = out["n_drugs"].astype("Float64").fillna(0).where(~unknown)
    out.loc[none, "n_drugs"] = 0
    out.loc[unknown, "n_drugs"] = pd.NA
    out["n_drugs"] = out["n_drugs"].astype("Int16")
    return out.drop(columns="rx_any")


def condition_features(persons: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame({
        "arthritis": _yes(persons[["arthritis"]]),
        "heart_disease": _yes(persons[HEART]),
        "stroke": _yes(persons[["stroke"]]),
        "lung_disease": _yes(persons[LUNG]),
        "cancer": _yes(persons[["cancer"]]),
        "diabetes": _yes(persons[["diabetes"]]),
        "hypertension": _yes(persons[["hypertension"]]),
        "asthma": _yes(persons[["asthma"]]),
        "liver_condition": _yes(persons[["liver_condition"]]),
        "thyroid_problem": _yes(persons[["thyroid_problem"]]),
    })
    flags["multimorbidity"] = flags[CONDITIONS].sum(axis=1, min_count=1).astype("Int16")
    return flags


def build_features(persons: pd.DataFrame, prescriptions: pd.DataFrame,
                   lookup: pd.DataFrame) -> pd.DataFrame:
    """One row per adult. Does not read phq_9, phq9_score, si or si_strict."""
    age = persons["age"]
    work = persons["work_status"]
    # smoking_now is only asked if smoked_100_cigarettes = yes: never smokers are not current smokers
    smoking = persons["smoking_now"].mask(persons["smoked_100_cigarettes"].eq(0).fillna(False).astype(bool), 3)
    extra = pd.DataFrame({
        "not_working": _flag(work, work.eq(4)),
        "looking_for_work": _flag(work, work.eq(3)),
        "uninsured": _flag(persons["insured"], persons["insured"].eq(0)),
        "current_smoker": _flag(smoking, smoking.isin([1, 2])),
        "not_asked_under_20": (age < 20).astype("Int16"),
        "not_asked_drug_use": ((age < 20) | (age > 59)).astype("Int16"),
    })
    return pd.concat(
        [
            persons[COPIED].reset_index(drop=True),
            medication_features(persons, prescriptions, lookup).drop(columns="seqn").reset_index(drop=True),
            condition_features(persons).reset_index(drop=True),
            extra.reset_index(drop=True),
        ],
        axis=1,
    )


def main(argv=None) -> int:
    persons = pd.read_parquet(DATA_PROCESSED / "nhanes_2005_2018.parquet")
    prescriptions = pd.read_parquet(DATA_PROCESSED / "nhanes_2005_2018_prescriptions.parquet")
    features = build_features(persons, prescriptions, read_xpt("RXQ_DRUG", DATA_RAW))
    path = DATA_PROCESSED / "nhanes_2005_2018_features.parquet"
    features.to_parquet(path, index=False)
    counts = features[DRUG_FLAGS].sum()
    print(f"Wrote {len(features):,} rows x {features.shape[1]} columns -> {path}")
    print(" ".join(f"{name}={int(counts[name])}" for name in DRUG_FLAGS))
    return 0