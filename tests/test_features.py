import pandas as pd

from hidden_risk.config import DATA_PROCESSED, DATA_RAW
from hidden_risk.features import CONDITIONS, DRUG_FLAGS, build_features
from hidden_risk.io import read_xpt

PERSONS = DATA_PROCESSED / "nhanes_2005_2018.parquet"
PRESCRIPTIONS = DATA_PROCESSED / "nhanes_2005_2018_prescriptions.parquet"


def features():
    return build_features(
        pd.read_parquet(PERSONS),
        pd.read_parquet(PRESCRIPTIONS),
        read_xpt("RXQ_DRUG", DATA_RAW),
    )


def test_one_row_per_adult_and_the_same_people():
    persons = pd.read_parquet(PERSONS, columns=["seqn"])
    out = features()
    assert len(out) == 36_259
    assert out["seqn"].tolist() == persons["seqn"].tolist()


def test_drug_classes_match_the_lookup():
    out = features()
    assert out[DRUG_FLAGS].sum().tolist() == [3736, 2365, 1713, 813]
    none = pd.read_parquet(PERSONS, columns=["rx_any"]).rx_any.eq(0).fillna(False)
    assert (out.loc[none.to_numpy(), DRUG_FLAGS + ["n_drugs"]] == 0).all().all()


def test_questions_that_were_not_asked_stay_missing():
    out = features()
    young, older = out["age"] < 20, out["age"] > 59
    assert (out.loc[young, "not_asked_under_20"] == 1).all()
    assert out.loc[young, "arthritis"].isna().all()
    assert (out.loc[young | older, "not_asked_drug_use"] == 1).all()
    assert out.loc[young | older, "ever_marijuana"].isna().all()
    assert out["multimorbidity"].dropna().between(0, len(CONDITIONS)).all()


def test_outcome_columns_are_not_features():
    forbidden = {"si", "si_strict", "phq_9", "phq9_score", "phq9_positive"}
    assert forbidden.isdisjoint(features().columns)