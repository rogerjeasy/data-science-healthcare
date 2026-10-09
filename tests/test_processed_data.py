"""Checks on the real processed data. Skipped until `scripts/build_dataset.py` has been run."""
import pandas as pd
import pytest

from hidden_risk.config import DATA_PROCESSED
from hidden_risk.variables import VARIABLES

PERSONS = DATA_PROCESSED / "nhanes_2005_2018.parquet"
PRESCRIPTIONS = DATA_PROCESSED / "nhanes_2005_2018_prescriptions.parquet"
TREND = DATA_PROCESSED / "nhanes_2021_2023_phq9.parquet"

pytestmark = pytest.mark.skipif(not PERSONS.exists(), reason="run scripts/build_dataset.py first")


@pytest.fixture(scope="module")
def persons():
    return pd.read_parquet(PERSONS)


def test_cohort_matches_project_description(persons):
    # docs/project_description.md, sections 3 and 4
    assert len(persons) == 36_259
    assert persons["si"].sum() == 1_366
    assert persons["si_strict"].sum() == 424
    train, test = persons[persons["split"] == "train"], persons[persons["split"] == "test"]
    assert (len(train), train["si"].sum()) == (26_057, 983)
    assert (len(test), test["si"].sum()) == (10_202, 383)
    assert len(pd.read_parquet(TREND)) == 5_455


def test_one_row_per_adult(persons):
    assert persons["seqn"].is_unique
    assert persons["age"].between(18, 80).all()


def test_no_stored_zeros_or_missing_codes_left(persons):
    numeric = persons.select_dtypes("number")
    assert not ((numeric.abs() < 1e-30) & (numeric != 0)).any().any()
    for v in VARIABLES:
        if v.kind == "binary":
            assert set(persons[v.name].dropna().unique()) <= {0, 1}, v.name
        elif v.values:
            assert set(persons[v.name].dropna().unique()) <= set(v.values), v.name


def test_prescriptions_match_reported_drug_count(persons):
    rx = pd.read_parquet(PRESCRIPTIONS)
    assert rx["seqn"].isin(persons["seqn"]).all()
    rows = rx.groupby("seqn").size().reindex(persons["seqn"], fill_value=0)
    assert (rows.to_numpy() == persons["rx_count"].fillna(0).to_numpy()).mean() > 0.999
