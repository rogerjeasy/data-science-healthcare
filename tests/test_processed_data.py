"""Checks on the real processed data. Skipped until `scripts/build_dataset.py` has been run."""
import pandas as pd
import pytest

from hidden_risk.config import CYCLES, DATA_PROCESSED, REPORT_TABLES
from hidden_risk.variables import DERIVED, VARIABLES

PERSONS = DATA_PROCESSED / "nhanes_2005_2018.parquet"
PRESCRIPTIONS = DATA_PROCESSED / "nhanes_2005_2018_prescriptions.parquet"
TREND = DATA_PROCESSED / "nhanes_2021_2023_phq9.parquet"

pytestmark = pytest.mark.skipif(not PERSONS.exists(), reason="run scripts/build_dataset.py first")


@pytest.fixture(scope="module")
def persons():
    return pd.read_parquet(PERSONS)


def test_cohort_matches_project_description(persons):
    # docs/project_description.md, sections 1, 3 and 4
    assert len(persons) == 36_259
    assert persons["si"].sum() == 1_366
    assert persons["si_strict"].sum() == 424
    train, test = persons[persons["split"] == "train"], persons[persons["split"] == "test"]
    assert (len(train), train["si"].sum()) == (26_057, 983)
    assert (len(test), test["si"].sum()) == (10_202, 383)
    assert len(pd.read_parquet(TREND)) == 5_455
    # "42% ... scored below 10 on the other eight PHQ items": 579 of 1,366
    cases = persons[persons["si"] == 1]
    assert (cases["phq8_positive"] == 0).sum() == 579


def test_phq9_screen_misses_fewer_cases_than_the_phq8_screen(persons):
    # Item 9 is the outcome, so it can only move cases from screen-negative to screen-positive
    cases = persons[persons["si"] == 1]
    assert (cases["phq9_positive"] == 0).sum() == 472
    moved = persons[(persons["phq8_positive"] == 0) & (persons["phq9_positive"] == 1)]
    assert len(moved) == 107
    assert (moved["si"] == 1).all()


def test_cohort_flow_matches_counts_taken_directly_from_the_raw_files():
    # Counted with pandas alone from DEMO and DPQ (examined = RIDSTATR 2), without project code
    flow = pd.read_csv(REPORT_TABLES / "cohort_flow_by_cycle.csv", index_col="cycle")
    assert flow.loc["2005-2006"].tolist() == [10_348, 5_563, 5_334, 4_836, 4_799]
    assert flow.loc["2017-2018"].tolist() == [9_254, 5_856, 5_533, 5_093, 5_068]
    assert flow.loc["total"].tolist() == [70_190, 42_143, 40_496, 36_459, 36_259]


def test_every_column_is_described_in_the_codebook(persons):
    described = {v.name for v in VARIABLES} | set(DERIVED)
    assert set(persons.columns) == described


def test_missingness_within_an_age_band_does_not_depend_on_the_cycle(persons):
    # "Each variable is kept only for the ages asked in every cycle": otherwise a question
    # asked from 18 in newer cycles and from 20 in older ones would reveal the survey year.
    # Real non-response differs by at most 10 points; a broken age rule by 50 or more.
    collected_every_cycle = [v.name for v in VARIABLES if all(v.source_in(c) for c in CYCLES)]
    for low, high in [(18, 19), (20, 59), (60, 80)]:
        band = persons[persons["age"].between(low, high)]
        share = band[collected_every_cycle].isna().groupby(band["cycle"]).mean()
        spread = share.max() - share.min()
        assert spread[spread > 0.2].empty, f"ages {low}-{high}: {spread[spread > 0.2].to_dict()}"


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
