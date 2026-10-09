import numpy as np
import pandas as pd
import pytest

from hidden_risk.cohort import add_outcome, select_cohort
from hidden_risk.harmonise import harmonise_variable
from hidden_risk.io import clean_xpt_frame
from hidden_risk.variables import BY_NAME, VARIABLES

SAS_ZERO = 5.397605346934028e-79


def raw_frame(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    return pd.DataFrame({"SEQN": np.arange(n, dtype=float), "RIDAGEYR": [40.0] * n, **cols})


def test_stored_zero_becomes_zero_and_missing_stays_missing():
    df = clean_xpt_frame(pd.DataFrame({"SEQN": [1.0, 2.0, 3.0], "DPQ090": [SAS_ZERO, 2.0, np.nan]}))
    assert df["DPQ090"].tolist()[:2] == [0.0, 2.0]
    assert np.isnan(df["DPQ090"].iloc[2])
    assert df["SEQN"].dtype == "int64"


def test_refused_and_dont_know_become_missing_but_real_values_survive():
    raw = raw_frame(HUQ071=[1.0, 2.0, 7.0, 9.0])
    assert harmonise_variable(BY_NAME["hospital_stay"], raw, "D").tolist() == [1, 0, pd.NA, pd.NA]
    # 7 and 9 hours are real answers for sleep; only 77 / 99 are codes
    raw = raw_frame(SLD010H=[7.0, 9.0, 77.0, 99.0])
    assert harmonise_variable(BY_NAME["sleep_hours"], raw, "D").tolist()[:2] == [7.0, 9.0]
    assert harmonise_variable(BY_NAME["sleep_hours"], raw, "D").isna().tolist() == [False, False, True, True]


def test_healthcare_visits_2013_categories_collapse_to_earlier_scale():
    old = harmonise_variable(BY_NAME["healthcare_visits"], raw_frame(HUQ050=[0.0, 3.0, 5.0, 99.0]), "G")
    new = harmonise_variable(BY_NAME["healthcare_visits"],
                             raw_frame(HUQ051=[0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 77.0]), "H")
    assert old.tolist() == [0, 3, 5, pd.NA]
    assert new.tolist() == [0, 3, 3, 3, 4, 5, 5, pd.NA]


def test_sleep_hours_comparable_across_modules():
    new = harmonise_variable(BY_NAME["sleep_hours"], raw_frame(SLD012=[2.0, 7.5, 14.5]), "I")
    assert new.tolist() == [2.0, 7.5, 12.0]


def test_age_topcoded_at_80_in_every_cycle():
    raw = raw_frame(RIDAGEYR=[18.0, 80.0, 85.0])
    assert harmonise_variable(BY_NAME["age"], raw, "D").tolist() == [18, 80, 80]


def test_education_for_18_19_year_olds_comes_from_youth_question():
    raw = pd.DataFrame({"SEQN": [1.0, 2.0, 3.0, 4.0], "RIDAGEYR": [18.0, 19.0, 19.0, 30.0],
                        "DMDEDUC2": [np.nan, np.nan, np.nan, 5.0],
                        "DMDEDUC3": [11.0, 13.0, 15.0, np.nan]})
    assert harmonise_variable(BY_NAME["education"], raw, "E").tolist() == [2, 3, 4, 5]


def test_age_restricted_questions_cover_the_same_ages_in_every_cycle():
    raw = pd.DataFrame({"SEQN": [1.0, 2.0, 3.0], "RIDAGEYR": [19.0, 40.0, 65.0],
                        "DUQ240": [1.0, 1.0, 1.0], "SMQ020": [1.0, 1.0, 1.0]})
    assert harmonise_variable(BY_NAME["ever_hard_drugs"], raw, "J").tolist() == [pd.NA, 1, pd.NA]
    assert harmonise_variable(BY_NAME["smoked_100_cigarettes"], raw, "J").tolist() == [pd.NA, 1, 1]


def test_cholesterol_never_checked_counts_as_never_told_before_2011():
    raw = raw_frame(BPQ080=[np.nan, np.nan, 1.0], BPQ060=[2.0, 9.0, 1.0])
    assert harmonise_variable(BY_NAME["high_cholesterol"], raw, "D").tolist() == [0, pd.NA, 1]


def test_variable_not_collected_in_a_cycle_is_all_missing():
    assert harmonise_variable(BY_NAME["copd"], raw_frame(HUQ010=[1.0, 2.0]), "D").isna().all()


def test_variable_missing_from_a_file_raises():
    with pytest.raises(KeyError):
        harmonise_variable(BY_NAME["hospital_stay"], raw_frame(HUQ010=[1.0]), "D")


def test_variable_names_are_unique():
    names = [v.name for v in VARIABLES]
    assert len(names) == len(set(names))


def test_cohort_keeps_adults_with_complete_phq9_and_defines_outcome():
    items = {f"phq_{i}": [0, 0, 0, 0] for i in range(1, 10)}
    df = pd.DataFrame({"age": [17, 18, 30, 50], **items}).astype("Int16")
    df.loc[2, "phq_3"] = pd.NA  # incomplete PHQ-9
    df.loc[3, ["phq_1", "phq_2", "phq_9"]] = [3, 3, 2]
    out = add_outcome(select_cohort(df))
    assert out["age"].tolist() == [18, 50]
    assert out["phq8_score"].tolist() == [0, 6]
    assert out["si"].tolist() == [0, 1]
    assert out["si_strict"].tolist() == [0, 1]
