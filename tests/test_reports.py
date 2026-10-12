import pandas as pd
import pytest

from hidden_risk.reports import cohort_by_cycle, cross_cycle_rules, missing_code_groups
from hidden_risk.variables import Var

RULE_VARS = [
    Var("visits", "HUQ", {"DEFG": "HUQ050", "HIJ": "HUQ051"}, "Visits", missing=(77, 99),
        note="2013 categories collapsed."),
    Var("drugs", "DUQ", "DUQ240", "Hard drugs", missing=(7, 9), min_age=20, max_age=59),
    Var("smoke", "SMQ", "SMQ020", "Smoked", missing=(7, 9), min_age=20),
    Var("copd", "MCQ", {"HIJ": "MCQ160O"}, "COPD", missing=(7, 9)),
    Var("sex", "DEMO", "RIAGENDR", "Sex"),
]


def test_cohort_table_reports_cases_missed_by_the_phq8_and_by_the_phq9_screen():
    df = pd.DataFrame({
        "cycle": ["2015-2016"] * 6,
        "split": ["test"] * 6,
        "si":         [1, 1, 1, 1, 0, 0],
        "si_strict":  [0, 0, 1, 0, 0, 0],
        "phq8_score": [3, 9, 9, 12, 0, 15],
        "phq9_score": [4, 10, 11, 13, 0, 15],  # only the first case stays below 10
    })
    row = cohort_by_cycle(df).set_index("cycle").loc["2015-2016"]
    assert (row["adults"], row["si_cases"], row["si_strict_cases"]) == (6, 4, 1)
    assert row["si_with_phq8_below_10_pct"] == 75.0
    assert row["si_with_phq9_below_10_pct"] == 25.0


def test_rules_table_lists_only_variables_that_differ_between_cycles():
    rules = cross_cycle_rules(RULE_VARS).set_index("variable")
    assert list(rules.index) == ["visits", "drugs", "smoke", "copd"]
    assert rules.loc["visits", "source"] == "`HUQ050` (2005-2012), `HUQ051` (2013-2018)"
    assert rules.loc["copd", "source"] == "`MCQ160O` (2013-2018)"
    assert rules["ages_kept"].to_dict() == {"visits": "all", "drugs": "20-59", "smoke": "20+", "copd": "all"}
    assert rules.loc["visits", "rule"] == "2013 categories collapsed."


def test_refused_and_dont_know_codes_are_grouped_per_code_set():
    assert missing_code_groups(RULE_VARS) == {
        "77, 99": ["visits"], "7, 9": ["drugs", "smoke", "copd"], "none": ["sex"]}


def test_a_transform_without_a_written_rule_cannot_reach_the_appendix():
    undocumented = Var("bmi_capped", "BMX", "BMXBMI", "BMI", kind="continuous",
                       transform=lambda s, cycle, raw: s.clip(upper=60))
    with pytest.raises(ValueError, match="bmi_capped"):
        cross_cycle_rules([undocumented])
