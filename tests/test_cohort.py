import pandas as pd

from hidden_risk.cohort import add_outcome, cohort_flow, select_cohort

PHQ = [f"phq_{i}" for i in range(1, 10)]


def phq_frame(rows: list[dict]) -> pd.DataFrame:
    """Harmonised-style frame; PHQ items default to 0 (not at all)."""
    defaults = {"cycle": "2005-2006", "age": 40, "weight_mec": 1000.0, **{item: 0 for item in PHQ}}
    df = pd.DataFrame([{**defaults, **row} for row in rows])
    return df.astype({item: "Int16" for item in PHQ} | {"age": "Int16"})


def test_item_9_can_turn_a_negative_phq8_screen_into_a_positive_phq9_screen():
    # PHQ-8 = 9 in both rows; only the second also reports item 9
    df = phq_frame([{"phq_1": 3, "phq_2": 3, "phq_3": 3},
                    {"phq_1": 3, "phq_2": 3, "phq_3": 3, "phq_9": 1}])
    out = add_outcome(df)
    assert out["phq8_positive"].tolist() == [0, 0]
    assert out["phq9_positive"].tolist() == [0, 1]


def test_cohort_flow_counts_each_exclusion_step_per_cycle():
    df = phq_frame([
        {"age": 10},                                       # child
        {"age": 17},                                       # child
        {"age": 18, "weight_mec": 0.0, **{i: pd.NA for i in PHQ}},  # adult, interview only
        {"age": 30, **{i: pd.NA for i in PHQ}},            # examined, PHQ-9 not administered
        {"age": 40, "phq_3": pd.NA},                       # PHQ-9 started but incomplete
        {"age": 50},                                       # in the cohort
        {"age": 60, "cycle": "2007-2008"},                 # in the cohort, other cycle
    ])
    flow = cohort_flow(df).set_index("cycle")
    assert flow.loc["2005-2006"].to_dict() == {
        "participants": 6, "adults": 4, "examined": 3, "phq9_started": 2, "phq9_complete": 1}
    assert flow.loc["2007-2008"].to_dict() == {
        "participants": 1, "adults": 1, "examined": 1, "phq9_started": 1, "phq9_complete": 1}
    assert flow.loc["total"].to_dict() == {
        "participants": 7, "adults": 5, "examined": 4, "phq9_started": 3, "phq9_complete": 2}


def test_last_flow_step_is_the_selected_cohort():
    df = phq_frame([{"age": 17}, {"age": 18}, {"age": 40, "phq_9": pd.NA}, {"age": 80, "phq_9": 3}])
    assert cohort_flow(df).set_index("cycle").loc["total", "phq9_complete"] == len(select_cohort(df)) == 2
