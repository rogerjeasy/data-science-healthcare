import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from hidden_risk.eda import risk_ratio, survey_proportion
from hidden_risk.evaluation import average_precision, calibration, roc_auc, top_fraction
from hidden_risk.models import FEATURE_SETS, LEAKY, MAIN_SETS, check_no_leakage, model_frame


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.04).astype(int)
    p = np.round(rng.random(5000) * 0.5 + 0.3 * y, 2)  # rounded, so there are ties
    return y, p


def test_fast_metrics_match_scikit_learn(scores):
    y, p = scores
    assert average_precision(y, p) == pytest.approx(average_precision_score(y, p), abs=1e-12)
    assert roc_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_top_five_percent_flags_exactly_one_in_twenty(scores):
    y, p = scores
    recall, precision = top_fraction(y, p)
    k = int(np.ceil(0.05 * len(p)))
    assert precision * k == pytest.approx(recall * y.sum())


def test_calibrated_predictions_have_slope_one_and_no_offset():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.3, 200_000)
    y = (rng.random(len(p)) < p).astype(int)
    citl, slope = calibration(y, p)
    assert abs(citl) < 0.03 and abs(slope - 1) < 0.05


def test_survey_proportion_matches_weighted_mean_and_covers_it():
    rng = np.random.default_rng(2)
    n = 4000
    df = pd.DataFrame({"y": rng.random(n) < 0.05, "w": rng.uniform(1, 5, n),
                       "h": rng.integers(0, 10, n), "j": rng.integers(1, 3, n)})
    r = survey_proportion(df.y, df.w, df.h, df.j)
    assert r["pct"] == pytest.approx(100 * np.average(df.y, weights=df.w))
    assert r["ci_low"] < r["pct"] < r["ci_high"]
    sub = survey_proportion(df.y, df.w, df.h, df.j, domain=df.h < 5)
    assert sub["n"] == (df.h < 5).sum()


def test_risk_ratio():
    exposed = pd.Series([1] * 100 + [0] * 100).astype("boolean")
    outcome = pd.Series([1] * 20 + [0] * 80 + [1] * 10 + [0] * 90)
    assert risk_ratio(exposed, outcome)["rr"] == pytest.approx(2.0)


def test_main_feature_sets_never_contain_the_outcome():
    for name in MAIN_SETS:
        assert not set(FEATURE_SETS[name]) & LEAKY
        assert not any(c.startswith("phq_") for c in FEATURE_SETS[name])
    with pytest.raises(ValueError):
        check_no_leakage(FEATURE_SETS["routine"] + ["phq9_score"], "routine")


def test_feature_sets_are_nested():
    assert set(FEATURE_SETS["demographics"]) < set(FEATURE_SETS["routine"]) < set(FEATURE_SETS["routine_phq8"])


def test_temporal_split_matches_the_project_description():
    df = model_frame()
    counts = df.groupby("split")["si"].agg(["size", "sum"])
    assert counts.loc["train"].tolist() == [26_057, 983]
    assert counts.loc["test"].tolist() == [10_202, 383]
    assert set(FEATURE_SETS["routine_phq8"]) <= set(df.columns)
