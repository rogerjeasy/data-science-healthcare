"""Step 4: nested feature sets, two algorithms, three imbalance strategies, tuned by CV on 2005-2014."""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DATA_PROCESSED, REPORT_TABLES

SEED = 42
MODEL_DIR = DATA_PROCESSED / "models"
PREDICTIONS = DATA_PROCESSED / "predictions_test.parquet"

DEMOGRAPHICS = ["age", "sex", "ethnicity"]
ROUTINE = DEMOGRAPHICS + [
    # social
    "education", "marital_status", "income_poverty_ratio", "not_working", "looking_for_work", "uninsured",
    # healthcare use
    "self_rated_health", "healthcare_visits", "hospital_stay", "mental_health_visit",
    # medication (step 2)
    "rx_antidepressant", "rx_opioid", "rx_benzo_z", "rx_antipsychotic_mood", "n_drugs",
    # chronic conditions (step 2)
    "arthritis", "heart_disease", "stroke", "lung_disease", "cancer", "diabetes", "hypertension",
    "asthma", "liver_condition", "thyroid_problem", "multimorbidity", "high_cholesterol",
    # sleep, body, substance use
    "sleep_trouble", "sleep_hours", "bmi", "smoked_100_cigarettes", "current_smoker",
    "ever_marijuana", "ever_hard_drugs", "ever_injected_drugs",
]
FEATURE_SETS: Dict[str, List[str]] = {
    "demographics": DEMOGRAPHICS,
    "routine": ROUTINE,
    "phq8": ["phq8_score"],
    "routine_phq8": ROUTINE + ["phq8_score"],
    # Reference only: phq9_score contains item 9, the outcome (docs/project_description.md, section 8)
    "phq9": ["phq9_score"],
    "routine_phq9": ROUTINE + ["phq9_score"],
}
MAIN_SETS = ["demographics", "routine", "phq8", "routine_phq8"]
REFERENCE_SETS = ["phq9", "routine_phq9"]
NOMINAL = ["ethnicity", "marital_status"]
OUTCOME_COLUMNS = {"si", "si_strict", "phq_9", "phq9_positive"}
LEAKY = OUTCOME_COLUMNS | {"phq9_score"}

ALGORITHMS = ["lr", "hgb"]
STRATEGIES = ["none", "balanced", "smote"]

LR_GRID = {
    "clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    "clf__l1_ratio": [0.0, 0.5, 1.0],
}
HGB_SPACE = {
    "clf__learning_rate": loguniform(0.01, 0.2),
    "clf__max_iter": [100, 200, 300, 500],
    "clf__max_leaf_nodes": [3, 7, 15, 31],
    "clf__min_samples_leaf": [20, 50, 100, 200, 400],
    "clf__l2_regularization": loguniform(1e-3, 10),
    "clf__max_features": [0.5, 0.8, 1.0],
}
HGB_ITER = 40
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "brier": "neg_brier_score"}


def model_frame() -> pd.DataFrame:
    """Step 2 features + outcome, PHQ scores and evaluation groups; nullable ints become float."""
    persons = pd.read_parquet(DATA_PROCESSED / "nhanes_2005_2018.parquet",
                              columns=["seqn", "si", "si_strict", "phq8_score", "phq9_score"])
    features = pd.read_parquet(DATA_PROCESSED / "nhanes_2005_2018_features.parquet")
    df = features.merge(persons, on="seqn", how="left", validate="one_to_one")
    numeric = [c for c in df.columns if c not in ("seqn", "cycle", "split")]
    df[numeric] = df[numeric].astype("float64")
    return df


def check_no_leakage(columns: List[str], name: str) -> None:
    bad = set(columns) & (OUTCOME_COLUMNS if name in REFERENCE_SETS else LEAKY)
    if bad:
        raise ValueError(f"feature set {name!r} contains outcome columns: {sorted(bad)}")


def _preprocessor(columns: List[str], impute: bool, scale: bool) -> ColumnTransformer:
    nominal = [c for c in columns if c in NOMINAL]
    numeric = [c for c in columns if c not in NOMINAL]
    num_steps = []
    if impute:  # median + one missing-indicator per column with missing values in training
        num_steps.append(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    if scale:
        num_steps.append(StandardScaler())
    num = make_pipeline(*num_steps) if num_steps else "passthrough"
    nom = make_pipeline(SimpleImputer(strategy="constant", fill_value=-1),  # missing = own category
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    parts = [("num", num, numeric)] + ([("nom", nom, nominal)] if nominal else [])
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def build_pipeline(columns: List[str], algorithm: str, strategy: str) -> Pipeline:
    """Preprocessing -> optional SMOTE (inside CV folds only) -> classifier."""
    weight = "balanced" if strategy == "balanced" else None
    if algorithm == "lr":
        prep = _preprocessor(columns, impute=True, scale=True)
        clf = LogisticRegression(solver="saga", max_iter=5000, tol=1e-3, class_weight=weight, random_state=SEED)
    elif algorithm == "hgb":
        # NaN handled natively, except before SMOTE, which needs complete data
        prep = _preprocessor(columns, impute=strategy == "smote", scale=False)
        clf = HistGradientBoostingClassifier(early_stopping=False, class_weight=weight, random_state=SEED)
    else:
        raise ValueError(algorithm)
    steps = [("prep", prep)]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def _refit_pr_auc_then_brier(cv_results) -> int:
    """Best CV PR-AUC; exact ties (e.g. every C for a single feature ranks patients alike) go to the best Brier."""
    pr = np.asarray(cv_results["mean_test_pr_auc"])
    tied = np.flatnonzero(pr >= np.nanmax(pr) - 1e-9)
    return int(tied[np.argmax(np.asarray(cv_results["mean_test_brier"])[tied])])


def tune(X: pd.DataFrame, y: pd.Series, algorithm: str, strategy: str, n_jobs: int = -1):
    """Stratified 5-fold CV on the training data, refit on PR-AUC (ties: Brier)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe = build_pipeline(list(X.columns), algorithm, strategy)
    if algorithm == "lr":
        grid = LR_GRID if X.shape[1] > 1 else {"clf__C": LR_GRID["clf__C"], "clf__l1_ratio": [0.0]}
        search = GridSearchCV(pipe, grid, scoring=SCORING, refit=_refit_pr_auc_then_brier, cv=cv, n_jobs=n_jobs)
    else:
        search = RandomizedSearchCV(pipe, HGB_SPACE, n_iter=HGB_ITER, scoring=SCORING, refit=_refit_pr_auc_then_brier,
                                    cv=cv, n_jobs=n_jobs, random_state=SEED)
    search.fit(X, y)
    return search


def _cv_row(search, **keys) -> dict:
    r, i = search.cv_results_, search.best_index_
    params = {k.replace("clf__", ""): (float(v) if isinstance(v, (float, np.floating)) else v)
              for k, v in search.best_params_.items()}
    return {**keys,
            "cv_pr_auc": r["mean_test_pr_auc"][i], "cv_pr_auc_sd": r["std_test_pr_auc"][i],
            "cv_roc_auc": r["mean_test_roc_auc"][i], "cv_roc_auc_sd": r["std_test_roc_auc"][i],
            "cv_brier": -r["mean_test_brier"][i], "best_params": json.dumps(params, default=str)}


def run_configs(df: pd.DataFrame, configs, outcome: str = "si", train_mask=None) -> tuple:
    """Tune and refit every (feature_set, algorithm, strategy); predict the test set."""
    train_mask = (df["split"] == "train") if train_mask is None else train_mask
    train, test = df[train_mask], df[df["split"] == "test"]
    rows, preds, fitted = [], {}, {}
    for name, algorithm, strategy in configs:
        cols = FEATURE_SETS[name]
        check_no_leakage(cols, name)
        start = time.time()
        search = tune(train[cols], train[outcome].astype(int), algorithm, strategy)
        key = f"{name}|{algorithm}|{strategy}"
        rows.append(_cv_row(search, feature_set=name, algorithm=algorithm, strategy=strategy, outcome=outcome))
        preds[key] = search.best_estimator_.predict_proba(test[cols])[:, 1]
        fitted[key] = search.best_estimator_
        print(f"  {key:<32} CV PR-AUC {rows[-1]['cv_pr_auc']:.3f}  ROC-AUC {rows[-1]['cv_roc_auc']:.3f}"
              f"  ({time.time() - start:.0f}s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(preds, index=test.index), fitted


def select_models(cv: pd.DataFrame) -> pd.DataFrame:
    """Per feature set: the algorithm with the higher CV PR-AUC without imbalance correction.

    Chosen on training-set CV only, so the test set stays untouched by model selection.
    """
    base = cv[(cv["strategy"] == "none") & (cv["outcome"] == "si")]
    best = base.loc[base.groupby("feature_set")["cv_pr_auc"].idxmax()]
    return best[["feature_set", "algorithm", "strategy", "cv_pr_auc", "cv_roc_auc"]].reset_index(drop=True)


def main(argv=None) -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    df = model_frame()
    test = df[df["split"] == "test"]

    configs = [(s, a, st) for s in MAIN_SETS for a in ALGORITHMS for st in STRATEGIES]
    configs += [(s, a, "none") for s in REFERENCE_SETS for a in ALGORITHMS]
    print(f"Main outcome si: {len(configs)} configurations")
    cv, preds, fitted = run_configs(df, configs)
    selected = select_models(cv)

    # Sensitivity: stricter outcome, same model choice
    strict = [(r.feature_set, r.algorithm, "none") for r in selected.itertuples() if r.feature_set in ("routine", "routine_phq8", "phq8")]
    print(f"Sensitivity outcome si_strict: {len(strict)} configurations")
    cv_s, preds_s, _ = run_configs(df, strict, outcome="si_strict")
    preds_s.columns = [f"si_strict:{c}" for c in preds_s.columns]

    # Screen-negative model: routine data, trained only on adults with PHQ-9 < 10
    routine_alg = selected.set_index("feature_set").loc["routine", "algorithm"]
    print("Screen-negative model (trained on PHQ-9 < 10 only)")
    neg_train = (df["split"] == "train") & (df["phq9_score"] < 10)
    cv_n, preds_n, fitted_n = run_configs(df, [("routine", routine_alg, "none")], train_mask=neg_train)
    cv_n["feature_set"] = "routine_trained_screen_negative"
    preds_n.columns = [f"screen_negative:{c}" for c in preds_n.columns]

    cv_all = pd.concat([cv, cv_s, cv_n], ignore_index=True)
    cv_all.round(4).to_csv(REPORT_TABLES / "step4_cv_results.csv", index=False)
    selected.round(4).to_csv(REPORT_TABLES / "step4_selected_models.csv", index=False)

    out = pd.concat([test[["seqn", "cycle", "si", "si_strict", "phq8_score", "phq9_score",
                           "age", "sex", "ethnicity"]], preds, preds_s, preds_n], axis=1)
    out.to_parquet(PREDICTIONS, index=False)
    for key, est in {**fitted, **{f"screen_negative:{k}": v for k, v in fitted_n.items()}}.items():
        if key.endswith("|none") or key.startswith("routine|"):
            joblib.dump(est, MODEL_DIR / f"{key.replace('|', '__').replace(':', '__')}.joblib")
    print("\nSelected models (CV on 2005-2014):")
    print(selected.round(3).to_string(index=False))
    print(f"Wrote {PREDICTIONS} and models to {MODEL_DIR}")
    return 0
