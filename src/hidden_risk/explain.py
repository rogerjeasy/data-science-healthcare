"""Step 6: odds ratios (logistic regression) and SHAP values (selected routine-data model)."""
from __future__ import annotations

import re
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm

from .cohort import SCREEN_CUTOFF
from .config import REPORT_TABLES
from .evaluation import TOP
from .models import FEATURE_SETS, MODEL_DIR, NOMINAL, model_frame
from .plotting import BLUE, INK_2, ORANGE, save, use_style

# Units for the odds-ratio model, so each OR reads "per <unit>"
UNITS = {"age": 10, "bmi": 5, "income_poverty_ratio": 1, "sleep_hours": 1, "n_drugs": 1}
ETHNICITY_LEVELS = {1: "Mexican American", 2: "Other Hispanic", 4: "Non-Hispanic Black", 5: "Other / multi-racial"}
MARITAL_LEVELS = {2: "widowed", 3: "divorced", 4: "separated", 5: "never married", 6: "living with partner"}


# --------------------------------------------------------------------------- #
# Odds ratios with 95% CIs (unpenalised logistic regression on the training set)
# --------------------------------------------------------------------------- #
def odds_ratio_design(df: pd.DataFrame) -> pd.DataFrame:
    """Routine features in readable units; missing values -> 0 (or median) plus a missing indicator.

    multimorbidity is left out: it is the sum of the condition flags, so it would be collinear.
    """
    cols = [c for c in FEATURE_SETS["routine"] if c not in NOMINAL + ["multimorbidity"]]
    X = pd.DataFrame(index=df.index)
    indicators = {}
    for c in cols:
        x = df[c] / UNITS.get(c, 1)
        if c == "sex":
            x = (df[c] == 2).astype(float)
        if x.isna().any():
            indicators[c] = x.isna().astype(float)
            x = x.fillna(x.median() if c in UNITS else 0)
        X[c] = x
    for code, label in ETHNICITY_LEVELS.items():  # reference: Non-Hispanic White
        X[f"ethnicity: {label}"] = (df["ethnicity"] == code).astype(float)
    for code, label in MARITAL_LEVELS.items():  # reference: married; under-20s get their own indicator
        X[f"marital: {label}"] = (df["marital_status"] == code).astype(float)
    indicators["marital_status"] = df["marital_status"].isna().astype(float)
    for c, ind in indicators.items():
        if ind.mean() >= 0.01:  # rarer gaps are only imputed: too few cases to estimate an OR
            X[f"missing: {c}"] = ind
    return X


def odds_ratios(df: pd.DataFrame, outcome: str = "si") -> pd.DataFrame:
    X = odds_ratio_design(df)
    X = X.loc[:, X.std() > 0]
    # Questions skipped for the same people (e.g. asked only from age 20) give (near-)identical
    # indicators, which would make the fit singular: keep the first of each such group
    corr = X.corr().abs()
    drop = [c for i, c in enumerate(X.columns) if c.startswith("missing:") and (corr.iloc[:i][c] > 0.98).any()]
    X = X.drop(columns=drop)
    fit = sm.Logit(df[outcome].astype(float), sm.add_constant(X)).fit(disp=0, maxiter=200)
    conf = fit.conf_int()
    table = pd.DataFrame({
        "feature": X.columns,
        "unit": [f"per {UNITS[c]}" if c in UNITS else ("female vs male" if c == "sex" else "") for c in X.columns],
        "odds_ratio": np.exp(fit.params[X.columns]),
        "ci_low": np.exp(conf.loc[X.columns, 0]), "ci_high": np.exp(conf.loc[X.columns, 1]),
        "p_value": fit.pvalues[X.columns],
    })
    return table.sort_values("odds_ratio", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def _original_feature(name: str, columns: List[str]) -> str:
    """Map a transformed column (one-hot level, missing indicator) back to its input feature."""
    name = re.sub(r"^missingindicator_", "", name)
    if name in columns:
        return name
    for c in sorted(columns, key=len, reverse=True):
        if name.startswith(c + "_"):
            return c
    return name


def shap_values(pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame) -> shap.Explanation:
    """SHAP values (log-odds) per input feature: one-hot levels and missing indicators are summed."""
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    Z_test = prep.transform(X_test)
    if clf.__class__.__name__.startswith("HistGradient"):
        explainer = shap.TreeExplainer(clf)
    else:
        background = prep.transform(X_train.sample(1000, random_state=0))
        explainer = shap.LinearExplainer(clf, background)
    sv = explainer(Z_test)
    values = np.asarray(sv.values)
    cols = list(X_test.columns)
    agg = pd.DataFrame(values, columns=names).T.groupby(lambda n: _original_feature(n, cols)).sum().T[cols]
    base = np.ravel(sv.base_values)
    return shap.Explanation(values=agg.to_numpy(), base_values=base,
                            data=X_test.to_numpy(dtype=float), feature_names=cols)


def global_importance(sv: shap.Explanation) -> pd.DataFrame:
    v = sv.values
    x = pd.DataFrame(sv.data, columns=sv.feature_names)
    rows = []
    for i, f in enumerate(sv.feature_names):
        ok = x[f].notna().to_numpy()
        # direction: + = higher value, higher risk (undefined for nominal codes)
        r = np.corrcoef(x.loc[ok, f], v[ok, i])[0, 1] if f not in NOMINAL and x.loc[ok, f].nunique() > 1 else np.nan
        rows.append({"feature": f, "mean_abs_shap": np.abs(v[:, i]).mean(), "direction_corr": r})
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def patient_explanations(sv: shap.Explanation, test: pd.DataFrame, p: np.ndarray, n: int = 3, top: int = 6):
    """The n highest-risk screen-negative (PHQ-9 < 10) adults with ideation who are in the top 5%."""
    thr = np.sort(p)[::-1][int(np.ceil(TOP * len(p))) - 1]
    eligible = np.flatnonzero((test["phq9_score"].to_numpy() < SCREEN_CUTOFF)
                              & (test["si"].to_numpy() == 1) & (p >= thr))
    chosen = eligible[np.argsort(-p[eligible], kind="mergesort")[:n]]
    rows = []
    for rank, i in enumerate(chosen, 1):
        contrib = pd.Series(sv.values[i], index=sv.feature_names)
        for f in contrib.abs().sort_values(ascending=False).index[:top]:
            rows.append({"patient": f"Patient {chr(64 + rank)}", "predicted_risk": p[i],
                         "phq9_score": test["phq9_score"].iloc[i], "feature": f,
                         "value": test[f].iloc[i], "shap_log_odds": contrib[f]})
    return pd.DataFrame(rows), len(eligible)


def plot_importance(imp: pd.DataFrame, title: str, k: int = 15) -> plt.Figure:
    t = imp.head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * k + 1.2))
    ax.barh(t["feature"], t["mean_abs_shap"], color=[ORANGE if f.startswith("rx_") else BLUE for f in t["feature"]],
            height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value| (log-odds); orange = medication class")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_odds_ratios(table: pd.DataFrame, k: int = 20) -> plt.Figure:
    t = table[~table["feature"].str.startswith("missing:")]
    t = t.reindex(t["odds_ratio"].apply(lambda v: abs(np.log(v))).sort_values(ascending=False).index).head(k)
    t = t.sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.errorbar(t["odds_ratio"], y, xerr=[t["odds_ratio"] - t["ci_low"], t["ci_high"] - t["odds_ratio"]],
                fmt="o", color=BLUE, elinewidth=1.5, capsize=0)
    ax.axvline(1, color=INK_2, linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    ax.set_yticks(y, [f"{f} {u}".strip() for f, u in zip(t["feature"], t["unit"])])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Adjusted odds ratio, 95% CI (log scale)")
    ax.set_title("Adjusted odds ratios, routine data (train 2005-2014)")
    fig.tight_layout()
    return fig


def _value_label(feature: str, value: float) -> str:
    if np.isnan(value):
        return "missing"
    if feature == "ethnicity":
        return ETHNICITY_LEVELS.get(int(value), "Non-Hispanic White")
    if feature == "marital_status":
        return MARITAL_LEVELS.get(int(value), "married")
    return f"{value:g}"


def plot_patients(expl: pd.DataFrame) -> plt.Figure:
    patients = expl["patient"].unique()
    fig, axes = plt.subplots(1, len(patients), figsize=(4.2 * len(patients), 3.2), sharex=True)
    for ax, pid in zip(np.atleast_1d(axes), patients):
        t = expl[expl["patient"] == pid].iloc[::-1]
        labels = [f"{f} = {_value_label(f, v)}" for f, v in zip(t["feature"], t["value"].astype(float))]
        ax.barh(labels, t["shap_log_odds"], color=[ORANGE if s > 0 else BLUE for s in t["shap_log_odds"]], height=0.6)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(f"{pid}: risk {t['predicted_risk'].iloc[0]:.0%}, PHQ-9 = {int(t['phq9_score'].iloc[0])}", fontsize=10)
        ax.set_xlabel("SHAP (log-odds)")
    fig.suptitle("Why screen-negative adults with ideation were flagged (top 5%)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    use_style()
    df = model_frame()
    train, test = df[df["split"] == "train"], df[df["split"] == "test"].reset_index(drop=True)
    sel = pd.read_csv(REPORT_TABLES / "step4_selected_models.csv").set_index("feature_set")

    ors = odds_ratios(train)
    ors.round(4).to_csv(REPORT_TABLES / "step6_odds_ratios.csv", index=False)
    save(plot_odds_ratios(ors), "14_odds_ratios")

    out = {}
    for name in ("routine", "routine_phq8"):
        alg = sel.loc[name, "algorithm"]
        # our own models, written by scripts/train_models.py to the git-ignored data/ folder
        pipe = joblib.load(MODEL_DIR / f"{name}__{alg}__none.joblib")
        cols = FEATURE_SETS[name]
        sv = shap_values(pipe, train[cols], test[cols])
        p = pipe.predict_proba(test[cols])[:, 1]
        logit = np.log(p / (1 - p))
        additivity = np.abs(sv.values.sum(axis=1) + sv.base_values - logit).max()
        imp = global_importance(sv)
        imp.round(4).to_csv(REPORT_TABLES / f"step6_shap_importance_{name}.csv", index=False)
        save(plot_importance(imp, f"SHAP feature importance: {name.replace('_', ' + ')} ({alg.upper()})"),
             f"15_shap_importance_{name}")
        if name == "routine":
            fig = plt.figure()
            shap.plots.beeswarm(sv, max_display=15, show=False)
            plt.title("SHAP values, routine-data model (test set)", loc="left")
            save(plt.gcf(), "16_shap_beeswarm_routine")
            expl, n_eligible = patient_explanations(sv, test, p)
            expl.round(4).to_csv(REPORT_TABLES / "step6_patient_explanations.csv", index=False)
            save(plot_patients(expl), "17_patient_explanations")
        out[name] = (imp, additivity)
        plt.close("all")

    print("Adjusted odds ratios (unpenalised LR, train set), strongest first:")
    print(ors[ors.p_value < 0.05].round(3).to_string(index=False))
    for name, (imp, add) in out.items():
        print(f"\nSHAP importance, {name} (max additivity error {add:.2e}):")
        print(imp.head(12).round(4).to_string(index=False))
    print(f"\nScreen-negative cases in the top 5%: {n_eligible}. Explanations:")
    print(expl.round(3).to_string(index=False))
    return 0
