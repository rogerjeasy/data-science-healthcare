"""Step 3: survey-weighted prevalence with design-based CIs, risk ratios, collinearity check."""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .cohort import SCREEN_CUTOFF
from .config import DATA_PROCESSED, REPORT_TABLES
from .plotting import BLUE, INK_2, MUTED, ORANGE, save, use_style

ETHNICITY = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
             4: "Non-Hispanic Black", 5: "Other / multi-racial"}
SEX = {1: "Men", 2: "Women"}
AGE_BINS = [17, 29, 44, 59, 80]
AGE_LABELS = ["18-29", "30-44", "45-59", "60-80"]

# Binary features from step 2 (1 = exposed), with readable labels
BINARY_FACTORS = {
    "mental_health_visit": "Mental-health visit (12 mo)",
    "rx_antidepressant": "Antidepressant",
    "rx_benzo_z": "Benzodiazepine / Z-drug",
    "rx_antipsychotic_mood": "Antipsychotic / mood stabiliser",
    "rx_opioid": "Opioid",
    "sleep_trouble": "Told doctor about trouble sleeping",
    "hospital_stay": "Overnight hospital stay (12 mo)",
    "looking_for_work": "Looking for work",
    "not_working": "Not working",
    "uninsured": "Uninsured",
    "current_smoker": "Current smoker",
    "ever_hard_drugs": "Ever cocaine / heroin / meth",
    "ever_injected_drugs": "Ever injected drugs",
    "ever_marijuana": "Ever marijuana",
    "stroke": "Stroke",
    "lung_disease": "Emphysema / chronic bronchitis",
    "heart_disease": "Heart disease",
    "liver_condition": "Liver condition",
    "arthritis": "Arthritis",
    "asthma": "Asthma",
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "thyroid_problem": "Thyroid problem",
    "cancer": "Cancer",
}


def analysis_frame() -> pd.DataFrame:
    """Step 2 features joined with outcome, PHQ scores and survey design (one row per adult)."""
    persons = pd.read_parquet(DATA_PROCESSED / "nhanes_2005_2018.parquet", columns=[
        "seqn", "si", "si_strict", "phq8_score", "phq9_score", "phq8_positive", "phq9_positive",
        "rx_count", "work_status", "weight_mec", "weight_mec_pooled", "strata", "psu"])
    features = pd.read_parquet(DATA_PROCESSED / "nhanes_2005_2018_features.parquet")
    df = features.merge(persons, on="seqn", how="left", validate="one_to_one")
    df["age_group"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS)
    return df


# --------------------------------------------------------------------------- #
# Design-based prevalence (Taylor linearisation, strata x PSU)
# --------------------------------------------------------------------------- #
def survey_proportion(y: pd.Series, weight: pd.Series, strata: pd.Series, psu: pd.Series,
                      domain: Optional[pd.Series] = None, level: float = 0.95) -> dict:
    """Weighted proportion of binary y in `domain` with a design-based CI.

    The variance uses every PSU, including those without domain members, as a domain
    estimate must. The CI is computed on the logit scale with df = #PSU - #strata.
    """
    d = pd.Series(True, index=y.index) if domain is None else domain.fillna(False).astype(bool)
    y = y.astype(float).fillna(0).to_numpy()
    w = (weight.astype(float).to_numpy() * d.to_numpy())
    total = w.sum()
    p = (w * y).sum() / total
    z = w * (y - p) / total
    cluster = pd.DataFrame({"h": strata.to_numpy(), "j": psu.to_numpy(), "z": z})
    per_psu = cluster.groupby(["h", "j"])["z"].sum().reset_index()
    var = 0.0
    for _, g in per_psu.groupby("h"):
        n_h = len(g)
        if n_h > 1:
            var += n_h / (n_h - 1) * ((g["z"] - g["z"].mean()) ** 2).sum()
    se = np.sqrt(var)
    dof = len(per_psu) - per_psu["h"].nunique()
    t = stats.t.ppf(0.5 + level / 2, dof)
    logit, se_logit = np.log(p / (1 - p)), se / (p * (1 - p))
    lo, hi = (1 / (1 + np.exp(-(logit + s * t * se_logit))) for s in (-1, 1))
    return {"n": int(d.sum()), "cases": int((y * d.to_numpy()).sum()),
            "pct": 100 * p, "se_pct": 100 * se, "ci_low": 100 * lo, "ci_high": 100 * hi}


def weighted_prevalence_table(df: pd.DataFrame, trend: Optional[pd.DataFrame] = None,
                              outcome: str = "si") -> pd.DataFrame:
    """Prevalence by cycle (2-year weights) and by subgroup (pooled weights), with 95% CIs."""
    rows = []

    def add(group: str, level: str, data: pd.DataFrame, weight: str, domain=None):
        rows.append({"group": group, "level": level,
                     **survey_proportion(data[outcome], data[weight], data["strata"], data["psu"], domain)})

    add("Overall", "2005-2018", df, "weight_mec_pooled")
    for cycle, g in df.groupby("cycle"):
        add("Cycle", cycle, g, "weight_mec")
    if trend is not None:
        add("Cycle", "2021-2023 (trend check)", trend, "weight_mec")
    groups = {
        "Sex": df["sex"].map(SEX),
        "Age group": df["age_group"].astype("string"),
        "Ethnicity": df["ethnicity"].map(ETHNICITY),
        "Depression screen (PHQ-9)": df["phq9_score"].ge(SCREEN_CUTOFF).map({True: "positive (>=10)", False: "negative (<10)"}),
        "Depression screen (PHQ-8)": df["phq8_score"].ge(SCREEN_CUTOFF).map({True: "positive (>=10)", False: "negative (<10)"}),
    }
    for name, labels in groups.items():
        for level in labels.dropna().unique():
            add(name, str(level), df, "weight_mec_pooled", labels.eq(level))
    out = pd.DataFrame(rows)
    order = {"Overall": 0, "Cycle": 1, "Sex": 2, "Age group": 3, "Ethnicity": 4}
    return (out.assign(_o=out["group"].map(order).fillna(9))
            .sort_values(["_o", "group", "level"]).drop(columns="_o").reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Crude risk ratios
# --------------------------------------------------------------------------- #
def risk_ratio(exposed: pd.Series, outcome: pd.Series) -> dict:
    """Crude (unweighted) risk ratio with a 95% CI (log / Katz method)."""
    ok = exposed.notna() & outcome.notna()
    e, y = exposed[ok].astype(bool).to_numpy(), outcome[ok].astype(int).to_numpy()
    a, n1 = y[e].sum(), e.sum()
    c, n0 = y[~e].sum(), (~e).sum()
    p1, p0 = a / n1, c / n0
    rr = p1 / p0
    se = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    return {"n_exposed": int(n1), "cases_exposed": int(a), "risk_exposed_pct": 100 * p1,
            "risk_unexposed_pct": 100 * p0, "rr": rr,
            "ci_low": np.exp(np.log(rr) - 1.96 * se), "ci_high": np.exp(np.log(rr) + 1.96 * se)}


def risk_ratio_table(df: pd.DataFrame, outcome: str = "si",
                     restrict: Optional[pd.Series] = None) -> pd.DataFrame:
    data = df if restrict is None else df[restrict.fillna(False).astype(bool)]
    extra = {
        "Fair / poor self-rated health": data["self_rated_health"].ge(4).astype("boolean").mask(data["self_rated_health"].isna()),
        "Below poverty line": data["income_poverty_ratio"].lt(1).astype("boolean").mask(data["income_poverty_ratio"].isna()),
        "5+ distinct prescription drugs": data["n_drugs"].ge(5).astype("boolean").mask(data["n_drugs"].isna()),
        "3+ chronic conditions": data["multimorbidity"].ge(3).astype("boolean").mask(data["multimorbidity"].isna()),
        "Never married": data["marital_status"].eq(5).astype("boolean").mask(data["marital_status"].isna()),
        "Divorced / separated": data["marital_status"].isin([3, 4]).astype("boolean").mask(data["marital_status"].isna()),
    }
    rows = [{"variable": col, "factor": label, **risk_ratio(data[col].astype("boolean"), data[outcome])}
            for col, label in BINARY_FACTORS.items()]
    rows += [{"variable": label, "factor": label, **risk_ratio(s, data[outcome])} for label, s in extra.items()]
    return pd.DataFrame(rows).sort_values("rr", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Collinearity among model features
# --------------------------------------------------------------------------- #
def correlation_pairs(df: pd.DataFrame, columns, threshold: float = 0.5) -> pd.DataFrame:
    """Spearman correlation (pairwise complete) for every feature pair, strongest first."""
    corr = df[columns].astype(float).corr(method="spearman")
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("spearman_r").reset_index()
    pairs.columns = ["feature_1", "feature_2", "spearman_r"]
    pairs["abs_r"] = pairs["spearman_r"].abs()
    pairs["collinear"] = pairs["abs_r"] > threshold
    return pairs.sort_values("abs_r", ascending=False).reset_index(drop=True), corr


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_prevalence(table: pd.DataFrame) -> plt.Figure:
    t = table[table["group"].isin(["Sex", "Age group", "Ethnicity", "Depression screen (PHQ-9)"])].iloc[::-1]
    overall = table.loc[table["group"] == "Overall", "pct"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.errorbar(t["pct"], y, xerr=[t["pct"] - t["ci_low"], t["ci_high"] - t["pct"]],
                fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.5, capsize=0)
    ax.axvline(overall, color=INK_2, linewidth=1)
    ax.set_yticks(y, [f"{g.split(' (')[0]}: {l}" for g, l in zip(t["group"], t["level"])])
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20], ["1%", "2%", "5%", "10%", "20%"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Weighted prevalence of suicidal ideation, 95% CI (log scale)")
    ax.set_title(f"Weighted prevalence by subgroup (line = overall {overall:.1f}%)")
    fig.tight_layout()
    return fig


def plot_risk_ratios(rr: pd.DataFrame) -> plt.Figure:
    t = rr.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(t) + 1.2))
    y = np.arange(len(t))
    med = t["variable"].str.startswith("rx_")
    for mask, color, label in [(~med, BLUE, "other routine factor"), (med, ORANGE, "medication class (step 2)")]:
        ax.errorbar(t.loc[mask, "rr"], y[mask.to_numpy()],
                    xerr=[t.loc[mask, "rr"] - t.loc[mask, "ci_low"], t.loc[mask, "ci_high"] - t.loc[mask, "rr"]],
                    fmt="o", color=color, elinewidth=1.5, capsize=0, label=label)
    ax.axvline(1, color=INK_2, linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    ax.set_yticks(y, t["factor"])
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_xlabel("Crude risk ratio, 95% CI (log scale)")
    ax.set_title("Crude risk ratios for suicidal ideation")
    fig.tight_layout()
    return fig


def plot_correlation(corr: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Spearman r")
    cbar.outline.set_visible(False)
    ax.set_title("Correlation between routine-data model features")
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    from .models import FEATURE_SETS

    use_style()
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    df = analysis_frame()
    trend = pd.read_parquet(DATA_PROCESSED / "nhanes_2021_2023_phq9.parquet")

    prev = weighted_prevalence_table(df, trend)
    prev.round(2).to_csv(REPORT_TABLES / "step3_weighted_prevalence.csv", index=False)
    prev_strict = weighted_prevalence_table(df, trend, outcome="si_strict")
    prev_strict.round(2).to_csv(REPORT_TABLES / "step3_weighted_prevalence_si_strict.csv", index=False)

    rr = risk_ratio_table(df)
    rr.round(3).to_csv(REPORT_TABLES / "step3_risk_ratios.csv", index=False)
    rr_neg = risk_ratio_table(df, restrict=df["phq9_score"] < SCREEN_CUTOFF)
    rr_neg.round(3).to_csv(REPORT_TABLES / "step3_risk_ratios_screen_negative.csv", index=False)

    numeric = [c for c in FEATURE_SETS["routine"] if c not in ("ethnicity", "marital_status")]
    pairs, corr = correlation_pairs(df, numeric)
    pairs.round(3).to_csv(REPORT_TABLES / "step3_feature_correlations.csv", index=False)

    for name, fig in [("06_weighted_prevalence_ci", plot_prevalence(prev)),
                      ("07_risk_ratios_step2_features", plot_risk_ratios(rr)),
                      ("08_feature_correlation", plot_correlation(corr))]:
        save(fig, name)
        plt.close(fig)

    show = prev[["group", "level", "n", "cases", "pct", "ci_low", "ci_high"]].round(2)
    print(show.to_string(index=False))
    print("\nTop crude risk ratios:")
    print(rr[["factor", "rr", "ci_low", "ci_high"]].head(10).round(2).to_string(index=False))
    print("\nFeature pairs with |Spearman r| > 0.5:")
    print(pairs[pairs["collinear"]][["feature_1", "feature_2", "spearman_r"]].round(2).to_string(index=False))
    av = pairs[(pairs[["feature_1", "feature_2"]].isin(["rx_antidepressant", "mental_health_visit"])).all(axis=1)]
    print(f"\nrx_antidepressant vs mental_health_visit: r = {av['spearman_r'].iloc[0]:.2f}")
    return 0
