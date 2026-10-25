"""Step 5: test-set metrics, paired bootstrap tests of H1-H4, screen-negative analysis, fairness."""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import brentq
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from .cohort import SCREEN_CUTOFF
from .config import REPORT_TABLES
from .eda import AGE_BINS, AGE_LABELS, ETHNICITY, SEX
from .models import MAIN_SETS, PREDICTIONS, REFERENCE_SETS
from .plotting import BLUE, INK_2, MUTED, ORANGE, save, use_style

B = 2000
SEED = 2026
TOP = 0.05  # clinical operating point: flag the top 5% (1 in 20) of patients
METRICS = ["pr_auc", "roc_auc", "brier", "recall_top5", "precision_top5"]


# --------------------------------------------------------------------------- #
# Metrics (numpy, so the bootstrap is fast; matched to scikit-learn in tests)
# --------------------------------------------------------------------------- #
def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Same definition as sklearn.metrics.average_precision_score."""
    order = np.argsort(-p, kind="mergesort")
    y, p = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(p)), len(y) - 1]
    tps = np.cumsum(y)[last]
    precision = tps / (last + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with tied scores counted as 1/2."""
    n1 = y.sum()
    n0 = len(y) - n1
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def top_fraction(y: np.ndarray, p: np.ndarray, frac: float = TOP) -> tuple:
    """Recall and precision when exactly ceil(frac * n) highest-risk patients are flagged."""
    k = int(np.ceil(frac * len(p)))
    flagged = np.argsort(-p, kind="mergesort")[:k]
    tp = y[flagged].sum()
    return tp / y.sum(), tp / k


def calibration(y: np.ndarray, p: np.ndarray) -> tuple:
    """Calibration-in-the-large (intercept with logit(p) as offset) and calibration slope."""
    lp = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1))
    citl = brentq(lambda a: np.sum(y - 1 / (1 + np.exp(-(lp + a)))), -10, 10)
    # lbfgs instead of plain Newton: near-0 / near-1 predictions (e.g. PHQ-9 = 0) make Newton diverge
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
    return citl, float(fit.coef_[0, 0])


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    recall, precision = top_fraction(y, p)
    return {"pr_auc": average_precision(y, p), "roc_auc": roc_auc(y, p),
            "brier": float(np.mean((p - y) ** 2)), "recall_top5": recall, "precision_top5": precision}


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #
def _boot_chunk(y, P: Dict[str, np.ndarray], seeds) -> list:
    rows = []
    for s in seeds:
        idx = np.random.default_rng(s).integers(0, len(y), len(y))
        yb = y[idx]
        row = {"prevalence": yb.mean()}
        for name, p in P.items():
            for m, v in metrics(yb, p[idx]).items():
                row[f"{name}::{m}"] = v
        rows.append(row)
    return rows


def bootstrap(y: np.ndarray, P: Dict[str, np.ndarray], n: int = B, seed: int = SEED,
              n_jobs: int = -1) -> pd.DataFrame:
    """Resample test patients with replacement; every model is scored on the same resample."""
    seeds = np.random.SeedSequence(seed).generate_state(n)
    chunks = np.array_split(seeds, 20)
    out = Parallel(n_jobs=n_jobs)(delayed(_boot_chunk)(y, P, c) for c in chunks)
    return pd.DataFrame([r for chunk in out for r in chunk])


def ci(values: pd.Series) -> tuple:
    return float(values.quantile(0.025)), float(values.quantile(0.975))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def test_metrics_table(y, preds: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in preds.columns:
        p = preds[key].to_numpy()
        point = metrics(y, p)
        citl, slope = calibration(y, p)
        name, alg, strat = key.split("|")
        row = {"feature_set": name, "algorithm": alg, "strategy": strat}
        for m in METRICS:
            lo, hi = ci(boot[f"{key}::{m}"])
            row.update({m: point[m], f"{m}_ci_low": lo, f"{m}_ci_high": hi})
        row.update({"mean_predicted": p.mean(), "observed": y.mean(),
                    "calibration_in_the_large": citl, "calibration_slope": slope})
        rows.append(row)
    return pd.DataFrame(rows)


def _diff_row(hypothesis, comparison, metric, boot, a, b, criterion, supported_fn, point):
    d = boot[f"{a}::{metric}"] - boot[f"{b}::{metric}"]
    lo, hi = ci(d)
    return {"hypothesis": hypothesis, "comparison": comparison, "metric": metric,
            "estimate": point, "ci_low": lo, "ci_high": hi, "criterion": criterion,
            "supported": bool(supported_fn(point, lo, hi))}


def hypotheses_table(y, preds, boot, selected: Dict[str, str], cv: pd.DataFrame,
                     groups: Dict[str, pd.Series]) -> pd.DataFrame:
    P = {k: preds[k].to_numpy() for k in preds.columns}
    pt = {k: metrics(y, p) for k, p in P.items()}
    routine, demo = selected["routine"], selected["demographics"]
    rows = []

    prev = y.mean()
    d = boot[f"{routine}::pr_auc"] - 2 * boot["prevalence"]
    lo, hi = ci(d)
    rows.append({"hypothesis": "H1", "comparison": f"routine ({routine.split('|')[1]}) PR-AUC - 2 x prevalence",
                 "metric": "pr_auc", "estimate": pt[routine]["pr_auc"] - 2 * prev, "ci_low": lo, "ci_high": hi,
                 "criterion": "CI lower bound > 0", "supported": lo > 0})
    rows.append(_diff_row("H1", "routine - demographics", "pr_auc", boot, routine, demo,
                          "CI of difference > 0", lambda e, l, h: l > 0,
                          pt[routine]["pr_auc"] - pt[demo]["pr_auc"]))
    for full, base, label in [(selected["routine_phq8"], selected["phq8"], "routine + PHQ-8 - PHQ-8 (leakage-free)"),
                              (selected["routine_phq9"], selected["phq9"], "routine + PHQ-9 - PHQ-9 (as registered; PHQ-9 contains the outcome item)")]:
        rows.append(_diff_row("H2", label, "pr_auc", boot, full, base, "CI of difference > 0",
                              lambda e, l, h: l > 0, pt[full]["pr_auc"] - pt[base]["pr_auc"]))
    for alg in ("lr", "hgb"):
        none = f"routine|{alg}|none"
        for strat in ("balanced", "smote"):
            other = f"routine|{alg}|{strat}"
            rows.append(_diff_row("H3", f"routine {alg}: {strat} - none", "pr_auc", boot, other, none,
                                  "no improvement: CI lower bound <= 0", lambda e, l, h: l <= 0,
                                  pt[other]["pr_auc"] - pt[none]["pr_auc"]))
            rows.append(_diff_row("H3", f"routine {alg}: {strat} - none", "brier", boot, other, none,
                                  "worse calibration: CI of Brier difference > 0", lambda e, l, h: l > 0,
                                  pt[other]["brier"] - pt[none]["brier"]))
    name, alg, strat = routine.split("|")
    cv_auc = cv[(cv.feature_set == name) & (cv.algorithm == alg) & (cv.strategy == strat) & (cv.outcome == "si")]["cv_roc_auc"].iloc[0]
    lo, hi = ci(boot[f"{routine}::roc_auc"] - cv_auc)
    est = pt[routine]["roc_auc"] - cv_auc
    rows.append({"hypothesis": "H4", "comparison": f"routine test ROC-AUC - CV ROC-AUC ({cv_auc:.3f})",
                 "metric": "roc_auc", "estimate": est, "ci_low": lo, "ci_high": hi,
                 "criterion": "|difference| < 0.05", "supported": abs(est) < 0.05})
    return pd.DataFrame(rows)


def _group_metrics(y, p, threshold, mask) -> dict:
    yg, pg = y[mask], p[mask]
    return {"roc_auc": roc_auc(yg, pg), "pr_auc": average_precision(yg, pg),
            "pr_auc_over_prevalence": average_precision(yg, pg) / yg.mean(),
            "recall_at_global_top5": yg[pg >= threshold].sum() / yg.sum()}


def _fair_chunk(y, p, masks, seeds):
    rows = []
    for s in seeds:
        idx = np.random.default_rng(s).integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        thr = np.sort(pb)[::-1][int(np.ceil(TOP * len(pb))) - 1]
        rows.append({(a, g, m): v for (a, g), mask in masks.items()
                     for m, v in _group_metrics(yb, pb, thr, mask[idx]).items()})
    return rows


def fairness_table(y, p, groups: Dict[str, pd.Series], n: int = B, seed: int = SEED):
    """Per subgroup metrics; the flagging threshold is the top 5% of the whole test set."""
    masks = {(attr, str(level)): (labels == level).to_numpy()
             for attr, labels in groups.items() for level in labels.dropna().unique()}
    thr = np.sort(p)[::-1][int(np.ceil(TOP * len(p))) - 1]
    seeds = np.random.SeedSequence(seed + 1).generate_state(n)
    out = Parallel(n_jobs=-1)(delayed(_fair_chunk)(y, p, masks, c) for c in np.array_split(seeds, 20))
    boot = pd.DataFrame([r for chunk in out for r in chunk])
    rows = []
    for (attr, level), mask in masks.items():
        row = {"attribute": attr, "group": level, "n": int(mask.sum()), "cases": int(y[mask].sum()),
               "prevalence": y[mask].mean()}
        for m, v in _group_metrics(y, p, thr, mask).items():
            lo, hi = ci(boot[(attr, level, m)])
            row.update({m: v, f"{m}_ci_low": lo, f"{m}_ci_high": hi})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["attribute", "group"]).reset_index(drop=True)
    gaps = []
    for attr in groups:
        levels = table[table.attribute == attr]
        hi_g = levels.loc[levels.roc_auc.idxmax(), "group"]
        lo_g = levels.loc[levels.roc_auc.idxmin(), "group"]
        d = boot[(attr, hi_g, "roc_auc")] - boot[(attr, lo_g, "roc_auc")]
        est = levels.roc_auc.max() - levels.roc_auc.min()
        gaps.append({"hypothesis": "H4", "comparison": f"ROC-AUC gap by {attr}: {hi_g} - {lo_g}",
                     "metric": "roc_auc", "estimate": est, "ci_low": ci(d)[0], "ci_high": ci(d)[1],
                     "criterion": "gap < 0.05", "supported": est < 0.05})
    return table, pd.DataFrame(gaps)


def screen_negative_table(df: pd.DataFrame, models: Dict[str, str], screen: str = "phq9_score",
                          n: int = B, seed: int = SEED) -> tuple:
    """Performance among adults the depression screen misses, and what flagging adds."""
    neg = (df[screen] < SCREEN_CUTOFF).to_numpy()
    y = df["si"].to_numpy().astype(int)
    yn = y[neg]
    P = {label: df[col].to_numpy()[neg] for label, col in models.items()}
    boot = bootstrap(yn, P, n=n, seed=seed + 2)
    rows = []
    for label, p in P.items():
        point = metrics(yn, p)
        row = {"screen": screen.replace("_score", "").upper() + " < 10", "model": label,
               "n": int(neg.sum()), "cases": int(yn.sum()), "prevalence": yn.mean()}
        for m in ["pr_auc", "roc_auc", "recall_top5", "precision_top5"]:
            lo, hi = ci(boot[f"{label}::{m}"])
            row.update({m: point[m], f"{m}_ci_low": lo, f"{m}_ci_high": hi})
        row["pr_auc_over_prevalence"] = point["pr_auc"] / yn.mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    diffs = []
    if {"routine + PHQ-8", "PHQ-8"} <= set(P):
        d = boot["routine + PHQ-8::pr_auc"] - boot["PHQ-8::pr_auc"]
        est = average_precision(yn, P["routine + PHQ-8"]) - average_precision(yn, P["PHQ-8"])
        diffs.append({"comparison": f"routine + PHQ-8 - PHQ-8 among {screen.split('_')[0].upper()} < 10",
                      "metric": "pr_auc", "estimate": est, "ci_low": ci(d)[0], "ci_high": ci(d)[1]})
    return table, pd.DataFrame(diffs)


def capture_table(df: pd.DataFrame, col: str, screen: str = "phq9_score") -> pd.DataFrame:
    """Screen first, then flag the top 5% of screen-negative adults by the model."""
    y = df["si"].to_numpy().astype(int)
    pos = (df[screen] >= SCREEN_CUTOFF).to_numpy()
    neg_idx = np.flatnonzero(~pos)
    p = df[col].to_numpy()[neg_idx]
    k = int(np.ceil(TOP * len(neg_idx)))
    flagged = neg_idx[np.argsort(-p, kind="mergesort")[:k]]
    extra = int(y[flagged].sum())
    return pd.DataFrame([{
        "screen": screen.replace("_score", "").upper() + " >= 10", "model": col,
        "test_adults": len(y), "test_cases": int(y.sum()),
        "screen_positive": int(pos.sum()), "cases_caught_by_screen": int(y[pos].sum()),
        "cases_missed_by_screen": int(y[~pos].sum()),
        "screen_negative_flagged_top5": k, "missed_cases_caught_by_model": extra,
        "share_of_missed_cases_caught": extra / y[~pos].sum(), "precision_of_extra_flags": extra / k,
        "total_flagged": int(pos.sum()) + k, "total_cases_caught": int(y[pos].sum()) + extra,
        "share_of_all_cases_caught": (int(y[pos].sum()) + extra) / y.sum(),
    }])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
LABELS = {"demographics": "Demographics", "routine": "Routine data", "phq8": "PHQ-8 score",
          "routine_phq8": "Routine data + PHQ-8"}


def plot_pr_curves(y, preds, selected) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    styles = {"demographics": (MUTED, ":"), "routine": (BLUE, "-"), "phq8": (ORANGE, "--"),
              "routine_phq8": (ORANGE, "-")}
    for name in MAIN_SETS:
        key = selected[name]
        p = preds[key].to_numpy()
        order = np.argsort(-p, kind="mergesort")
        tp = np.cumsum(y[order])
        color, ls = styles[name]
        ax.plot(tp / y.sum(), tp / np.arange(1, len(y) + 1), color=color, linestyle=ls,
                label=f"{LABELS[name]} ({key.split('|')[1].upper()}), PR-AUC {average_precision(y, p):.3f}")
    ax.axhline(y.mean(), color=INK_2, linewidth=1)
    ax.text(0.5, y.mean() - 0.035, f"prevalence {y.mean():.3f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Recall (share of cases flagged)")
    ax.set_ylabel("Precision")
    ax.set_ylim(-0.06, 1)
    ax.legend(loc="upper right")
    ax.set_title("Precision-recall on the 2015-2018 test set")
    fig.tight_layout()
    return fig


def plot_calibration(y, preds, alg: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors = {"none": BLUE, "balanced": ORANGE, "smote": MUTED}
    top = 0.0
    for strat in ("none", "balanced", "smote"):
        p = preds[f"routine|{alg}|{strat}"].to_numpy()
        bins = pd.qcut(p, 10, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
        ax.plot(g["p"], g["y"], marker="o", color=colors[strat], label=f"{strat} (Brier {np.mean((p - y) ** 2):.4f})")
        top = max(top, g["p"].max(), g["y"].max())
    ax.plot([0, top], [0, top], color=INK_2, linewidth=1, linestyle=":")
    ax.set_xlabel("Mean predicted risk (decile)")
    ax.set_ylabel("Observed share with suicidal ideation")
    ax.legend(loc="upper left", title="imbalance correction")
    ax.set_title(f"Calibration, routine data ({alg.upper()})")
    fig.tight_layout()
    return fig


def plot_fairness(table: pd.DataFrame, overall: float) -> plt.Figure:
    t = table.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.errorbar(t["roc_auc"], y, xerr=[t["roc_auc"] - t["roc_auc_ci_low"], t["roc_auc_ci_high"] - t["roc_auc"]],
                fmt="o", color=BLUE, elinewidth=1.5, capsize=0)
    ax.axvline(overall, color=INK_2, linewidth=1)
    ax.axvspan(overall - 0.05, overall + 0.05, color=MUTED, alpha=0.12, linewidth=0)
    ax.set_yticks(y, [f"{a}: {g}" for a, g in zip(t["attribute"], t["group"])])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ROC-AUC, 95% CI (band = overall ± 0.05)")
    ax.set_title("Routine-data model by subgroup (test set)")
    fig.tight_layout()
    return fig


def plot_capture(cap: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2.6))
    caught, extra = cap["cases_caught_by_screen"], cap["missed_cases_caught_by_model"]
    missed = cap["test_cases"] - caught - extra
    left = 0
    for value, color, label in [(caught, BLUE, "caught by PHQ-9 screen"),
                                (extra, ORANGE, "screen-negative, flagged by routine model"),
                                (missed, "#d9d8d2", "still missed")]:
        ax.barh(0, value, left=left, color=color, height=0.5, label=f"{label}: {value}")
        left += value
    ax.set_yticks([])
    ax.set_xlim(0, cap["test_cases"])
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=8)
    ax.set_title(f"Test-set adults with suicidal ideation (n = {cap['test_cases']}): screen, then flag the top 5% "
                 f"of screen-negatives ({cap['screen_negative_flagged_top5']} people)", fontsize=10)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.82, bottom=0.42)
    return fig


def main(argv=None) -> int:
    use_style()
    df = pd.read_parquet(PREDICTIONS)
    cv = pd.read_csv(REPORT_TABLES / "step4_cv_results.csv")
    sel = pd.read_csv(REPORT_TABLES / "step4_selected_models.csv")
    selected = {r.feature_set: f"{r.feature_set}|{r.algorithm}|{r.strategy}" for r in sel.itertuples()}
    y = df["si"].to_numpy().astype(int)
    model_cols = [c for c in df.columns if c.count("|") == 2 and ":" not in c]
    preds = df[model_cols]

    print(f"Bootstrapping {len(model_cols)} models x {B} resamples ...", flush=True)
    boot = bootstrap(y, {k: preds[k].to_numpy() for k in model_cols})
    metrics_table = test_metrics_table(y, preds, boot)
    metrics_table["selected"] = [f"{r.feature_set}|{r.algorithm}|{r.strategy}" in selected.values()
                                 for r in metrics_table.itertuples()]
    metrics_table.round(4).to_csv(REPORT_TABLES / "step5_test_metrics.csv", index=False)

    groups = {"sex": df["sex"].map(SEX),
              "age group": pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS).astype("string"),
              "ethnicity": df["ethnicity"].map(ETHNICITY)}
    hyp = hypotheses_table(y, preds, boot, selected, cv, groups)
    fair, gaps = fairness_table(y, preds[selected["routine"]].to_numpy(), groups)
    fair.round(4).to_csv(REPORT_TABLES / "step5_fairness_routine.csv", index=False)
    fair2, _ = fairness_table(y, preds[selected["routine_phq8"]].to_numpy(), groups)
    fair2.round(4).to_csv(REPORT_TABLES / "step5_fairness_routine_phq8.csv", index=False)
    hyp = pd.concat([hyp, gaps], ignore_index=True)
    hyp.round(4).to_csv(REPORT_TABLES / "step5_hypotheses.csv", index=False)

    neg_models = {"routine": selected["routine"], "routine (trained on screen-negative)":
                  next(c for c in df.columns if c.startswith("screen_negative:")),
                  "PHQ-8": selected["phq8"], "routine + PHQ-8": selected["routine_phq8"],
                  "demographics": selected["demographics"]}
    neg_tables, neg_diffs, caps = [], [], []
    for screen in ("phq9_score", "phq8_score"):
        t, d = screen_negative_table(df, neg_models, screen)
        neg_tables.append(t)
        neg_diffs.append(d)
        caps.append(capture_table(df, selected["routine"], screen))
        caps.append(capture_table(df, selected["routine_phq8"], screen))
    pd.concat(neg_tables).round(4).to_csv(REPORT_TABLES / "step5_screen_negative.csv", index=False)
    pd.concat(neg_diffs).round(4).to_csv(REPORT_TABLES / "step5_screen_negative_differences.csv", index=False)
    cap = pd.concat(caps, ignore_index=True)
    cap.round(4).to_csv(REPORT_TABLES / "step5_screen_then_flag.csv", index=False)

    strict_cols = [c for c in df.columns if c.startswith("si_strict:")]
    ys = df["si_strict"].to_numpy().astype(int)
    strict = pd.DataFrame([{"model": c.split(":")[1], "outcome": "si_strict", "cases": int(ys.sum()),
                            "prevalence": ys.mean(), **metrics(ys, df[c].to_numpy())} for c in strict_cols])
    strict.round(4).to_csv(REPORT_TABLES / "step5_sensitivity_si_strict.csv", index=False)

    overall_auc = metrics(y, preds[selected["routine"]].to_numpy())["roc_auc"]
    for name, fig in [("09_pr_curves", plot_pr_curves(y, preds, selected)),
                      ("10_calibration_imbalance_lr", plot_calibration(y, preds, "lr")),
                      ("11_calibration_imbalance_hgb", plot_calibration(y, preds, "hgb")),
                      ("12_fairness_roc_auc", plot_fairness(fair, overall_auc)),
                      ("13_screen_then_flag", plot_capture(cap.iloc[0]))]:
        save(fig, name)
        plt.close(fig)

    show = metrics_table[metrics_table.strategy.eq("none") | metrics_table.feature_set.eq("routine")]
    print(show[["feature_set", "algorithm", "strategy", "selected", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high",
                "roc_auc", "brier", "recall_top5", "precision_top5", "calibration_slope"]].round(3).to_string(index=False))
    print("\nHypotheses:")
    print(hyp.round(3).to_string(index=False))
    print("\nScreen-negative adults:")
    print(pd.concat(neg_tables)[["screen", "model", "n", "cases", "prevalence", "pr_auc", "pr_auc_ci_low",
                                 "pr_auc_ci_high", "roc_auc", "recall_top5"]].round(3).to_string(index=False))
    print(pd.concat(neg_diffs).round(3).to_string(index=False))
    print("\nScreen, then flag top 5% of screen-negatives:")
    print(cap.round(3).T.to_string())
    print("\nFairness (routine):")
    print(fair[["attribute", "group", "n", "cases", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
                "pr_auc_over_prevalence", "recall_at_global_top5"]].round(3).to_string(index=False))
    print("\nSensitivity (si_strict):")
    print(strict.round(3).to_string(index=False))
    return 0
