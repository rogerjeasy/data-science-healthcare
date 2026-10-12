# Hidden Suicide Risk: Can routinely recorded health information identify suicidal ideation that a depression screen misses?

*Data Science in Health Care. Sections 1–6 follow the preliminary study submitted on 09.10.2026.*

## 1. Problem statement

Suicide is a leading cause of preventable death. More than 720,000 people die by suicide worldwide every year. Many people who die by suicide were in contact with a physician in the preceding year [1]. The routine tool for spotting mental distress in primary care is a depression questionnaire such as the PHQ-9, with a positive screen at a score of 10 or more [2]. **But suicidal thoughts are not limited to people who screen positive for depression.** In our pilot analysis of NHANES, **42% of adults who reported thoughts of self-harm or of being better off dead scored below 10 on the other eight PHQ items**, so a standard depression screen would miss them.

At the same time, an electronic health record already holds structured information a clinician rarely has time to cross-reference: prescriptions (antidepressants, opioids, benzodiazepines), healthcare use (hospital stays, mental-health visits), chronic conditions, sleep problems, employment and insurance status. **It is unclear whether this routinely recorded information carries enough signal to flag suicidal ideation, especially in patients the depression screen does not catch.** This project quantifies that signal on real, nationally representative health data.

## 2. Research question and hypotheses

**Main research question**

> To what extent can routinely recorded health information identify adults with current suicidal ideation, and does it add value beyond a depression screen, in particular for adults who screen negative (PHQ-9 < 10)?

**Sub-questions**

1. How should the severe class imbalance (about 1:26) be handled: class weighting, SMOTE, or no correction?
2. Does performance hold when a model trained on older survey waves is applied to newer ones, and is it similar across sex, age groups and ethnicity?

**Hypotheses** (all tested on the 2015–2018 test set with paired bootstrap 95% CIs)

*Main question*

- **H1:** A model using routine health data achieves a PR-AUC at least **twice the outcome prevalence**, and higher than a demographics-only model (CI of the difference excludes 0).
- **H2:** Adding routine health data to the PHQ-9 score increases PR-AUC compared with PHQ-9 alone (CI of the difference excludes 0).

*Sub-question 1 (class imbalance)*

- **H3:** Class weighting and SMOTE do not improve PR-AUC over the uncorrected model, but they worsen calibration (higher Brier score; CI of the difference excludes 0), because they shift predicted probabilities away from the true prevalence [3].

*Sub-question 2 (robustness)*

- **H4:** The model trained on 2005–2014 keeps its discrimination on 2015–2018 (test ROC-AUC within 0.05 of the cross-validated ROC-AUC), and ROC-AUC differs by less than 0.05 between sexes, age groups and ethnic groups. ROC-AUC is used here rather than PR-AUC because PR-AUC depends on prevalence, which differs between waves and subgroups.

## 3. Data sources

| Source                                                                  | Content                                                                                                                                                                                                                                                                                                                                                                                 | Size / years                                                                                                       | Role                                        |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| **NHANES** (US CDC / NCHS) [4]                                    | Individual-level interview and examination data, linked by the respondent ID`SEQN`                                                                                                                                                                                                                                                                                                    | 7 cycles 2005–2018 →**36,259 adults** with a complete PHQ-9, **1,366** with suicidal ideation (3.8%) | Main modelling dataset                      |
| NHANES modules used                                                     | `DPQ` (PHQ-9), `DEMO` (age, sex, ethnicity, education, income ratio, marital status, survey weights), `RXQ_RX` (prescription drug names), `HUQ` (self-rated health, hospital stay, mental-health visit), `MCQ`/`DIQ`/`BPQ` (chronic conditions), `SLQ` (sleep trouble), `SMQ` (smoking), `DUQ` (drug use), `OCQ` (employment), `HIQ` (insurance), `BMX` (BMI) | 13 modules × 7 cycles = 91 files                                                                                  | Feature construction                        |
| NHANES 2021–2023 (`DPQ_L`)                                           | PHQ-9 only (the public drug file lacks drug names)                                                                                                                                                                                                                                                                                                                                      | 5,455 adults                                                                                                       | Descriptive trend check only                |

**Outcome:** `si = 1` if PHQ-9 item 9 ("thoughts that you would be better off dead, or of hurting yourself") is answered on at least *several days* in the past two weeks. A stricter outcome (*more than half the days*, 424 cases, ≈1.2%) is used as a sensitivity analysis.

**Leakage rule:** PHQ items 1–9 are **excluded** from the "routine data" feature set and enter only as the PHQ-9 score in the explicit comparison (H2).

## 4. Approach and methodology

**Step 1 — Acquisition and harmonisation (week 1, done).** Download the 91 `.xpt` files, verified against SHA-256 checksums, and merge all modules on `SEQN` per cycle. Restrict to adults (18+) with a complete PHQ-9. SAS transport stores 0 as ≈5·10⁻⁷⁹, so these values are set to 0 on import. "Refused / don't know" codes are recoded to missing per variable. Variables that change between cycles are harmonised, for example `HUQ050` → `HUQ051` (healthcare visits, 2013) and the new sleep module from 2015; each variable is kept only for the ages asked in every cycle. All rules are listed in the appendix, [`harmonisation_rules.md`](harmonisation_rules.md). Outputs: one row per adult, prescriptions in long format (one row per drug), the 2021–23 PHQ-9 file, and tables of sample size and missingness per cycle.

**Step 2 — Feature engineering (week 1–2).**

- *Medication:* map free-text drug names in `RXQ_RX` (one row per drug) to four classes (antidepressants, opioids, benzodiazepines/Z-drugs, antipsychotics/mood stabilisers) plus a **polypharmacy** count of distinct drugs.
- *Multimorbidity:* count of 10 diagnosed chronic conditions (arthritis, heart disease, stroke, COPD, cancer, diabetes, hypertension, …).
- *Healthcare use:* hospital stay and mental-health visit in the past year, number of visits, self-rated health.
- *Social:* poverty-income ratio, not working, uninsured, marital status, education.
- *Structural missingness:* drug use is only available for ages 20–59, and chronic conditions, smoking and marital status only from age 20 (step 1). We add missing-indicators rather than imputing these as "no".

**Step 3 — Exploratory analysis.** Report prevalence by cycle, sex, age and ethnicity with the NHANES survey weights (`WTMEC2YR` combined over cycles, strata `SDMVSTRA`, PSU `SDMVPSU`) so the figures describe the US adult population. Produce univariate risk ratios and a correlation check for collinear features (e.g. antidepressant use vs. mental-health visit).

**Step 4 — Modelling.** Temporal split: **train 2005–2014 (26,057 adults, 983 cases) / test 2015–2018 (10,202 adults, 383 cases)**. Hyperparameters are tuned with stratified 5-fold cross-validation inside the training set only. We fit four nested feature sets (demographics → routine data → PHQ-9 only → routine + PHQ-8) with two algorithms:

- **L2- or elastic-net logistic regression**, interpretable as odds ratios;
- **gradient boosting** (scikit-learn `HistGradientBoostingClassifier` or XGBoost).

The imbalance comparison (sub-question 1, H3) runs each model three ways: with no correction, with balanced class weights, and with SMOTE applied inside the CV folds only.

**Step 5 — Evaluation.**

- Primary metric: **PR-AUC**, compared against the prevalence baseline. Secondary: ROC-AUC, Brier score and a calibration plot.
- Clinical operating point: **recall and precision when flagging the top 5% of patients**, reported as "flag 1 in 20 patients → catch X% of cases, of whom Y% are true positives".
- Hypothesis tests via bootstrap CIs on the test set.
- Fairness (H4): ROC-AUC, PR-AUC relative to subgroup prevalence, and recall at top 5% by sex, age group and ethnicity.

**Step 6 — Explainability.** SHAP values for the gradient-boosting model and odds ratios for logistic regression. Global importance ranking plus 2–3 individual patient explanations (e.g. "flagged mainly because of antipsychotic prescription, poor self-rated health and not working").

**Tools:** Python (pandas, scikit-learn, imbalanced-learn, shap, matplotlib); code in a shared Git repository.

## 5. Ethics and limitations

- **Public, de-identified data.** NHANES is released for public research and needs no ethics approval. We report only aggregate results.
- **Not a diagnostic tool.** The outcome is *self-reported ideation*, not suicide attempts or deaths, and the design is **cross-sectional**. The model *flags*; it does not *predict the future*. We discuss the harm of false positives (stigma, unnecessary referrals) and false negatives.
- **Transferability.** The data are from the US.
- **Label quality.** PHQ item 9 captures passive thoughts as well as active ideation, so it is known to over-identify risk [2].

## 6. Out of scope (future work)

NLP on clinical notes (ClinicalBERT), multimodal late fusion, and 30-day post-discharge prediction. These need credentialed access to MIMIC-IV/ScAN and longitudinal outcome data, and they are the natural next step once the structured-data signal has been quantified here.

## 7. Timeline and task split

| Period          | Milestone                                                                        | Owner        |
| --------------- | -------------------------------------------------------------------------------- | ------------ |
| 25.09 – 01.10  | Download, merge and harmonise all cycles; outcome definition; missingness report | 2 members    |
| 25.09 – 09.10  | Write preliminary study (this plan, max. 2 pages + appendix)                     | all          |
| **09.10** | **Hand-in preliminary study**                                              |              |
| 09.10 – 16.10  | Feature engineering (medication mapping, multimorbidity), weighted EDA           | 2 members    |
| 12.10 – 21.10  | Modelling, imbalance comparison, bootstrap tests                                 | 2 members    |
| 19.10 – 24.10  | SHAP, fairness, calibration, clinical operating point                            | 1–2 members |
| 22.10 – 28.10  | Slides (≤ 10), code clean-up, GenAI disclosure                                  | all          |
| **28.10** | **Hand-in final project (.zip)** · 30.10 presentation in Lucerne          |              |

## 8. Open point for step 4: PHQ-8 or PHQ-9?

The submitted text uses the PHQ-9 score in H2 and in the leakage rule, but step 4 still lists "routine + PHQ-8". The two scores answer different questions, and step 1 provides both (`phq8_score`, `phq9_score`, `phq8_positive`, `phq9_positive`):

- **As the screen (who counts as "screen negative"):** PHQ-9 ≥ 10 is the screen used in practice. With it, 34.6% of cases (472 of 1,366) screen negative. The 42% in the problem statement (579 cases) refers to the other eight items, i.e. PHQ-8 < 10.
- **As a predictor:** the PHQ-9 score contains item 9, which is the outcome. All 107 adults whom item 9 moves from screen-negative to screen-positive are cases, and nobody with a PHQ-9 score of 0 can be a case. On the 2015–2018 test set the score alone reaches PR-AUC 0.42 with PHQ-9 against 0.32 with PHQ-8, a gain that comes only from the label. A PHQ-9 baseline in H2 is therefore inflated; the PHQ-8 score is the leakage-free baseline.

---

## References

1. Ahmedani, B. K. et al. (2014). Health care contacts in the year before suicide death. *Journal of General Internal Medicine*, 29(6), 870–877.
2. Kroenke, K., Spitzer, R. L., & Williams, J. B. W. (2001). The PHQ-9: validity of a brief depression severity measure. *Journal of General Internal Medicine*, 16(9), 606–613.
3. van den Goorbergh, R., van Smeden, M., Timmerman, D., & Van Calster, B. (2022). The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. *Journal of the American Medical Informatics Association*, 29(9), 1525–1534.
4. Centers for Disease Control and Prevention, National Center for Health Statistics. *National Health and Nutrition Examination Survey data, 2005–2018*. https://wwwn.cdc.gov/nchs/nhanes/, accessed 25.09.2026.
