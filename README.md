# Hidden Suicide Risk — Data Science in Health Care

Using NHANES 2005–2018, we test whether routinely recorded health information can identify adults with suicidal ideation, including those a depression screen misses (PHQ-9 < 10). Research question, hypotheses and plan: [`docs/project_description.md`](docs/project_description.md).

| Milestone                    | Date       |
| ---------------------------- | ---------- |
| Preliminary study hand-in    | 09.10.2026 |
| Final project hand-in (.zip) | 28.10.2026 |
| Presentation (Lucerne)       | 30.10.2026 |

## Repository layout

```
data-science-healthcare/
├── docs/
│   ├── project_description.md    # research question, hypotheses, methods
│   ├── codebook.md               # every variable and its NHANES source (generated)
│   └── harmonisation_rules.md    # appendix: every harmonisation rule (generated)
├── scripts/                      # command-line entry points, run in this order
│   ├── fetch_data.py             # step 1a: download and verify the raw data
│   ├── checksums.sha256          # SHA-256 of every raw file, so everyone has the same data
│   ├── build_dataset.py          # step 1b: merge and harmonise
│   ├── build_features.py         # step 2: medication classes, multimorbidity, social flags
│   ├── run_eda.py                # step 3: weighted prevalence with CIs, risk ratios, collinearity
│   ├── train_models.py           # step 4: tune and fit all models on 2005-2014 (~30 min)
│   ├── evaluate_models.py        # step 5: test metrics, bootstrap tests H1-H4, fairness
│   ├── explain_models.py         # step 6: odds ratios, SHAP
│   └── write_results.py          # docs/results.md from reports/tables
├── src/hidden_risk/              # all reusable code; imported by scripts, notebooks and tests
│   ├── config.py                 # paths, survey cycles, train/test split
│   ├── sources.py                # list of raw files, download URLs, checksum check
│   ├── io.py                     # read .xpt files (fixes SAS stored zeros)
│   ├── variables.py              # variable list: sources per cycle, missing codes, harmonisation
│   ├── harmonise.py              # merge modules on SEQN, apply variables.py
│   ├── cohort.py                 # adults with a complete PHQ-9, outcome, depression screens
│   ├── reports.py                # sample-size and missingness tables, codebook, appendix
│   ├── build.py                  # runs step 1: writes data/processed, reports/tables, docs
│   ├── plotting.py               # one chart style for every figure
│   ├── features.py               # step 2
│   ├── eda.py                    # step 3
│   ├── models.py                 # step 4: feature sets, pipelines, CV tuning
│   ├── evaluation.py             # step 5
│   ├── explain.py                # step 6
│   └── summary.py                # writes docs/results.md
├── notebooks/                    # 01 data overview, 03 step 3, 04 steps 4-6 (show outputs only)
├── reports/
│   ├── tables/                   # aggregate tables (committed)
│   └── figures/                  # plots for slides and report
├── tests/                        # pytest
├── pyproject.toml                # package definition and dependencies
├── requirements.txt              # installs the package and dev tools
└── data/                         # never committed, re-created by the scripts
    ├── raw/                      # downloads, never edited
    └── processed/                # analysis-ready .parquet files
```

**Where code goes:** anything used more than once belongs in `src/hidden_risk/`, not in a notebook. Notebooks import from it (`from hidden_risk.config import DATA_PROCESSED`) and only explore and plot.

## 1. Setup (once)

Requires Python 3.9 or newer and git. Check with `python3 --version` and `git --version`.

### macOS

```bash
# Only if python3 or git is missing:
xcode-select --install             # or, with Homebrew: brew install python git

git clone git@github.com:rogerjeasy/data-science-healthcare.git
cd data-science-healthcare
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Linux

```bash
# Only if python3, venv or git is missing:
sudo apt install python3 python3-venv git    # Debian / Ubuntu
sudo dnf install python3 git                 # Fedora

git clone git@github.com:rogerjeasy/data-science-healthcare.git
cd data-science-healthcare
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
git clone git@github.com:rogerjeasy/data-science-healthcare.git
cd data-science-healthcare
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- **No SSH key on GitHub?** Clone over HTTPS instead: `git clone https://github.com/rogerjeasy/data-science-healthcare.git`.
- **Every new terminal:** run `cd data-science-healthcare` and `source .venv/bin/activate` (Windows: `.venv\Scripts\Activate.ps1`) again before running the scripts. Your prompt shows `(.venv)` when the environment is active.

## 2. Download the data

```bash
python scripts/fetch_data.py
```

This downloads about 210 MB (under a minute) from [NHANES](https://wwwn.cdc.gov/nchs/nhanes/) into `data/raw/nhanes/`:

| Files | Content |
| ----- | ------- |
| 13 modules × 7 cycles (2005–2018) = 91 files | `DPQ`, `DEMO`, `RXQ_RX`, `HUQ`, `MCQ`, `DIQ`, `BPQ`, `SLQ`, `SMQ`, `DUQ`, `OCQ`, `HIQ`, `BMX` |
| 3 extra files | `RXQ_DRUG` (drug class lookup for step 2), `DEMO_L` and `DPQ_L` (2021–23 trend check) |

The run must end with **`Verified 94/94 files`**. Every file is checked against `scripts/checksums.sha256`, so a successful run means you have exactly the same data as the rest of the team.

- **Safe to re-run:** files already present are skipped.
- **Check without downloading:** `python scripts/fetch_data.py --verify-only`.
- **A file "differs from the team's copy":** re-run with `--force`. If it still differs, the source has changed, so tell the team. Only after the team agrees do we run `--write-checksums` and commit the new `checksums.sha256`.
- **All options:** `python scripts/fetch_data.py --help`.

## 3. Build the analysis dataset

```bash
python scripts/build_dataset.py
```

This takes a few seconds. It first checks the raw files against the checksums and stops if one is missing or differs. It then merges the 13 modules of each cycle on `SEQN` and keeps adults (18+) with a complete PHQ-9. It also harmonises variables that change between cycles and turns "refused / don't know" codes into missing values. Every rule is listed in [`docs/harmonisation_rules.md`](docs/harmonisation_rules.md), every variable in [`docs/codebook.md`](docs/codebook.md). It writes:

| File | Content |
|---|---|
| `data/processed/nhanes_2005_2018.parquet` | 36,259 adults, one row each: features, survey design, PHQ items and scores, outcome `si`, `split` |
| `data/processed/nhanes_2005_2018_prescriptions.parquet` | Their prescriptions, one row per drug (for the medication mapping in step 2) |
| `data/processed/nhanes_2021_2023_phq9.parquet` | 5,455 adults, PHQ-9 only (trend check) |
| `reports/tables/cohort_flow_by_cycle.csv` | Participants left after each selection step (all → adults → examined → complete PHQ-9) |
| `reports/tables/cohort_by_cycle.csv` | Adults and suicidal-ideation cases per cycle, and the share of cases each depression screen misses |
| `reports/tables/missingness_by_cycle.csv` | % missing per variable and cycle |
| `docs/codebook.md`, `docs/harmonisation_rules.md` | Generated documentation: do not edit by hand |

The run must end with a total of **36259 adults and 1366 SI cases**. Load the data in a notebook with:

```python
import pandas as pd
from hidden_risk.config import DATA_PROCESSED
df = pd.read_parquet(DATA_PROCESSED / "nhanes_2005_2018.parquet")
```

**Two PHQ scores:** `phq9_score` / `phq9_positive` are the depression screen as used in practice (PHQ-9 ≥ 10). They contain item 9, which is the outcome, so they must not be used as model features. `phq8_score` / `phq8_positive` leave item 9 out and are safe as a predictor.

**Check everything:** `python -m pytest` runs the tests, including checks that the built dataset matches the numbers in the project description.

## 4. Explore the data

```bash
jupyter lab notebooks/01_data_overview.ipynb
```

`01_data_overview` shows the cohort, suicidal ideation over time, the cases the depression screen misses, subgroups, crude risk ratios and missing data. It also saves its charts to `reports/figures/` for the report and slides. For new charts, call `use_style()` from `hidden_risk.plotting` so every figure shares the same colour-blind-safe style. If you use VS Code instead, open the notebook and choose the `.venv` kernel.

## 5. Steps 2–6: features, analysis, models

```bash
python scripts/build_features.py    # step 2 -> data/processed/nhanes_2005_2018_features.parquet
python scripts/run_eda.py           # step 3 -> reports/tables/step3_*, figures 06-08
python scripts/train_models.py      # step 4 -> reports/tables/step4_*, test predictions, models (~30 min)
python scripts/evaluate_models.py   # step 5 -> reports/tables/step5_*, figures 09-13
python scripts/explain_models.py    # step 6 -> reports/tables/step6_*, figures 14-17
python scripts/write_results.py     # -> docs/results.md
```

All results are in [`docs/results.md`](docs/results.md). Models are tuned with stratified 5-fold CV on 2005–2014 only; the algorithm per feature set is chosen by CV PR-AUC, so the 2015–2018 test set is used once, for evaluation.

## Data rules

- **Never commit data.** `data/` and `*.xpt` are git-ignored; everyone re-creates them with the scripts.
- **Report only aggregate results** in slides and reports.

## Data sources and licences

- CDC / NCHS, *National Health and Nutrition Examination Survey*, 2005–2018 and 2021–2023. Public domain.
