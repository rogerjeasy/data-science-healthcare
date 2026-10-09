# Hidden Suicide Risk — Data Science in Health Care

Using NHANES 2005–2018, we test whether routinely recorded health information can identify adults with suicidal ideation, including those a depression screen misses (PHQ-8 < 10). Research question, hypotheses and plan: [`docs/project_description.md`](docs/project_description.md).

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
│   └── codebook.md               # every variable and its NHANES source (generated)
├── scripts/                      # command-line entry points, run in this order
│   ├── fetch_data.py             # step 0: download raw data
│   ├── checksums.sha256          # SHA-256 of every raw file, so everyone has the same data
│   └── build_dataset.py          # step 1: merge and harmonise
├── src/hidden_risk/              # all reusable code; imported by scripts, notebooks and tests
│   ├── config.py                 # paths, survey cycles, train/test split
│   ├── io.py                     # read .xpt files (fixes SAS stored zeros)
│   ├── variables.py              # variable list: sources per cycle, missing codes, harmonisation
│   ├── harmonise.py              # merge modules on SEQN, apply variables.py
│   ├── cohort.py                 # adults with a complete PHQ-9, outcome definition
│   └── build.py                  # writes data/processed, reports/tables, docs/codebook.md
├── notebooks/                    # exploration only, numbered: 01_eda.ipynb, 02_...
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

**Where code goes:** anything used more than once belongs in `src/hidden_risk/`, not in a notebook. Notebooks import from it (`from hidden_risk.config import DATA_PROCESSED`) and only explore and plot. Step 2 (features) and step 4 (models) will add `features.py` and `models.py` next to the existing modules.

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

This downloads about 210 MB (under a minute) into `data/raw/`:

| Source                                                | Files                                                                                                                                          |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| [NHANES](https://wwwn.cdc.gov/nchs/nhanes/) 2005–2018 | 13 modules × 7 cycles:`DPQ`, `DEMO`, `RXQ_RX`, `HUQ`, `MCQ`, `DIQ`, `BPQ`, `SLQ`, `SMQ`, `DUQ`, `OCQ`, `HIQ`, `BMX` |
| NHANES extras                                         | `RXQ_DRUG` (drug class lookup), `DEMO_L` and `DPQ_L` (2021–23 trend check)                                                              |
| [BFS](https://www.bfs.admin.ch/)                       | Suicides by method, sex and age; assisted suicide by sex and age (Swiss context only)                                                          |

The run must end with **`Verified 97/97 files`**. Every file is checked against `scripts/checksums.sha256`, so a successful run means you have exactly the same data as the rest of the team.

- **Safe to re-run:** files already present are skipped.
- **One source only:** `python scripts/fetch_data.py nhanes` or `bfs`.
- **Check without downloading:** `python scripts/fetch_data.py --verify-only`.
- **A file "differs from the team's copy":** re-run with `--force`. If it still differs, the source has changed, so tell the team. Only after the team agrees do we run `--write-checksums` and commit the new `checksums.sha256`.
- **All options:** `python scripts/fetch_data.py --help`.

## 3. Build the analysis dataset

```bash
python scripts/build_dataset.py
```

This takes a few seconds. It merges the 13 modules of each cycle on `SEQN` and keeps adults (18+) with a complete PHQ-9. It also harmonises variables that change between cycles and turns "refused / don't know" codes into missing values. Every rule is listed in [`docs/codebook.md`](docs/codebook.md). It writes:

| File | Content |
|---|---|
| `data/processed/nhanes_2005_2018.parquet` | 36,259 adults, one row each: features, survey design, PHQ items, outcome `si`, `split` |
| `data/processed/nhanes_2005_2018_prescriptions.parquet` | Their prescriptions, one row per drug (for the medication mapping in step 2) |
| `data/processed/nhanes_2021_2023_phq9.parquet` | 5,455 adults, PHQ-9 only (trend check) |
| `reports/tables/cohort_by_cycle.csv` | Adults and suicidal-ideation cases per cycle |
| `reports/tables/missingness_by_cycle.csv` | % missing per variable and cycle |

The run must end with a total of **36259 adults and 1366 SI cases**. Load the data in a notebook with:

```python
import pandas as pd
from hidden_risk.config import DATA_PROCESSED
df = pd.read_parquet(DATA_PROCESSED / "nhanes_2005_2018.parquet")
```

**Check everything:** `python -m pytest` runs the tests, including a check that the built dataset matches the numbers in the project description.

## 4. Explore the data

```bash
jupyter lab notebooks/01_data_overview.ipynb
```

`01_data_overview` shows the cohort, suicidal ideation over time, the cases the depression screen misses, subgroups, crude risk ratios and missing data. It also saves its charts to `reports/figures/` for the report and slides. For new charts, call `use_style()` from `hidden_risk.plotting` so every figure shares the same colour-blind-safe style. If you use VS Code instead, open the notebook and choose the `.venv` kernel.

## Data rules

- **Never commit data.** `data/` and `*.xpt` are git-ignored; everyone re-creates them with the scripts.
- **Report only aggregate results** in slides and reports.

## Data sources and licences

- CDC / NCHS, *National Health and Nutrition Examination Survey*, 2005–2018 and 2021–2023. Public domain.
- Bundesamt für Statistik (BFS), tables T 14.03.04.01.11, .12 and .14, via opendata.swiss.
