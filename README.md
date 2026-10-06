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
├── docs/project_description.md   # research question, hypotheses, methods
├── scripts/fetch_data.py         # downloads all raw data
├── scripts/checksums.sha256      # SHA-256 of every raw file, so everyone has the same data
├── requirements.txt
└── data/raw/                     # created by the script, never committed
    ├── nhanes/                   # 94 .xpt files
    └── bfs/                      # 3 .xlsx files
```

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

## Data rules

- **Never commit data.** `data/` and `*.xpt` are git-ignored; everyone re-creates them with the script.
- **Report only aggregate results** in slides and reports.

## Data sources and licences

- CDC / NCHS, *National Health and Nutrition Examination Survey*, 2005–2018 and 2021–2023. Public domain.
- Bundesamt für Statistik (BFS), tables T 14.03.04.01.11, .12 and .14, via opendata.swiss.
