"""Project paths and the NHANES survey cycles used in the study."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
REPORT_TABLES = REPO_ROOT / "reports" / "tables"
REPORT_FIGURES = REPO_ROOT / "reports" / "figures"
DOCS = REPO_ROOT / "docs"

# NHANES file suffix -> survey cycle
CYCLES = {
    "D": "2005-2006",
    "E": "2007-2008",
    "F": "2009-2010",
    "G": "2011-2012",
    "H": "2013-2014",
    "I": "2015-2016",
    "J": "2017-2018",
}
TREND_CYCLE = "L"  # 2021-2023: descriptive trend check only
CYCLE_LABELS = {**CYCLES, TREND_CYCLE: "2021-2023"}

# Temporal split (docs/project_description.md, step 4)
TRAIN_CYCLES = ("D", "E", "F", "G", "H")  # 2005-2014
TEST_CYCLES = ("I", "J")                  # 2015-2018
