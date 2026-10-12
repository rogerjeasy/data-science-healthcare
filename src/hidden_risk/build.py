"""Step 1: build the analysis-ready datasets from the raw NHANES files.

Outputs
-------
data/processed/nhanes_2005_2018.parquet                adults with a complete PHQ-9, one row each
data/processed/nhanes_2005_2018_prescriptions.parquet  their prescriptions, one row per drug
data/processed/nhanes_2021_2023_phq9.parquet           2021-23 PHQ-9 for the trend check
reports/tables/cohort_flow_by_cycle.csv                participants left after each selection step
reports/tables/cohort_by_cycle.csv                     sample size and outcome per cycle
reports/tables/missingness_by_cycle.csv                % missing per variable and cycle
docs/codebook.md                                       every variable and its NHANES source
docs/harmonisation_rules.md                            appendix: every harmonisation rule
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .cohort import PHQ_ITEMS, add_outcome, cohort_flow, select_cohort
from .config import CYCLES, DATA_PROCESSED, DATA_RAW, DOCS, REPORT_TABLES, TRAIN_CYCLES, TREND_CYCLE
from .harmonise import harmonise_cycle, load_cycle, load_prescriptions
from .io import read_xpt
from .reports import cohort_by_cycle, codebook_markdown, harmonisation_markdown, missingness_by_cycle
from .sources import nhanes_files, verify
from .variables import BY_NAME

TREND_VARIABLES = ["age", "sex", "ethnicity", "weight_interview", "weight_mec", "strata", "psu",
                   *PHQ_ITEMS, "phq_difficulty"]


def harmonise_participants(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Every NHANES participant of 2005-2018, all modules merged and harmonised."""
    frames = []
    for cycle in CYCLES:
        print(f"  {CYCLES[cycle]}: merging and harmonising")
        frames.append(harmonise_cycle(load_cycle(cycle, raw_dir), cycle))
    return pd.concat(frames, ignore_index=True)


def build_persons(participants: pd.DataFrame) -> pd.DataFrame:
    """The 2005-2018 cohort with outcome, split and pooled weight."""
    df = add_outcome(select_cohort(participants))
    train = df["cycle"].isin([CYCLES[c] for c in TRAIN_CYCLES])
    df.insert(2, "split", train.map({True: "train", False: "test"}))
    df["weight_mec_pooled"] = df["weight_mec"] / len(CYCLES)
    return df


def build_prescriptions(persons: pd.DataFrame, raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    rx = pd.concat([load_prescriptions(c, raw_dir) for c in CYCLES], ignore_index=True)
    return rx[rx["seqn"].isin(persons["seqn"])].reset_index(drop=True)


def build_trend(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    raw = read_xpt(f"DEMO_{TREND_CYCLE}", raw_dir).merge(
        read_xpt(f"DPQ_{TREND_CYCLE}", raw_dir), on="SEQN", how="left", validate="one_to_one")
    variables = [BY_NAME[name] for name in TREND_VARIABLES]
    return add_outcome(select_cohort(harmonise_cycle(raw, TREND_CYCLE, variables)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=DATA_RAW, help="raw data folder (default: data/raw)")
    ap.add_argument("--out", type=Path, default=DATA_PROCESSED, help="output folder (default: data/processed)")
    args = ap.parse_args(argv)

    problems = verify(nhanes_files(), args.raw)
    if problems:
        print(f"Cannot build: {len(problems)} raw file(s) are missing or differ from the team's copy.")
        for line in problems:
            print(f"  ! {line}")
        print("Run `python scripts/fetch_data.py` first.")
        return 1
    print(f"Verified {len(nhanes_files())} raw files against the recorded checksums")

    args.out.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)

    print("Building the 2005-2018 cohort")
    participants = harmonise_participants(args.raw)
    persons = build_persons(participants)
    prescriptions = build_prescriptions(persons, args.raw)
    trend = build_trend(args.raw)

    outputs = {
        "nhanes_2005_2018.parquet": persons,
        "nhanes_2005_2018_prescriptions.parquet": prescriptions,
        "nhanes_2021_2023_phq9.parquet": trend,
    }
    for name, df in outputs.items():
        df.to_parquet(args.out / name, index=False)
        print(f"Wrote {len(df):>7,} rows x {df.shape[1]:>2} columns -> {args.out / name}")

    cohort = cohort_by_cycle(persons)
    tables = {
        "cohort_flow_by_cycle.csv": cohort_flow(participants),
        "cohort_by_cycle.csv": cohort,
        "missingness_by_cycle.csv": missingness_by_cycle(persons).reset_index(),
    }
    for name, table in tables.items():
        table.to_csv(REPORT_TABLES / name, index=False)
    (DOCS / "codebook.md").write_text(codebook_markdown())
    (DOCS / "harmonisation_rules.md").write_text(harmonisation_markdown())
    print(f"Wrote reports/tables/{{{', '.join(tables)}}}, docs/codebook.md, "
          "docs/harmonisation_rules.md\n")
    print(cohort.to_string(index=False))
    return 0
