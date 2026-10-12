"""The raw NHANES files of the project: where they come from and their SHA-256 checksums.

scripts/fetch_data.py downloads this list; the build checks it before reading anything,
so every team member builds from identical data.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List

from .config import CYCLE_LABELS, CYCLES, REPO_ROOT, TREND_CYCLE

NHANES_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
NHANES_MODULES = [
    "DPQ",                # PHQ-9 depression screener (outcome = item 9)
    "DEMO",               # demographics, survey weights
    "RXQ_RX",             # prescription medications (one row per drug)
    "HUQ",                # healthcare use, self-rated health
    "MCQ", "DIQ", "BPQ",  # chronic conditions, diabetes, blood pressure
    "SLQ",                # sleep
    "SMQ", "DUQ",         # smoking, drug use
    "OCQ", "HIQ",         # employment, insurance
    "BMX",                # body measures (BMI)
]
# 2021-23 cycle: PHQ-9 only, for the descriptive trend check
TREND_MODULES = ["DEMO", "DPQ"]
# Drug lookup table with therapeutic classes, for the medication mapping in step 2.
# It is not tied to a cycle; CDC keeps it in the 1988 folder.
DRUG_LOOKUP = ("1988", "RXQ_DRUG")

CHECKSUMS = REPO_ROOT / "scripts" / "checksums.sha256"


def nhanes_files() -> Dict[str, str]:
    """Map each raw file's path under data/raw/ to its download URL."""
    jobs = [(CYCLE_LABELS[c][:4], f"{module}_{c}") for c in CYCLES for module in NHANES_MODULES]
    jobs += [(CYCLE_LABELS[TREND_CYCLE][:4], f"{module}_{TREND_CYCLE}") for module in TREND_MODULES]
    jobs.append(DRUG_LOOKUP)
    return {f"nhanes/{name}.xpt": NHANES_URL.format(year=year, name=name) for year, name in jobs}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_checksums(checksums: Path = CHECKSUMS) -> Dict[str, str]:
    """Read the recorded checksums (same format as `shasum -a 256`)."""
    if not checksums.exists():
        return {}
    lines = (line.split(maxsplit=1) for line in checksums.read_text().splitlines() if line.strip())
    return {rel: digest for digest, rel in lines}


def write_checksums(files: Iterable[str], raw_dir: Path, checksums: Path = CHECKSUMS) -> int:
    """Record the checksums of the local copies of `files`; return how many were written."""
    entries = {rel: sha256(raw_dir / rel) for rel in files if (raw_dir / rel).exists()}
    checksums.write_text("".join(f"{digest}  {rel}\n" for rel, digest in sorted(entries.items())))
    return len(entries)


def verify(files: Iterable[str], raw_dir: Path, checksums: Path = CHECKSUMS) -> List[str]:
    """Compare the local copies of `files` with the recorded checksums; return the problems."""
    expected = read_checksums(checksums)
    problems = []
    for rel in sorted(files):
        path = raw_dir / rel
        if not path.exists():
            problems.append(f"missing: {rel}")
        elif rel not in expected:
            problems.append(f"no recorded checksum: {rel}")
        elif sha256(path) != expected[rel]:
            problems.append(f"differs from the team's copy: {rel}")
    return problems
