#!/usr/bin/env python3
"""Download the project's raw data into data/raw/ and check it matches the team's copy.

Sources
-------
nhanes  CDC NHANES 2005-2018 (13 modules x 7 cycles), the drug lookup table
        RXQ_DRUG and the 2021-23 PHQ-9 trend files DEMO_L / DPQ_L.
bfs     Swiss Federal Statistical Office suicide tables (context only).
all     both (default)

Examples
--------
    python scripts/fetch_data.py                 # download everything, then verify
    python scripts/fetch_data.py nhanes          # one source only
    python scripts/fetch_data.py --force         # re-download files already present
    python scripts/fetch_data.py --verify-only   # only check local files

Every file is compared with the SHA-256 recorded in scripts/checksums.sha256,
so all team members can confirm they work on identical data.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: run `pip install -r requirements.txt` first.")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "raw"
CHECKSUMS = Path(__file__).with_name("checksums.sha256")
USER_AGENT = "Mozilla/5.0 (data-science-healthcare course project; fetch_data.py)"

# --------------------------------------------------------------------------- #
# Source definitions (see docs/project_description.md, section 3)
# --------------------------------------------------------------------------- #
NHANES_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
NHANES_CYCLES = {"D": 2005, "E": 2007, "F": 2009, "G": 2011, "H": 2013, "I": 2015, "J": 2017}
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
# Extra single files: (year folder, file name)
NHANES_EXTRA = [
    (1988, "RXQ_DRUG"),  # drug lookup table with therapeutic classes, for the medication mapping
    (2021, "DEMO_L"),    # 2021-23 cycle: descriptive trend check only
    (2021, "DPQ_L"),
]

BFS_URL = "https://dam-api.bfs.admin.ch/hub/api/dam/assets/{asset}/master"
BFS_TABLES = {
    "suicide_methods_by_sex_1995_2024": 36262349,       # T 14.03.04.01.12
    "suicide_methods_by_age_sex": 36262368,             # T 14.03.04.01.11
    "assisted_suicide_by_sex_age_2003_2024": 36262348,  # T 14.03.04.01.14
}

XPT_MAGIC = b"HEADER"  # SAS transport file
XLSX_MAGIC = b"PK"     # xlsx is a zip archive


def nhanes_files() -> dict[str, tuple[str, bytes]]:
    """Map each NHANES file's path under data/raw/ to (download URL, expected first bytes)."""
    jobs = [(year, f"{m}_{c}") for c, year in NHANES_CYCLES.items() for m in NHANES_MODULES]
    return {f"nhanes/{name}.xpt": (NHANES_URL.format(year=year, name=name), XPT_MAGIC)
            for year, name in jobs + NHANES_EXTRA}


def bfs_files() -> dict[str, tuple[str, bytes]]:
    """Map each BFS table's path under data/raw/ to (download URL, expected first bytes)."""
    return {f"bfs/{name}.xlsx": (BFS_URL.format(asset=asset), XLSX_MAGIC)
            for name, asset in BFS_TABLES.items()}


SOURCES = {"nhanes": nhanes_files, "bfs": bfs_files}


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def has_magic(path: Path, magic: bytes) -> bool:
    with path.open("rb") as fh:
        return fh.read(len(magic)) == magic


def download(s: requests.Session, url: str, path: Path, magic: bytes, retries: int = 3) -> None:
    """Download url to path, retrying server errors; raise if the file type is wrong."""
    for attempt in range(1, retries + 1):
        try:
            r = s.get(url, timeout=120)
            if r.status_code < 500:
                break
        except requests.RequestException:
            if attempt == retries:
                raise
        time.sleep(2 * attempt)
    r.raise_for_status()
    # CDC answers some missing files with a 200 HTML page, so check the file signature
    if not r.content.startswith(magic):
        raise ValueError("server did not return the expected file type")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(path)


def fetch(files: dict[str, tuple[str, bytes]], out: Path, force: bool, workers: int) -> list[str]:
    """Download every file not yet present (or all with force); return the failures."""
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT

    def one(rel: str) -> tuple[str, str]:
        url, magic = files[rel]
        path = out / rel
        if path.exists() and not force and has_magic(path, magic):
            return rel, "present"
        try:
            download(s, url, path, magic)
            return rel, "downloaded"
        except (requests.RequestException, ValueError) as e:
            return rel, f"{e} ({url})"

    print(f"Downloading {len(files)} files -> {out}")
    downloaded = present = 0
    failures = []
    with ThreadPoolExecutor(workers) as pool:
        for rel, status in pool.map(one, sorted(files)):
            if status == "downloaded":
                downloaded += 1
                print(f"  + {rel}")
            elif status == "present":
                present += 1
            else:
                failures.append(f"{rel}: {status}")
    print(f"Downloaded {downloaded}, already present {present}, failed {len(failures)}")
    return failures


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_checksums() -> dict[str, str]:
    """Read scripts/checksums.sha256 (same format as `shasum -a 256`)."""
    if not CHECKSUMS.exists():
        return {}
    lines = (line.split(maxsplit=1) for line in CHECKSUMS.read_text().splitlines() if line.strip())
    return {rel: digest for digest, rel in lines}


def write_checksums(files: dict[str, tuple[str, bytes]], out: Path) -> None:
    entries = read_checksums()
    entries.update({rel: sha256(out / rel) for rel in files if (out / rel).exists()})
    CHECKSUMS.write_text("".join(f"{digest}  {rel}\n" for rel, digest in sorted(entries.items())))
    print(f"Wrote {len(entries)} checksums -> {CHECKSUMS.relative_to(REPO_ROOT)}")


def verify(files: dict[str, tuple[str, bytes]], out: Path) -> list[str]:
    """Compare local files with the recorded checksums; return the problems."""
    expected = read_checksums()
    problems = []
    for rel in sorted(files):
        path = out / rel
        if not path.exists():
            problems.append(f"missing: {rel}")
        elif rel not in expected:
            problems.append(f"no recorded checksum: {rel}")
        elif sha256(path) != expected[rel]:
            problems.append(f"differs from the team's copy: {rel}")
    print(f"Verified {len(files) - len(problems)}/{len(files)} files against "
          f"{CHECKSUMS.relative_to(REPO_ROOT)}")
    return problems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default="all", choices=["all", *SOURCES])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="target folder (default: data/raw)")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads (default: 6)")
    ap.add_argument("--verify-only", action="store_true", help="skip downloading, only check local files")
    ap.add_argument("--write-checksums", action="store_true",
                    help="record the local files' checksums in scripts/checksums.sha256 "
                         "(only when the team agrees to move to a new data version)")
    args = ap.parse_args()

    files: dict[str, tuple[str, bytes]] = {}
    for name in SOURCES if args.source == "all" else [args.source]:
        files.update(SOURCES[name]())

    failures = [] if args.verify_only else fetch(files, args.out, args.force, args.workers)
    if args.write_checksums:
        write_checksums(files, args.out)
        problems = []
    else:
        problems = verify(files, args.out)

    for line in failures + problems:
        print(f"  ! {line}")
    if any(p.startswith("differs") for p in problems):
        print("Some files differ from the team's copy: the source may have been updated. "
              "Re-run with --force; if they still differ, tell the team.")
    return 1 if failures or problems else 0


if __name__ == "__main__":
    sys.exit(main())
