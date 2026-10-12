#!/usr/bin/env python3
"""Download the project's raw data into data/raw/ and check it matches the team's copy.

Source: CDC NHANES 2005-2018 (13 modules x 7 cycles = 91 files), the drug lookup
table RXQ_DRUG and the 2021-23 PHQ-9 trend files DEMO_L / DPQ_L. The file list is
in src/hidden_risk/sources.py.

Examples
--------
    python scripts/fetch_data.py                 # download everything, then verify
    python scripts/fetch_data.py --force         # re-download files already present
    python scripts/fetch_data.py --verify-only   # only check local files

Every file is compared with the SHA-256 recorded in scripts/checksums.sha256,
so all team members can confirm they work on identical data.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import requests

    from hidden_risk.config import DATA_RAW, REPO_ROOT
    from hidden_risk.sources import CHECKSUMS, nhanes_files, verify, write_checksums
except ImportError:
    sys.exit("Missing dependency: run `pip install -r requirements.txt` first.")

USER_AGENT = "Mozilla/5.0 (data-science-healthcare course project; fetch_data.py)"
XPT_MAGIC = b"HEADER"  # first bytes of a SAS transport file


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def is_xpt(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(len(XPT_MAGIC)) == XPT_MAGIC


def download(s: requests.Session, url: str, path: Path, retries: int = 3) -> None:
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
    if not r.content.startswith(XPT_MAGIC):
        raise ValueError("server did not return a SAS transport file")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(path)


def fetch(files: dict[str, str], out: Path, force: bool, workers: int) -> list[str]:
    """Download every file not yet present (or all with force); return the failures."""
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT

    def one(rel: str) -> tuple[str, str]:
        url = files[rel]
        path = out / rel
        if path.exists() and not force and is_xpt(path):
            return rel, "present"
        try:
            download(s, url, path)
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
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA_RAW, help="target folder (default: data/raw)")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads (default: 6)")
    ap.add_argument("--verify-only", action="store_true", help="skip downloading, only check local files")
    ap.add_argument("--write-checksums", action="store_true",
                    help="record the local files' checksums in scripts/checksums.sha256 "
                         "(only when the team agrees to move to a new data version)")
    args = ap.parse_args()

    files = nhanes_files()
    checksums = CHECKSUMS.relative_to(REPO_ROOT)
    failures = [] if args.verify_only else fetch(files, args.out, args.force, args.workers)
    if args.write_checksums:
        print(f"Wrote {write_checksums(files, args.out)} checksums -> {checksums}")
        problems = []
    else:
        problems = verify(files, args.out)
        print(f"Verified {len(files) - len(problems)}/{len(files)} files against {checksums}")

    for line in failures + problems:
        print(f"  ! {line}")
    if any(p.startswith("differs") for p in problems):
        print("Some files differ from the team's copy: the source may have been updated. "
              "Re-run with --force; if they still differ, tell the team.")
    return 1 if failures or problems else 0


if __name__ == "__main__":
    sys.exit(main())
