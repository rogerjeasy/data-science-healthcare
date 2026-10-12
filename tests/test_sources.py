import hashlib
from pathlib import Path

import pytest

from hidden_risk import build
from hidden_risk.config import CYCLES
from hidden_risk.sources import nhanes_files, verify, write_checksums
from hidden_risk.variables import VARIABLES

CONTENT = b"HEADER RECORD*******LIBRARY HEADER RECORD"


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    """A raw-data folder with one file, and its checksum recorded next to it."""
    folder = tmp_path / "raw"
    (folder / "nhanes").mkdir(parents=True)
    (folder / "nhanes" / "DPQ_D.xpt").write_bytes(CONTENT)
    (tmp_path / "checksums.sha256").write_text(
        f"{hashlib.sha256(CONTENT).hexdigest()}  nhanes/DPQ_D.xpt\n")
    return folder


def test_file_matching_its_recorded_checksum_is_accepted(raw):
    assert verify(["nhanes/DPQ_D.xpt"], raw, raw.parent / "checksums.sha256") == []


def test_altered_file_is_reported(raw):
    (raw / "nhanes" / "DPQ_D.xpt").write_bytes(CONTENT + b"!")
    assert verify(["nhanes/DPQ_D.xpt"], raw, raw.parent / "checksums.sha256") == [
        "differs from the team's copy: nhanes/DPQ_D.xpt"]


def test_missing_file_and_file_without_recorded_checksum_are_reported(raw):
    (raw / "nhanes" / "HUQ_D.xpt").write_bytes(CONTENT)
    problems = verify(["nhanes/DEMO_D.xpt", "nhanes/HUQ_D.xpt"], raw, raw.parent / "checksums.sha256")
    assert problems == ["missing: nhanes/DEMO_D.xpt", "no recorded checksum: nhanes/HUQ_D.xpt"]


def test_recorded_checksums_replace_the_old_list_and_detect_later_changes(raw):
    checksums = raw.parent / "checksums.sha256"
    checksums.write_text("0" * 64 + "  bfs/no_longer_used.xlsx\n")
    write_checksums(["nhanes/DPQ_D.xpt"], raw, checksums)
    assert checksums.read_text() == f"{hashlib.sha256(CONTENT).hexdigest()}  nhanes/DPQ_D.xpt\n"
    (raw / "nhanes" / "DPQ_D.xpt").write_bytes(b"HEADER changed")
    assert verify(["nhanes/DPQ_D.xpt"], raw, checksums) == [
        "differs from the team's copy: nhanes/DPQ_D.xpt"]


def test_every_file_the_build_reads_is_in_the_download_list():
    needed = {f"nhanes/{v.module}_{cycle}.xpt" for v in VARIABLES for cycle in CYCLES}
    needed |= {"nhanes/DEMO_L.xpt", "nhanes/DPQ_L.xpt"}  # 2021-23 trend file
    assert needed <= set(nhanes_files())


def test_download_urls_point_to_the_cycle_folder_of_each_file():
    files = nhanes_files()
    base = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
    assert files["nhanes/DPQ_D.xpt"] == f"{base}/2005/DataFiles/DPQ_D.xpt"
    assert files["nhanes/RXQ_RX_J.xpt"] == f"{base}/2017/DataFiles/RXQ_RX_J.xpt"
    assert files["nhanes/DPQ_L.xpt"] == f"{base}/2021/DataFiles/DPQ_L.xpt"
    assert files["nhanes/RXQ_DRUG.xpt"] == f"{base}/1988/DataFiles/RXQ_DRUG.xpt"


def test_build_refuses_raw_data_that_is_missing_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "processed"
    status = build.main(["--raw", str(tmp_path / "empty"), "--out", str(out)])
    assert status == 1
    assert not list(out.glob("*.parquet"))
    assert "missing: nhanes/DEMO_D.xpt" in capsys.readouterr().out
