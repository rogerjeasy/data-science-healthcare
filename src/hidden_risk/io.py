"""Read NHANES SAS transport (.xpt) files into clean pandas DataFrames."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_RAW

# SAS transport stores 0 as about 5.4e-79; anything this small is a stored zero.
TINY = 1e-30


def clean_xpt_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Turn stored zeros back into 0, make SEQN an integer and strip text columns."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].mask(df[col].abs() < TINY, 0.0)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()
    if "SEQN" in df.columns:
        df["SEQN"] = df["SEQN"].astype("int64")
    return df


def read_xpt(name: str, raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Read one NHANES file by name, e.g. read_xpt("DEMO_D")."""
    path = raw_dir / "nhanes" / f"{name}.xpt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/fetch_data.py` first.")
    return clean_xpt_frame(pd.read_sas(path, format="xport", encoding="latin-1"))
