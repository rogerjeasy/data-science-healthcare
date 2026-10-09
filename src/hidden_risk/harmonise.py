"""Merge the NHANES modules of one cycle on SEQN and harmonise them across cycles."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CYCLE_LABELS, DATA_RAW
from .io import read_xpt
from .variables import VARIABLES, Var

# Modules with one row per person; RXQ_RX has one row per drug and is handled separately
PERSON_MODULES = sorted({v.module for v in VARIABLES} - {"DEMO", "RXQ_RX"})
RX_CODES = {"55555": "unknown", "77777": "refused", "99999": "don't know"}


def load_cycle(cycle: str, raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    """All raw columns of one cycle, one row per person in DEMO, merged on SEQN."""
    raw = read_xpt(f"DEMO_{cycle}", raw_dir)
    for module in PERSON_MODULES:
        df = read_xpt(f"{module}_{cycle}", raw_dir)
        if not df["SEQN"].is_unique:
            raise ValueError(f"{module}_{cycle}: SEQN is not unique")
        raw = raw.merge(df, on="SEQN", how="left", validate="one_to_one")
    rx = read_xpt(f"RXQ_RX_{cycle}", raw_dir)
    rx_person = rx.groupby("SEQN", as_index=False)[["RXDUSE", "RXDCOUNT"]].first()
    return raw.merge(rx_person, on="SEQN", how="left", validate="one_to_one")


def harmonise_variable(var: Var, raw: pd.DataFrame, cycle: str) -> pd.Series:
    """One harmonised column: missing codes removed, cross-cycle transform, age coverage."""
    source = var.source_in(cycle)
    if source is None:  # not collected in this cycle
        values = pd.Series(float("nan"), index=raw.index)
    elif source not in raw.columns:
        raise KeyError(f"{var.name}: {source} not found in cycle {cycle}")
    else:
        values = raw[source].astype("float64")
    values = values.mask(values.isin(var.missing))
    if var.transform is not None:
        values = var.transform(values, cycle, raw)
    if var.min_age is not None:
        values = values.where(raw["RIDAGEYR"] >= var.min_age)
    if var.max_age is not None:
        values = values.where(raw["RIDAGEYR"] <= var.max_age)
    return values.astype(var.dtype)


def harmonise_cycle(raw: pd.DataFrame, cycle: str, variables=VARIABLES) -> pd.DataFrame:
    """Harmonised person-level table for one cycle."""
    out = pd.DataFrame({"seqn": raw["SEQN"].astype("int64"), "cycle": CYCLE_LABELS[cycle]})
    for var in variables:
        out[var.name] = harmonise_variable(var, raw, cycle)
    return out


def load_prescriptions(cycle: str, raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Long table, one row per reported prescription drug (people without drugs dropped)."""
    rx = read_xpt(f"RXQ_RX_{cycle}", raw_dir)
    rx = rx[rx["RXDDRUG"].fillna("") != ""]
    days = rx["RXDDAYS"].astype("float64")
    return pd.DataFrame({
        "seqn": rx["SEQN"].astype("int64"),
        "cycle": CYCLE_LABELS[cycle],
        "drug_name": rx["RXDDRUG"].mask(rx["RXDDRUG"].isin(list(RX_CODES))).str.upper(),
        "drug_id": rx["RXDDRGID"].replace("", pd.NA),
        "days_taken": days.mask(days.isin([77777, 99999])),
    }).reset_index(drop=True)
