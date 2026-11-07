"""The single cutoff guard (design.md sec. 9). Every dated row enters the pipeline through admit().
T2.1, T2.2, T2.3, T2.5."""
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

DROPS = {}          # source -> rows dropped for being after the cutoff (reported)


def check_path(path):
    """T2.3: refuse anything that looks like the frozen judging file."""
    if path is not None and "frozen" in Path(str(path)).name.lower():
        raise SystemExit(f"Refusing to read {path}: files named 'frozen' hold judging data and must never be used.")
    return path


def admit(df, date_col, cutoff, source):
    """Drop rows dated after `cutoff` (or undated); assert nothing later survives."""
    cut = pd.Timestamp(cutoff)
    d = pd.to_datetime(df[date_col], errors="coerce")
    keep = d.notna() & (d <= cut)
    DROPS[source] = DROPS.get(source, 0) + int((~keep).sum())
    out = df.loc[keep].copy()
    out[date_col] = d[keep]
    if len(out):
        assert out[date_col].max() <= cut, "guard failed: a row after the cutoff survived"
    return out.reset_index(drop=True)


def strip(df):
    """Remove columns that carry post-advert outcomes (T2.5)."""
    return df.drop(columns=[c for c in df.columns if c.lower() in C.LEASE_DROP], errors="ignore")


def mask_flats(flats, cutoff):
    """T2.2: null input dates after the cutoff, and the advertised flag they imply."""
    f = flats.copy()
    cut = pd.Timestamp(cutoff)
    counts = {}
    for col in ("first_ever_listed", "attrs_last_seen"):
        if col in f.columns:
            d = pd.to_datetime(f[col], errors="coerce")
            d = d.where(d.dt.year > 1900)                     # 0001-01-01 sentinel -> unknown
            late = d > cut
            counts[col] = int(late.sum())
            if col == "first_ever_listed" and "ever_advertised_for_lease" in f.columns:
                f.loc[late, "ever_advertised_for_lease"] = np.nan
            f[col] = d.where(~late)
    for col in ("first_ever_listed", "attrs_last_seen"):
        if col in f.columns and f[col].notna().any():
            assert f[col].max() <= cut
    return f, counts
