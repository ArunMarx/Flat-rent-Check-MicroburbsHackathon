"""Standalone checker for rents_<team-name>.csv (T1.1 to T1.6).
Usage:  python validate_submission.py out/rents_my-team.csv ../data/rents-input-flats-city-of-sydney.csv"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def validate(path, flats, raise_on_fail=False):
    path = Path(path)
    if isinstance(flats, (str, Path)):
        flats = pd.read_csv(flats, dtype=str)
    fails = []
    if not re.fullmatch(r"rents_[a-z0-9-]+\.csv", path.name):
        fails.append(f"T1.6 file name {path.name!r} must be rents_<lower-case-hyphenated-team>.csv")
    raw = path.read_bytes()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        fails.append("T1.6 file is not UTF-8")
    s = pd.read_csv(path, dtype={"gnaf_id": str})
    if list(s.columns) != ["gnaf_id", "rent_with_car_space", "rent_without_car_space"]:
        fails.append(f"T1.2 columns are {list(s.columns)}")
    ids_in = set(flats["gnaf_id"].astype(str).str.strip())
    ids_out = s["gnaf_id"].astype(str)
    if ids_out.duplicated().any():
        fails.append(f"T1.1 {int(ids_out.duplicated().sum())} duplicate ids")
    if set(ids_out) != ids_in or len(s) != len(ids_in):
        fails.append(f"T1.1 ids differ: {len(ids_in - set(ids_out))} missing, {len(set(ids_out) - ids_in)} extra")
    for c in ("rent_with_car_space", "rent_without_car_space"):
        if c in s.columns:
            v = pd.to_numeric(s[c], errors="coerce")
            if v.isna().any() or (v <= 0).any():
                fails.append(f"T1.3 {c}: {int(v.isna().sum())} blank, {int((v <= 0).sum())} zero/negative")
            if (np.abs(v * 2 - np.round(v * 2)) > 1e-9).any():
                fails.append(f"T1.4 {c} has values that are not whole dollars or .50")
    if {"rent_with_car_space", "rent_without_car_space"} <= set(s.columns):
        bad = (s["rent_with_car_space"] < s["rent_without_car_space"]).sum()
        if bad:
            fails.append(f"T1.5 {int(bad)} rows have with < without")
    if fails:
        msg = "SUBMISSION CHECK FAILED:\n  " + "\n  ".join(fails)
        if raise_on_fail:
            raise SystemExit(msg)
        print(msg)
        return False
    print(f"format OK: {path.name}, {len(s):,} rows, median with ${s['rent_with_car_space'].median():.0f}, "
          f"without ${s['rent_without_car_space'].median():.0f}")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(0 if validate(sys.argv[1], sys.argv[2]) else 1)
