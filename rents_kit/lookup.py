"""Demo lookup (US13, T13.1 to T13.4). Offline: reads the input flats file and the final CSV only.
  python lookup.py "unit 506 20 dunkerley place"
  python lookup.py "506/20 dunkerley pl" --asking 850 --parking no"""
import argparse
import difflib
import re
import sys
from pathlib import Path

import pandas as pd

import config as C


def norm(s):
    s = str(s).upper()
    s = re.sub(r"^\s*([A-Z]?\d+[A-Z]?)\s*/\s*", r"UNIT \1 ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\b(NSW|UNIT|APARTMENT|APT)\b", " ", s)
    s = re.sub(r"\bST\b", "STREET", s)
    s = re.sub(r"\bPL\b", "PLACE", s)
    s = re.sub(r"\bRD\b", "ROAD", s)
    s = re.sub(r"\bAVE?\b", "AVENUE", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("address")
    ap.add_argument("--asking", type=float)
    ap.add_argument("--parking", choices=["yes", "no"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--flats", default=str(C.DATA_DIR / "rents-input-flats-city-of-sydney.csv"))
    a = ap.parse_args()
    csv = Path(a.csv) if a.csv else next(iter(sorted(C.OUT_DIR.glob("rents_*.csv"))), None)
    if csv is None or not Path(a.flats).exists():
        sys.exit("Need the final CSV (run.py final) and the flats file (--flats).")
    f = pd.read_csv(a.flats, dtype=str, usecols=["gnaf_id", "address", "suburb", "bedrooms", "car_space"])
    r = pd.read_csv(csv, dtype={"gnaf_id": str}).set_index("gnaf_id")
    q = norm(a.address)
    qt = set(q.split())
    f["n"] = f["address"].map(norm)
    f["tok"] = f["n"].map(lambda s: len(qt & set(s.split())))
    cand = f[f["tok"] == f["tok"].max()].copy()
    cand["score"] = cand["n"].map(lambda s: difflib.SequenceMatcher(None, q, s).ratio())
    top = cand.sort_values("score", ascending=False).head(3)
    for i, (_, row) in enumerate(top.iterrows()):
        w, wo = r.loc[row.gnaf_id, "rent_with_car_space"], r.loc[row.gnaf_id, "rent_without_car_space"]
        beds = row.bedrooms if pd.notna(row.bedrooms) else "?"
        print(f"{'>>' if i == 0 else '  '} {row.address}  ({row.suburb}, {beds} bed)")
        print(f"   with a car space ${w:,.0f}/wk | without ${wo:,.0f}/wk | parking premium ${w - wo:,.0f}/wk")
        if i == 0 and a.asking:
            park = a.parking or ("yes" if row.car_space == "1.0" or row.car_space == "1" else "no")
            est = w if park == "yes" else wo
            gap = a.asking - est
            label = "LOOKS HIGH" if gap > C.LOOKUP_BAND else "LOOKS LOW" if gap < -C.LOOKUP_BAND else "IN LINE"
            print(f"   asking ${a.asking:,.0f} vs our ${est:,.0f} ({'with' if park == 'yes' else 'without'} parking): "
                  f"{gap:+,.0f} ({gap / est:+.0%})  ->  {label}")


if __name__ == "__main__":
    main()
