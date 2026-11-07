"""REHEARSAL ONLY: score a submission against the synthetic frozen file, the way the judges will (T3.2 rules).
Never point this at real judging data. Usage:
  python score_private.py out/rents_x.csv ../data_synth/rents-input-flats-city-of-sydney.csv ../data_synth/private/synthetic_frozen.csv"""
import sys

import numpy as np
import pandas as pd

sub, flats, frozen = sys.argv[1:4]
s = pd.read_csv(sub, dtype={"gnaf_id": str}).set_index("gnaf_id")
f = pd.read_csv(flats, dtype={"gnaf_id": str}).set_index("gnaf_id")
z = pd.read_csv(frozen, dtype={"gnaf_pid": str})
cs = z["car_spaces"].gt(0).astype(float).where(z["car_spaces"].notna(), z["gnaf_pid"].map(f["car_space"]))
z = z[cs.notna()]
cs = cs[cs.notna()]
pred = np.where(cs == 1, z["gnaf_pid"].map(s["rent_with_car_space"]), z["gnaf_pid"].map(s["rent_without_car_space"]))
e = np.abs(pred - z["weekly_rent"].to_numpy())
print(f"rehearsal score on {len(z):,} frozen adverts: median abs error ${np.nanmedian(e):.1f}, "
      f"within $100 {np.nanmean(e <= 100):.1%}, blanks {int(np.isnan(pred).sum())}")
