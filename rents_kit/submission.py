"""Assemble and write rents_<team-name>.csv (design.md sec. 8.8; T1.1 to T1.6)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def team_slug(name):
    s = re.sub(r"\s+", "-", str(name).strip().lower())
    s = re.sub(r"[^a-z0-9-]", "", s)
    if not s:
        raise SystemExit("Team name is empty after cleaning - set config.TEAM_NAME or --team")
    return s


def _round(x, step):
    return np.round(np.asarray(x, float) / step) * step


def assemble(flats_raw, ids, rent_with, rent_without, team, outdir, train_rents):
    lo, hi = np.percentile(np.asarray(train_rents, float), C.CLIP_PCT)
    rw = np.clip(np.asarray(rent_with, float), lo, hi)
    rwo = np.clip(np.asarray(rent_without, float), lo, hi)
    med = np.nanmedian(rwo)
    rwo = np.where(np.isfinite(rwo), rwo, med)
    rw = np.where(np.isfinite(rw), rw, rwo)
    rw = np.fmax(rw, rwo)
    rw, rwo = _round(rw, C.ROUND_TO), _round(rwo, C.ROUND_TO)
    rw = np.fmax(rw, rwo)
    sub = pd.DataFrame({"gnaf_id": ids, "rent_with_car_space": rw, "rent_without_car_space": rwo})
    order = flats_raw["gnaf_id"].astype(str).str.strip()
    sub = sub.set_index("gnaf_id").reindex(order).reset_index()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"rents_{team_slug(team)}.csv"
    sub.to_csv(path, index=False, encoding="utf-8", float_format="%.1f")
    from validate_submission import validate
    validate(path, flats_raw, raise_on_fail=True)
    return path
