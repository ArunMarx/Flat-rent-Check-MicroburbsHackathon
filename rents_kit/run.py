"""Command line (design.md sec. 8.11). Windows PowerShell examples:
  python run.py inspect   --data ..\\data
  python run.py calibrate --data ..\\data
  python run.py eval      --data ..\\data [--sources merged] [--confirm] [--late-api drop] [--no-views]
  python run.py final     --data ..\\data --team "My Team" [--sources merged]
  python run.py report    --data ..\\data
  python run.py test      [--data ..\\data_synth]
--data is the folder holding the three supplied files; individual files can be given with --flats/--leases/--blocks.
--col key="Their Column" maps an unrecognised lease column (keys: see config.LEASE_ALIASES)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import guard
from io_files import load_blocks, load_flats, load_leases, log, match_leases, make_campaigns

FLATS = "rents-input-flats-city-of-sydney.csv"
LEASES = "rents-past-leases-city-of-sydney.csv"
BLOCKS = "blocks-homes-and-people-2021-2026.csv"


def paths(a):
    d = Path(a.data) if a.data else C.DATA_DIR
    return (Path(a.flats) if a.flats else d / FLATS, Path(a.leases) if a.leases else d / LEASES,
            Path(a.blocks) if a.blocks else d / BLOCKS)


def overrides(a):
    out = {}
    for kv in a.col or []:
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def load_all(a):
    from features import prep_flats
    fp, lp, bp = paths(a)
    flats_raw = load_flats(fp)
    flats, masked = guard.mask_flats(flats_raw, C.CUTOFF)
    blocks = load_blocks(bp)
    d = prep_flats(flats, blocks)
    sup, linfo = load_leases(lp, overrides(a))
    sup = guard.admit(sup, "advert_date", C.CUTOFF, "supplied")
    return flats_raw, d, sup, {"masked": masked, "lease": linfo, "guard_drops": dict(guard.DROPS)}


def cmd_inspect(a):
    fp, lp, bp = paths(a)
    flats_raw = load_flats(fp)
    print(f"\nFLATS {fp.name}: {len(flats_raw):,} rows, {flats_raw['suburb'].nunique()} suburbs")
    for c in ("bedrooms", "car_space", "floor_level", "block_homes_2026", "first_ever_listed", "attrs_last_seen"):
        print(f"  blank {c:<22} {flats_raw[c].isna().mean():6.1%}")
    big = int((flats_raw["bedrooms"] > C.BED_FLAG_ABOVE).sum())
    print(f"  bedrooms above {C.BED_FLAG_ABOVE}: {big} (grouped as 4+ for group statistics, T7.6)")
    _, masked = guard.mask_flats(flats_raw, C.CUTOFF)
    print(f"  input dates after {C.CUTOFF} (masked): {masked}")
    miss = flats_raw[flats_raw["block_homes_2026"].isna()]
    print(f"  flats with no block data (missing from City files): {len(miss):,} in suburbs "
          f"{miss['suburb'].value_counts().head(6).to_dict()}")
    raw = pd.read_csv(guard.check_path(lp), dtype=str, nrows=5)
    print(f"\nLEASES {lp.name}: columns {list(raw.columns)}")
    L, info = load_leases(lp, overrides(a))
    late = int((L["advert_date"] > pd.Timestamp(C.CUTOFF)).sum())
    print(f"  date range {L['advert_date'].min().date()} .. {L['advert_date'].max().date()}  ({info['date_format']}); "
          f"rows after cutoff (will be dropped): {late:,}")
    pm = L.set_index("advert_date").resample("MS")["rent"].size()
    print(f"  adverts per month (last 6): {pm.tail(6).to_dict()}")
    print(f"  median weekly rent ${L['rent'].median():.0f}; with bedrooms {L['bedrooms'].notna().mean():.0%}; "
          f"with parking {L['car_space'].notna().mean():.0%}")
    M, counts = match_leases(L, flats_raw.assign(suburb=flats_raw["suburb"]))
    print(f"  match methods (T11.5): {counts}")
    if L["gnaf_id"].notna().any():
        print(f"  missing-file flats with a lease row: {miss['gnaf_id'].isin(set(L['gnaf_id'])).mean():.0%} (T7.7 gate)")
    b = load_blocks(bp)
    print(f"\nBLOCKS {bp.name}: {'not found' if b is None else f'{len(b):,} rows, columns {list(b.columns)[:8]}'}\n")


def get_adverts(a, d, sup):
    from pipeline import curate
    if a.cache:
        C.CACHE_DIR = Path(a.cache)
    return curate(d, sup, a.sources, a.late_api)


def cmd_calibrate(a):
    from pipeline import attach_attributes, calibrate
    flats_raw, d, sup, _ = load_all(a)
    L, _ = match_leases(sup, d)
    L, _ = make_campaigns(L)
    A = attach_attributes(L, d)
    m, checks = calibrate(A, d)
    print(f"\nCALIBRATION vs sample report (T3.5): MdAE ${m['mdae']:.1f} (ref 100), within $100 {m['within100']:.1%} "
          f"(ref 56.8%), R2 {m['r2']:.3f} (ref 0.645), n {m['n']:,} (ref 3,954)")
    print("  " + "  ".join(f"{k}: {'PASS' if v else 'GAP'}" for k, v in checks.items()) + "\n")
    from pipeline import append_result
    append_result("sample method (calibration)", "sample window", m)
    return checks


def cmd_eval(a):
    from pipeline import print_validation, run_validation
    flats_raw, d, sup, _ = load_all(a)
    A, info = get_adverts(a, d, sup)
    log(f"adverts: {info}")
    tr, te = (C.CONFIRM_TRAIN_END, C.CONFIRM_TEST) if a.confirm else (C.TUNE_TRAIN_END, C.TUNE_TEST)
    label = ("confirm" if a.confirm else "tune") + f"/{a.sources}" + ("" if a.views else "/noviews") + \
            (f"/lateapi-{a.late_api or C.LATE_API}" if a.sources == "merged" else "") + f"/{C.INDEX_END_MODE}"
    res, te_df, fit, F = run_validation(d, A, tr, te, label, use_views=a.views)
    print_validation(res)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.OUT_DIR / f"eval_{label.replace('/', '_')}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def cmd_final(a):
    from pipeline import run_final
    flats_raw, d, sup, info = load_all(a)
    A, ainfo = get_adverts(a, d, sup)
    path, fit, F, rw, rwo = run_final(d, flats_raw, A, a.team or C.TEAM_NAME, blend_weight=a.blend)
    meta = {"sources": a.sources, "adverts": ainfo, "inputs": info, "index": fit.idx.kind,
            "index_pairs": fit.idx.n_pairs, "n_train": fit.n_train, "csv": str(path),
            "premium_median": float(np.median(rw - rwo)), "guard_drops": dict(guard.DROPS),
            "premium_log_by_bed": {str(k): v for k, v in fit.premium.items()},
            "index_end": C.INDEX_END_MODE, "blend": a.blend if a.blend is not None else C.BLEND_WEIGHT,
            "anchor_src": {k: int(v) for k, v in F["anchor_src"].value_counts().sort_index().items()}}
    fit.idx.table().to_csv(C.OUT_DIR / "index.csv", index=False)
    (C.OUT_DIR / "final_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return path


def cmd_report(a):
    from report import make_report
    flats_raw, d, sup, info = load_all(a)
    A, ainfo = get_adverts(a, d, sup)
    make_report(d, A, info, ainfo)


def cmd_test(a):
    import subprocess
    data = a.data or str(C.ROOT.parent / "data_synth")
    r = subprocess.run([sys.executable, str(C.ROOT / "tests" / "run_tests.py"), data])
    sys.exit(r.returncode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inspect", "calibrate", "eval", "final", "report", "test"])
    ap.add_argument("--data")
    ap.add_argument("--flats")
    ap.add_argument("--leases")
    ap.add_argument("--blocks")
    ap.add_argument("--cache", help="API cache folder (default ../cache/api)")
    ap.add_argument("--sources", choices=["supplied", "merged"], default="supplied")
    ap.add_argument("--late-api", choices=["keep", "drop"], default=None)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--no-views", dest="views", action="store_false")
    ap.add_argument("--index-end", choices=["flat", "trend"], default=None, help="T9.3 look-ahead mode")
    ap.add_argument("--team")
    ap.add_argument("--blend", type=float, default=None, help="blend weight from eval (1.0 = model only)")
    ap.add_argument("--col", action="append", help='map a lease column, e.g. --col rent="Weekly Rent"')
    a = ap.parse_args()
    t0 = time.time()
    if a.index_end:
        C.INDEX_END_MODE = a.index_end
    {"inspect": cmd_inspect, "calibrate": cmd_calibrate, "eval": cmd_eval, "final": cmd_final,
     "report": cmd_report, "test": cmd_test}[a.cmd](a)
    log(f"done in {time.time() - t0:.0f}s")
