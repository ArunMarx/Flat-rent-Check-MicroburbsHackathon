"""Import the team's public layers (G-NAF extras + LiDAR views) into two slim files the pipeline reads.
Source: the microburbs-hackathon-2026 repo, folder public-data/_derived (config.PUBLIC_SRC).
Writes config.PUBLIC_DIR/flat_public_features.csv (one row per input flat, keyed gnaf_id),
       config.PUBLIC_DIR/floor_view.csv (one row per OSM building x floor) and manifest.json.
No date column is carried across: every value is a static fact about the building (T2.x, see guard.check_public).
Usage:  python import_public.py [--src <folder>] [--crosscheck ..\\data\\rents-input-flats-city-of-sydney.csv]
--crosscheck prints aggregate agreement between the G-NAF floor and the organiser's floor_level (no rows are shown)."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import guard

GNAF_MAP = {                                   # source column -> pipeline column
    "floor_best": "gnaf_floor", "floor_rel": "gnaf_floor_rel", "is_ground_floor": "gnaf_is_ground",
    "is_top_floor": "gnaf_is_top", "is_penthouse": "gnaf_is_penthouse", "units_on_floor": "gnaf_units_on_floor",
    "bld_n_residential_units": "gnaf_bld_units", "bld_n_nonres_units": "gnaf_bld_nonres_units",
    "bld_max_floor": "gnaf_bld_max_floor", "bld_n_floors_seen": "gnaf_bld_floors_known",
    "sp_year_best": "gnaf_sp_year", "building_age_years": "gnaf_bld_age", "bld_footprint_dist_m": "pb_join_dist",
}
BLD_MAP = {
    "h_p90_2020": "pb_h_2020", "levels_from_lidar_2020": "pb_levels_lidar",
    "new_2013_2020": "pb_new_build", "dist_station_m": "pb_dist_station", "dist_water_m": "pb_dist_water",
    "canopy_mmb_pct": "pb_canopy_pct", "lst_c": "pb_lst_c", "footprint_m2": "pb_footprint_m2",
    "under_16_34_path_1km": "pb_air_path",
    "rwy_16_34_min_lateral_m": "pb_rwy_lateral", "abs_sa2_median_rent_2021": "pb_abs_rent_2021",
}
VIEW_MAP = {"roof_margin_m": "vw_roof_margin", "roofline_clear_frac": "vw_roofline_clear",
            "water_view_frac": "vw_water", "park_view_frac": "vw_park"}
FLOOR_SRC = {"gnaf_level": 2.0, "flat_number": 1.0, "none": 0.0}
DATE_CHECKS = {}
# Inside Airbnb (branch airbnb-features of the team repo; removed from its main). Snapshot scraped 16-17 Sep 2026;
# listings kept only when their first review is on or before the cutoff (and still online in September). Counts of
# such listings are facts about before the cutoff; "active" counts, prices, revenue, review scores, host counts and STRA
# licences are September values (see public_features.AB_SEPT_COLS, off unless config.AIRBNB_SEPT_VALUES).
AB_BLD_MAP = {"n_listings_300m": "ab_n_300m", "n_entire_home_300m": "ab_n_entire_300m",
              "n_entire_home_active_300m": "ab_n_active_300m", "n_stra_registered_300m": "ab_n_stra_300m",
              "n_commercial_host_300m": "ab_n_commercial_300m", "median_price_night_entire_300m": "ab_price_night_300m",
              "mean_review_location_300m": "ab_loc_score_300m"}
AB_SUB_MAP = {"n_listings": "ab_sub_n", "n_entire_home_active": "ab_sub_n_active", "median_months_listed": "ab_sub_months_listed",
              "share_stra_registered": "ab_sub_share_stra", "median_price_night_entire": "ab_sub_price_night",
              "median_est_revenue_entire": "ab_sub_revenue", "mean_review_location": "ab_sub_loc_score"}
AB_REF = "airbnb-features"


def _num(s):
    """Numbers, with True/False strings mapped to 1/0."""
    return pd.to_numeric(s.replace({"True": 1, "False": 0, "true": 1, "false": 0}), errors="coerce")


def git_csv(repo, ref, path):
    """Read a CSV that lives only on another branch of the team repo (git show ref:path)."""
    import io
    r = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise SystemExit(f"cannot read {ref}:{path} from {repo}: {r.stderr.strip()}")
    return pd.read_csv(io.StringIO(r.stdout), dtype=str, keep_default_na=False, na_values=[""])


def airbnb(src, ref=AB_REF):
    """(building table keyed osm_id, suburb table keyed SUBURB upper-case) from the airbnb-features branch."""
    repo = Path(src).resolve().parents[1]
    ab = git_csv(repo, ref, "public-data/_derived/airbnb_building_features.csv")
    sb = git_csv(repo, ref, "public-data/_derived/airbnb_suburb_features.csv")
    abb = pd.DataFrame({"osm_id": pd.to_numeric(ab["osm_id"], errors="coerce").astype("Int64")})
    for s, t in AB_BLD_MAP.items():
        abb[t] = _num(ab[s])
    assert not abb["osm_id"].duplicated().any(), "osm_id not unique in airbnb_building_features"
    sub = pd.DataFrame({"suburb": sb["suburb"].str.upper().str.strip()})
    for s, t in AB_SUB_MAP.items():
        sub[t] = _num(sb[s])
    return abb, sub


def build(src, with_airbnb=True):
    src = Path(src)
    g = pd.read_csv(src / "gnaf_input_flats.csv", dtype=str, keep_default_na=False, na_values=[""], low_memory=False)
    b = pd.read_csv(src / "buildings_features_v2.csv", dtype=str, keep_default_na=False, na_values=[""], low_memory=False)
    fv = pd.read_csv(src / "lidar_floor_view.csv", dtype=str, keep_default_na=False, na_values=[""], low_memory=False)

    flat = pd.DataFrame({"gnaf_id": g["gnaf_id"].str.strip()})
    flat["osm_id"] = pd.to_numeric(g["bld_osm_id"], errors="coerce").round().astype("Int64")   # '62203262.0' -> 62203262
    for s, t in GNAF_MAP.items():
        flat[t] = _num(g[s])
    flat["gnaf_floor_src"] = g["floor_source"].map(FLOOR_SRC)
    # source date checks (T2.x): nothing created after the cutoff may be present; addresses edited after the cutoff
    # lose their per-flat G-NAF fields (the edit may have touched the flat number or level)
    created = pd.to_datetime(g["address_created"], errors="coerce")
    modified = pd.to_datetime(g["address_modified"], errors="coerce")
    assert not (created > pd.Timestamp(C.CUTOFF)).any(), "G-NAF address created after the cutoff"
    late_mod = (modified > pd.Timestamp(C.CUTOFF)).to_numpy()
    per_flat = ["gnaf_floor", "gnaf_floor_rel", "gnaf_is_ground", "gnaf_is_top", "gnaf_is_penthouse", "gnaf_units_on_floor"]
    flat.loc[late_mod, per_flat] = np.nan
    flat.loc[late_mod, "gnaf_floor_src"] = np.nan
    DATE_CHECKS["gnaf_addresses_modified_after_cutoff_nulled"] = int(late_mod.sum())
    DATE_CHECKS["gnaf_latest_address_created"] = str(created.max().date())
    # strata plans registered in 2026 may post-date the cutoff (some were registered Aug-Sep 2026): treat the year as unknown
    late_sp = flat["gnaf_sp_year"] >= 2026
    flat.loc[late_sp, ["gnaf_sp_year", "gnaf_bld_age"]] = np.nan
    DATE_CHECKS["strata_year_2026_nulled"] = int(late_sp.sum())
    assert not (flat["gnaf_sp_year"] >= 2026).any()
    flat["gnaf_nonres"] = (g["dwelling_kind"] == "non_residential_unit").astype(float)
    flat["gnaf_whole_building"] = (g["dwelling_kind"] == "no_flat_number").astype(float)

    osm_ts = pd.to_datetime(b["osm_timestamp_base"], errors="coerce", utc=True).dt.tz_localize(None)
    DATE_CHECKS["osm_snapshot"] = str(osm_ts.max().date())
    DATE_CHECKS["osm_note"] = ("OSM pulled after the cutoff: tag columns (building:levels, neighbour levels) are NOT imported; "
                               "footprints and distances to stations/water are kept as physical facts; LiDAR columns are 2013/2020 surveys")
    bb = pd.DataFrame({"osm_id": pd.to_numeric(b["osm_id"], errors="coerce").astype("Int64")})
    for s, t in BLD_MAP.items():
        bb[t] = _num(b[s])
    assert not bb["osm_id"].duplicated().any(), "osm_id is not unique in buildings_features_v2"
    flat = flat.merge(bb, on="osm_id", how="left")

    view = pd.DataFrame({"osm_id": pd.to_numeric(fv["osm_id"], errors="coerce").astype("Int64"),
                         "floor": pd.to_numeric(fv["floor"], errors="coerce").astype("Int64")})
    for s, t in VIEW_MAP.items():
        view[t] = _num(fv[s])
    view = view.dropna(subset=["osm_id", "floor"])
    assert not view.duplicated(["osm_id", "floor"]).any(), "duplicate building x floor rows in lidar_floor_view"

    sub = None
    if with_airbnb:
        abb, sub = airbnb(src)
        flat = flat.merge(abb, on="osm_id", how="left")
        guard.check_public(sub, "airbnb_suburb")

    # checks the pipeline relies on
    assert flat["gnaf_id"].is_unique, "duplicate gnaf_id in gnaf_input_flats"
    assert len(flat) == 65_777, f"expected 65,777 input flats, got {len(flat):,}"
    joined = flat["osm_id"].notna() & flat["pb_footprint_m2"].notna()
    assert joined.sum() == flat["osm_id"].notna().sum(), "some bld_osm_id has no row in buildings_features_v2"
    guard.check_public(flat, "flat_public_features")
    guard.check_public(view, "floor_view")
    return flat, view, sub


def crosscheck(flat, flats_path):
    """Aggregate agreement of the G-NAF floor with the organiser's floor_level. Prints counts only."""
    f = pd.read_csv(guard.check_path(flats_path), dtype=str, usecols=["gnaf_id", "floor_level"])
    f["floor_level"] = pd.to_numeric(f["floor_level"], errors="coerce")
    m = f.merge(flat[["gnaf_id", "gnaf_floor", "gnaf_floor_src"]], on="gnaf_id", how="left")
    both = m["floor_level"].notna() & m["gnaf_floor"].notna()
    diff = (m.loc[both, "gnaf_floor"] - m.loc[both, "floor_level"]).abs()
    out = {"flats": int(len(m)), "organiser_floor_known": int(m["floor_level"].notna().sum()),
           "gnaf_floor_known": int(m["gnaf_floor"].notna().sum()), "both_known": int(both.sum()),
           "exact_agree_share": round(float((diff == 0).mean()), 4) if both.any() else None,
           "within_1_share": round(float((diff <= 1).mean()), 4) if both.any() else None,
           "median_abs_diff": float(diff.median()) if both.any() else None,
           "blanks_filled_by_gnaf": int((m["floor_level"].isna() & m["gnaf_floor"].notna()).sum()),
           "floor_known_after_fill": int((m["floor_level"].notna() | m["gnaf_floor"].notna()).sum())}
    for src, name in ((1.0, "flat_number"), (2.0, "gnaf_level")):
        k = both & (m["gnaf_floor_src"] == src)
        if k.any():
            dd = (m.loc[k, "gnaf_floor"] - m.loc[k, "floor_level"]).abs()
            out[f"exact_agree_share_{name}"] = round(float((dd == 0).mean()), 4)
            out[f"n_{name}"] = int(k.sum())
    return out


def main(src, out_dir, flats_path=None, with_airbnb=True):
    t0 = time.time()
    flat, view, sub = build(src, with_airbnb)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat.to_csv(out_dir / "flat_public_features.csv", index=False)
    view.to_csv(out_dir / "floor_view.csv", index=False)
    if sub is not None:
        sub.to_csv(out_dir / "airbnb_suburb.csv", index=False)
    try:
        commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = None
    man = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "source": str(Path(src).resolve()), "source_commit": commit or "unknown",
           "flats": int(len(flat)), "flats_with_building": int(flat["osm_id"].notna().sum()),
           "flats_with_gnaf_floor": int(flat["gnaf_floor"].notna().sum()), "flats_with_sp_year": int(flat["gnaf_sp_year"].notna().sum()),
           "date_checks": DATE_CHECKS, "floor_view_rows": int(len(view)),
           "airbnb": None if sub is None else {"branch": AB_REF, "snapshot": "Inside Airbnb Sydney, scraped 2026-09-16/17; listings still online then whose first review <= cutoff (removals in Aug-Sep 2026 are missing)",
                                               "note": "counts of such listings are the pre-cutoff facts used by default; active counts, prices, revenue, scores, licences and host counts are September values (config.AIRBNB_SEPT_VALUES)",
                                               "flats_with_building_counts": int(flat["ab_n_300m"].notna().sum()), "suburbs": int(len(sub))},
           "columns": [c for c in flat.columns if c != "gnaf_id"], "cutoff_rule": "no dated column carried; static building facts only",
           "credits": ["Geoscape G-NAF Aug 2026 (c) Geoscape Australia, Open G-NAF EULA",
                       "LiDAR 2013/2020 (c) State of NSW (Spatial Services) via ELVIS, CC BY",
                       "OpenStreetMap (c) OpenStreetMap contributors, ODbL",
                       "Inside Airbnb (insideairbnb.com), CC BY 4.0"]}
    if flats_path:
        man["organiser_crosscheck"] = crosscheck(flat, flats_path)
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "columns"}, indent=1))
    print(f"wrote {out_dir} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(C.PUBLIC_SRC))
    ap.add_argument("--out", default=str(C.PUBLIC_DIR))
    ap.add_argument("--crosscheck", help="organiser flats file, for the aggregate floor agreement check")
    ap.add_argument("--no-airbnb", dest="airbnb", action="store_false")
    a = ap.parse_args()
    main(a.src, a.out, a.crosscheck, a.airbnb)
