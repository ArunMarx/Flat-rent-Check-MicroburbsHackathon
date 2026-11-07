"""Public per-flat features from the team's G-NAF and LiDAR layers (files written by import_public.py).
Joined on gnaf_id; the LiDAR per-floor view is picked with the flat's floor (organiser floor_level, filled from the
G-NAF floor when blank and GNAF_FLOOR_FILL is on). Nothing here is dated: guard.check_public refuses any date column.
Settings: config.USE_GNAF, config.USE_LIDAR, config.GNAF_FLOOR_FILL, config.PUBLIC_DIR."""
import numpy as np
import pandas as pd

import config as C
import guard
from io_files import log

GNAF_COLS = ["gnaf_floor_rel", "gnaf_floor_src", "gnaf_is_ground", "gnaf_is_top", "gnaf_is_penthouse",
             "gnaf_units_on_floor", "gnaf_bld_units", "gnaf_bld_nonres_units", "gnaf_bld_max_floor",
             "gnaf_bld_floors_known", "gnaf_sp_year", "gnaf_nonres", "gnaf_whole_building"]
PB_COLS = ["pb_h_2020", "pb_levels_lidar", "pb_new_build", "pb_dist_station", "pb_dist_water",
           "pb_canopy_pct", "pb_lst_c", "pb_footprint_m2", "pb_air_path", "pb_rwy_lateral",
           "pb_abs_rent_2021", "pb_join_dist"]
VIEW_COLS = ["vw_roof_margin", "vw_roofline_clear", "vw_water", "vw_park", "vw_from_floor"]
# Inside Airbnb, scraped 16-17 Sep 2026. "Safe" = counts of listings whose FIRST review predates the cutoff, i.e. they
# existed before 31 Jul 2026 (caveat: only listings still online in September are counted, so removals in Aug-Sep
# are missing). Suburb rows exist for the 29 suburbs fully inside the City; Newtown, Paddington, Annandale and
# St Peters flats (8%) get NaN suburb columns but keep the 300 m building counts.
AB_SAFE_COLS = ["ab_n_300m", "ab_n_entire_300m", "ab_sub_n", "ab_sub_months_listed"]
# September-2026 values, after the cutoff (prices, revenue, scores, licences, host counts, and the "active" counts,
# whose rule uses the last review date as of September): only with config.AIRBNB_SEPT_VALUES
AB_SEPT_COLS = ["ab_n_active_300m", "ab_sub_n_active", "ab_n_stra_300m", "ab_n_commercial_300m", "ab_price_night_300m",
                "ab_loc_score_300m", "ab_sub_share_stra", "ab_sub_price_night", "ab_sub_revenue", "ab_sub_loc_score"]
_CACHE = {}


def load_public():
    """(flat table, floor-view table) or (None, None) when the files are absent (pipeline runs without them)."""
    if _CACHE.get("dir") != str(C.PUBLIC_DIR):
        _CACHE.clear()
        _CACHE["dir"] = str(C.PUBLIC_DIR)
    if "flat" in _CACHE:
        return _CACHE["flat"], _CACHE["view"], _CACHE["sub"]
    fp, vp = C.PUBLIC_DIR / "flat_public_features.csv", C.PUBLIC_DIR / "floor_view.csv"
    if not fp.exists():
        log(f"public features: {fp} not found - run import_public.py; continuing without G-NAF/LiDAR features")
        _CACHE["flat"], _CACHE["view"], _CACHE["sub"] = None, None, None
        return None, None, None
    flat = pd.read_csv(guard.check_path(fp), dtype={"gnaf_id": str})
    view = pd.read_csv(guard.check_path(vp)) if vp.exists() else None
    sp = C.PUBLIC_DIR / "airbnb_suburb.csv"
    sub = pd.read_csv(guard.check_path(sp), dtype={"suburb": str}) if sp.exists() else None
    guard.check_public(flat, fp.name)
    if view is not None:
        guard.check_public(view, vp.name)
    if sub is not None:
        guard.check_public(sub, sp.name)
    flat["gnaf_id"] = flat["gnaf_id"].str.strip()
    assert flat["gnaf_id"].is_unique, "flat_public_features.csv must have one row per gnaf_id"
    _CACHE["flat"], _CACHE["view"], _CACHE["sub"] = flat, view, sub
    return flat, view, sub


def attach_public(d):
    """Merge the public columns onto the flats frame (one row per flat kept), fill blank floors, pick per-floor views.
    Returns d unchanged when both layers are off or the files are missing."""
    if not (C.USE_GNAF or C.USE_LIDAR or C.USE_AIRBNB):
        return d
    flat, view, sub = load_public()
    if flat is None:
        return d
    ab_bld = [c for c in AB_SAFE_COLS + AB_SEPT_COLS if not c.startswith("ab_sub_")]
    cols = (["gnaf_id", "osm_id", "gnaf_floor"] + (GNAF_COLS if C.USE_GNAF else []) + (PB_COLS if C.USE_LIDAR else [])
            + (ab_bld if C.USE_AIRBNB else []))
    cols = list(dict.fromkeys(c for c in cols if c in flat.columns))
    n = len(d)
    d = d.merge(flat[cols], on="gnaf_id", how="left")
    assert len(d) == n, "public join changed the number of flats"
    d["floor_src"] = np.where(d["floor"].notna(), 1.0, np.nan)               # 1 organiser, 2 G-NAF fill
    if C.USE_GNAF and C.GNAF_FLOOR_FILL:
        filled = d["floor"].isna() & d["gnaf_floor"].notna()
        d.loc[filled, "floor"] = d.loc[filled, "gnaf_floor"]
        d.loc[filled, "floor_src"] = 2.0
        log(f"public features: G-NAF floor fills {int(filled.sum()):,} blank floors (mostly ground floors, which the organiser leaves blank) "
            f"(floor known for {d['floor'].notna().mean():.0%} of flats)")
    if C.USE_LIDAR and view is not None:
        top = view.groupby("osm_id")["floor"].max().rename("vw_top")
        med = view.groupby("osm_id")[["vw_roof_margin", "vw_roofline_clear", "vw_water", "vw_park"]].median()
        d["vw_top"] = d["osm_id"].map(top)
        # a floor more than one above the LiDAR building's top floor means a post-2020 build or a wrong building:
        # use the building median rather than the top floor's view
        fl = d["floor"].where(d["floor"] <= d["vw_top"] + 1).clip(lower=0)
        fl = np.minimum(fl, d["vw_top"]).round()
        key = pd.DataFrame({"osm_id": d["osm_id"], "floor": fl}).astype({"osm_id": "Int64", "floor": "Int64"})
        v = view.astype({"osm_id": "Int64", "floor": "Int64"})
        per = key.merge(v, on=["osm_id", "floor"], how="left")
        vcols = ["vw_roof_margin", "vw_roofline_clear", "vw_water", "vw_park"]
        hit = per[vcols].notna().all(axis=1).to_numpy()
        for c in vcols:
            bmed = d["osm_id"].map(med[c]).to_numpy()
            d[c] = np.where(hit, per[c].to_numpy(), bmed)
        d["vw_from_floor"] = np.where(hit, 1.0, np.where(d["vw_water"].notna(), 0.0, np.nan))
        d = d.drop(columns=["vw_top"])
        log(f"public features: LiDAR building joined for {d['osm_id'].notna().mean():.0%} of flats; "
            f"per-floor view for {(d['vw_from_floor'] == 1).mean():.0%}, building median for {(d['vw_from_floor'] == 0).mean():.0%}")
    if C.USE_AIRBNB and sub is not None:
        n = len(d)
        d = d.merge(sub.rename(columns={"suburb": "suburb_clean"}), on="suburb_clean", how="left")
        assert len(d) == n, "airbnb suburb join changed the number of flats"
        log(f"public features: Airbnb counts within 300 m for {d['ab_n_300m'].notna().mean():.0%} of flats, "
            f"suburb table for {d['ab_sub_n'].notna().mean():.0%}; September values {'ON' if C.AIRBNB_SEPT_VALUES else 'off'}")
    if not C.USE_GNAF:
        d = d.drop(columns=["gnaf_floor"])
    return d


def public_cols(d=None):
    """Public feature columns the model may use, limited to those present on the frame."""
    want = ((GNAF_COLS if C.USE_GNAF else []) + (PB_COLS + VIEW_COLS if C.USE_LIDAR else [])
            + (AB_SAFE_COLS + (AB_SEPT_COLS if C.AIRBNB_SEPT_VALUES else []) if C.USE_AIRBNB else []))
    if d is None:
        return []
    return [c for c in want if c in d.columns]
