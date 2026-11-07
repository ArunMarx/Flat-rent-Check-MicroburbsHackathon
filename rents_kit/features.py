"""Features (design.md sec. 8.4). Every neighbour feature excludes the target flat's own adverts and uses only
adverts dated before the row's as-of date (T8.1). Rents are log rents in cutoff dollars (la).
Adapted from the valuation kit."""
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import config as C
from io_files import bldg_key, clean_suburb
from public_features import attach_public, public_cols

LAT0, LON0 = -33.8731, 151.2065          # Sydney Town Hall


def to_xy(lat, lon):
    return ((np.asarray(lon, float) - LON0) * 111320.0 * np.cos(np.radians(LAT0)),
            (np.asarray(lat, float) - LAT0) * 110574.0)


def bed_group(b):
    return np.clip(np.asarray(b, float), 0, C.BED_GROUPS_MAX)


# --------------------------------------------------------------------------- flats
def prep_flats(df, blocks=None, cutoff=None):
    d = df.copy().reset_index(drop=True)
    d["suburb_clean"] = clean_suburb(d["suburb"]).to_numpy()
    b = d["building_address"].where(d["building_address"].notna(), d["address"])
    d["bkey"] = bldg_key(b.to_numpy(), d["suburb"].to_numpy()).to_numpy()
    d["bkey"] = d["bkey"].where(pd.notna(d["bkey"]), "G:" + d["gnaf_id"])
    lat = d["latitude"].fillna(d.groupby("suburb_clean")["latitude"].transform("mean")).fillna(d["latitude"].mean())
    lon = d["longitude"].fillna(d.groupby("suburb_clean")["longitude"].transform("mean")).fillna(d["longitude"].mean())
    d["x"], d["y"] = to_xy(lat, lon)
    d["dist_cbd"] = np.hypot(d["x"], d["y"])
    d["floor"] = d["floor_level"]
    d = attach_public(d)                   # G-NAF floor fill + public columns (config.USE_GNAF/USE_LIDAR)
    d["bed_group"] = bed_group(d["bedrooms"])
    d["is_studio"] = (d["bedrooms"] == 0).astype(float).where(d["bedrooms"].notna())      # T7.4
    d["max_floor_bldg"] = d.groupby("bkey")["floor"].transform("max")
    d["rel_floor"] = d["floor"] / d["max_floor_bldg"].replace(0, np.nan)
    d["n_in_input"] = d.groupby("bkey")["gnaf_id"].transform("size")
    # floors above neighbours (T8.4): own floor minus the tallest other building within VIEW_RADIUS_M
    bl = d.groupby("bkey").agg(x=("x", "mean"), y=("y", "mean"), top=("max_floor_bldg", "max")).reset_index()
    tree = cKDTree(bl[["x", "y"]].to_numpy())
    nbrs = tree.query_ball_point(bl[["x", "y"]].to_numpy(), r=C.VIEW_RADIUS_M)
    tops = bl["top"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nmax = np.array([np.nanmax(np.r_[tops[[j for j in ix if j != i]], np.nan]) if len(ix) > 1 else np.nan
                     for i, ix in enumerate(nbrs)]) if len(bl) else np.array([])
    d["nbr_max_floor"] = d["bkey"].map(pd.Series(nmax, index=bl["bkey"]))
    d["floors_above_nbrs"] = d["floor"] - d["nbr_max_floor"]
    d["new_home_share"] = d["block_new_homes_2021_2026"] / d["block_homes_2026"].replace(0, np.nan)
    if blocks is not None:
        extra = [c for c in blocks.columns if c != "mb"][:6]
        d = d.merge(blocks[["mb"] + extra].rename(columns={c: f"blk_{c}" for c in extra}), on="mb", how="left")
    d["suburb_code"] = pd.factorize(d["suburb_clean"])[0]
    cut = pd.Timestamp(cutoff or C.CUTOFF)
    d["years_since_first_listed"] = (cut - pd.to_datetime(d["first_ever_listed"], errors="coerce")).dt.days / 365.25
    d["attrs_age"] = (cut - pd.to_datetime(d["attrs_last_seen"], errors="coerce")).dt.days / 365.25
    return d


def block_cols(d):
    return [c for c in d.columns if c.startswith("blk_")]


# --------------------------------------------------------------------------- feature blocks
OWN = ["la_prev", "prev_age", "n_prev"]
BLD = ["bld_n", "bld_wmean", "bld_wmean_attr", "bld_gap", "bld_std"]
MB = ["mb_n", "mb_wmean", "mb_wmean_attr", "mb_gap", "mb_std"]
KNN = ["knn10_med", "knn40_med", "knn10_dist", "ob_knn10_med", "ob_knn25_med", "ob_knn10_dist"]
SUB = ["sub_mean", "subbed_mean"]
DERIVED = ["d_prev_bld", "d_bld_ob"]
STATIC = ["bedrooms", "bed_group", "bathrooms", "car_space", "car_spaces", "floor", "rel_floor",
          "flats_in_building", "max_floor_bldg", "n_in_input", "x", "y", "dist_cbd", "suburb_code", "is_studio",
          "block_new_homes_2021_2026", "block_homes_2026", "new_home_share", "years_since_first_listed", "attrs_age"]
VIEWS = ["floors_above_nbrs", "nbr_max_floor"]
CATEGORICAL = ["suburb_code", "anchor_src"]


def feature_list(d=None, use_views=None):
    use_views = C.USE_VIEWS if use_views is None else use_views
    return (OWN + BLD + MB + KNN + SUB + DERIVED + STATIC + (VIEWS if use_views else [])
            + (block_cols(d) if d is not None else []) + public_cols(d))


def _years(x):
    a = np.asarray(pd.to_datetime(x), dtype="datetime64[ns]")
    return (a - np.datetime64("2000-01-01", "ns")) / np.timedelta64(1, "D") / 365.25


def _codes(a_t, a_p):
    codes, _ = pd.factorize(pd.Series(np.concatenate([np.asarray(a_t, dtype=object), np.asarray(a_p, dtype=object)])))
    return codes[:len(a_t)], codes[len(a_t):]


def _own_prev(Tid, asof, P):
    left = pd.DataFrame({"pos": np.arange(len(Tid)), "gid": Tid,
                         "asof": np.asarray(pd.to_datetime(asof), dtype="datetime64[ns]")}).sort_values("asof")
    pp = P[P["campaign_first"] & P["gnaf_id"].notna()]
    right = pd.DataFrame({"gid": pp["gnaf_id"].to_numpy(), "t": np.asarray(pp["t"], dtype="datetime64[ns]"),
                          "la_prev": pp["la"].to_numpy(), "n_prev": pp["n_before"].to_numpy() + 1.0}).sort_values("t")
    m = pd.merge_asof(left, right, left_on="asof", right_on="t", by="gid", direction="backward",
                      allow_exact_matches=False).sort_values("pos")
    age = (m["asof"] - m["t"]).dt.days.to_numpy(float) / 365.25
    return {"la_prev": m["la_prev"].to_numpy(float), "prev_age": age, "n_prev": m["n_prev"].to_numpy(float)}


def _kernel(Tc, Pc, Tg, Pg, Tb, Pb, ty, py, Tfl, Pfl, Tbd, Pbd, Pla, Pw, exclude, prefix):
    """Time/floor/bedroom-weighted mean of earlier adverts in the same group, excluding own flat (or building)."""
    n = len(Tc)
    res = {f"{prefix}_{k}": np.full(n, np.nan) for k in ("n", "wmean", "wmean_attr", "gap", "std")}
    tdict = pd.Series(np.arange(n)).groupby(Tc).indices
    pdict = pd.Series(np.arange(len(Pc))).groupby(Pc).indices
    for k, ti in tdict.items():
        if k < 0:
            continue
        pi = pdict.get(k)
        if pi is None:
            continue
        for chunk in np.array_split(ti, max(1, len(ti) * len(pi) // 4_000_000 + 1)):
            dt = ty[chunk][:, None] - py[pi][None, :]
            other = (Tg[chunk][:, None] != Pg[pi][None, :]) if exclude == "gnaf" else (Tb[chunk][:, None] != Pb[pi][None, :])
            valid = other & (dt > 0)                                    # strictly earlier adverts only (T8.1)
            if not valid.any():
                continue
            la = Pla[pi][None, :]
            w = np.exp(-dt / C.TIME_TAU_YEARS) * valid * Pw[pi][None, :]
            dfl = Tfl[chunk][:, None] - Pfl[pi][None, :]
            dbd = Tbd[chunk][:, None] - Pbd[pi][None, :]
            fk = np.where(np.isfinite(dfl), np.exp(-np.abs(dfl) / C.FLOOR_SCALE), 1.0)
            bk = np.where(np.isfinite(dbd), np.exp(-np.abs(dbd) / C.BED_SCALE), 1.0)
            w2 = w * fk * bk
            cnt = valid.sum(1).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                ws, ws2 = w.sum(1), w2.sum(1)
                res[f"{prefix}_wmean"][chunk] = np.where(ws > 0, (w * la).sum(1) / ws, np.nan)
                res[f"{prefix}_wmean_attr"][chunk] = np.where(ws2 > 1e-9, (w2 * la).sum(1) / ws2, np.nan)
                mu = (valid * la).sum(1) / cnt
                var = (valid * (la - mu[:, None]) ** 2).sum(1) / cnt
                res[f"{prefix}_std"][chunk] = np.where(cnt >= 2, np.sqrt(var), np.nan)
                gap = np.where(valid, dt, np.inf).min(1)
                res[f"{prefix}_gap"][chunk] = np.where(np.isfinite(gap), gap, np.nan)
            res[f"{prefix}_n"][chunk] = np.where(cnt > 0, cnt, np.nan)
    return res


def _knn(Tx, Ty, ty, Tg, Tb, P, Pg, Pb, py):
    k = min(C.KNN_QUERY, len(P))
    tree = cKDTree(np.c_[P["x"].to_numpy(), P["y"].to_numpy()])
    dist, idx = tree.query(np.c_[Tx, Ty], k=k, workers=-1)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    la = P["la"].to_numpy()[idx]
    later = py[idx] >= ty[:, None]                                       # T8.1: no same-day or later adverts
    out = {}

    def block(bad, ks, name):
        order = np.argsort(bad, axis=1, kind="stable")
        l2 = np.take_along_axis(np.where(bad, np.nan, la), order, 1)
        d2 = np.take_along_axis(np.where(bad, np.nan, dist), order, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for kk in ks:
                out[f"{name}{kk}_med"] = np.nanmedian(l2[:, :kk], axis=1)
            out[f"{name}10_dist"] = np.nanmean(d2[:, :10], axis=1)

    block((Pg[idx] == Tg[:, None]) | later, (10, 40), "knn")
    block((Pb[idx] == Tb[:, None]) | later, (10, 25), "ob_knn")
    return out


def _group_mean_before(kT, gT, ty, kP, gP, py, la):
    """Mean la of earlier adverts in group k, excluding the target flat. Uses a time-sorted cumulative sum."""
    out = np.full(len(kT), np.nan)
    Pdf = pd.DataFrame({"k": kP, "g": gP, "t": py, "la": la}).dropna(subset=["k"]).sort_values("t")
    grp = {k: (g["t"].to_numpy(), np.cumsum(g["la"].to_numpy())) for k, g in Pdf.groupby("k")}
    own = {kg: (g["t"].to_numpy(), np.cumsum(g["la"].to_numpy())) for kg, g in Pdf.groupby(["k", "g"])}
    for i, (k, g, t) in enumerate(zip(kT, gT, ty)):
        c = grp.get(k)
        if c is None:
            continue
        j = np.searchsorted(c[0], t, side="left")
        if j == 0:
            continue
        s, n = c[1][j - 1], j
        o = own.get((k, g))
        if o is not None:
            jo = np.searchsorted(o[0], t, side="left")
            if jo:
                s -= o[1][jo - 1]
                n -= jo
        if n > 0:
            out[i] = s / n
    return out


def make_features(T, P, asof, fallback_la, d_flats=None, use_views=None):
    """T: target rows (flat attributes + 'gid' identity); P: pool of earlier adverts in cutoff dollars.
    asof: per-target date (training: own advert date; prediction: cutoff + 1 day)."""
    T = T.reset_index(drop=True)
    P = P.reset_index(drop=True)
    ty, py = _years(asof), _years(P["t"])
    Pid = P["gnaf_id"].fillna("R:" + pd.Series(np.arange(len(P))).astype(str)).to_numpy()
    F = _own_prev(T["gid"].to_numpy(), asof, P)
    Tg, Pg = _codes(T["gid"].to_numpy(), Pid)
    Tb, Pb = _codes(T["bkey"].to_numpy(), P["bkey"].to_numpy())
    Tfl, Pfl = T["floor"].to_numpy(float), P["floor"].to_numpy(float)
    Tbd, Pbd = T["bedrooms"].to_numpy(float), P["bedrooms"].to_numpy(float)
    Pla = P["la"].to_numpy(float)
    Pw = 1.0 / P["campaign_n"].to_numpy(float)
    F.update(_kernel(Tb, Pb, Tg, Pg, Tb, Pb, ty, py, Tfl, Pfl, Tbd, Pbd, Pla, Pw, "gnaf", "bld"))
    Tm, Pm = _codes(T["mb"].astype(object).to_numpy(), P["mb"].astype(object).to_numpy())
    F.update(_kernel(Tm, Pm, Tg, Pg, Tb, Pb, ty, py, Tfl, Pfl, Tbd, Pbd, Pla, Pw, "bkey", "mb"))
    F.update(_knn(T["x"].to_numpy(), T["y"].to_numpy(), ty, Tg, Tb, P, Pg, Pb, py))
    F["sub_mean"] = _group_mean_before(T["suburb_clean"].to_numpy(object), Tg, ty,
                                       P["suburb_clean"].to_numpy(object), Pg, py, Pla)
    sbT = T["suburb_clean"].astype(str) + "|" + pd.Series(bed_group(T["bedrooms"])).astype(str).to_numpy()
    sbP = P["suburb_clean"].astype(str) + "|" + pd.Series(bed_group(P["bedrooms"])).astype(str).to_numpy()
    F["subbed_mean"] = _group_mean_before(sbT.to_numpy(object), Tg, ty, sbP.to_numpy(object), Pg, py, Pla)
    F = pd.DataFrame(F)
    # shrink thin building / block means toward suburb-by-bedroom (T8.5)
    prior = F["subbed_mean"].fillna(F["sub_mean"])
    for pre in ("bld", "mb"):
        w = F[f"{pre}_n"] / (F[f"{pre}_n"] + C.SHRINK_K)
        for c in (f"{pre}_wmean", f"{pre}_wmean_attr"):
            F[c] = np.where(F[c].notna() & prior.notna(), w * F[c] + (1 - w) * prior, F[c])
    F["d_prev_bld"] = F["la_prev"] - F["bld_wmean"]
    F["d_bld_ob"] = F["bld_wmean_attr"] - F["ob_knn10_med"]
    cols = feature_list(T, use_views)
    for c in cols:
        if c not in F.columns:
            F[c] = pd.to_numeric(T[c], errors="coerce").to_numpy(float) if c in T.columns else np.nan
    F = F[cols].copy()
    cands = [F["la_prev"].where(F["prev_age"] <= C.OWN_LAST_MAX_AGE_YEARS), F["bld_wmean_attr"], F["bld_wmean"],
             F["ob_knn10_med"], F["knn10_med"], F["subbed_mean"], F["sub_mean"]]
    anchor = pd.Series(np.nan, index=F.index)
    src = pd.Series(len(cands), index=F.index)
    for i, c in enumerate(cands):
        m = anchor.isna() & c.notna()
        anchor[m] = c[m]
        src[m] = i
    F["anchor"] = anchor.fillna(fallback_la)
    F["anchor_src"] = src.astype(int)
    F["suburb_code"] = F["suburb_code"].fillna(-1).astype(int).clip(lower=0)
    return F


ANCHOR_NAMES = ["own last advert", "building (attr-weighted)", "building", "other buildings nearby",
                "nearest flats", "suburb x bedrooms", "suburb", "citywide"]


def with_parking(F, value):
    """T6.1: flip only car_space and car_spaces."""
    G = F.copy()
    G["car_space"] = float(value)
    G["car_spaces"] = np.where(value == 1, np.fmax(G["car_spaces"].fillna(1), 1), 0.0)
    return G
