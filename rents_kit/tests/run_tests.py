"""All tests in one command (T11.3):  python run.py test --data ..\\data_synth
Uses a small 3-suburb slice so it runs in about a minute. Exits non-zero on any failure."""
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import guard  # noqa: E402
from api_client import BlockedEndpoint, Client, endpoint_allowed  # noqa: E402
from features import make_features, prep_flats  # noqa: E402
from io_files import addr_key, load_flats, load_leases, make_campaigns, match_leases, merge_leases  # noqa: E402
from pipeline import attach_attributes, curate, predict, train  # noqa: E402
from validate_submission import validate  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else C.ROOT.parent / "data_synth")
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def small():
    flats_raw = load_flats(DATA / "rents-input-flats-city-of-sydney.csv")
    keep = sorted(flats_raw["suburb"].unique())[:2] + sorted(flats_raw["suburb"].unique())[-1:]
    fr = flats_raw[flats_raw["suburb"].isin(keep)].reset_index(drop=True)
    f, _ = guard.mask_flats(fr, C.CUTOFF)
    d = prep_flats(f)
    sup, _ = load_leases(DATA / "rents-past-leases-city-of-sydney.csv")
    sup = sup[sup["gnaf_id"].isin(set(d["gnaf_id"]))]
    return fr, d, sup


def t_guard_and_cutoff(d, sup):
    print("cutoff guard (T2.1, T2.2, T2.3)")
    s_all = sup.copy()
    clean = guard.admit(s_all, "advert_date", C.CUTOFF, "test")
    check("post-cutoff lease rows dropped", clean["advert_date"].max() <= pd.Timestamp(C.CUTOFF),
          f"{len(s_all) - len(clean)} dropped")
    planted = pd.concat([clean, clean.sample(200, random_state=1).assign(
        advert_date=pd.Timestamp("2026-08-20"), rent=lambda x: x["rent"] * 3)], ignore_index=True)
    A1, _ = curate(d, guard.admit(clean, "advert_date", C.CUTOFF, "t"), "supplied")
    A2, _ = curate(d, guard.admit(planted, "advert_date", C.CUTOFF, "t"), "supplied")
    end = pd.Timestamp(C.CUTOFF) + pd.Timedelta(days=1)
    C.SEEDS = [0]
    f1 = train(d, A1, end, verbose=False)
    f2 = train(d, A2, end, verbose=False)
    p1 = predict(f1, d)[1]
    p2 = predict(f2, d)[1]
    check("planted post-cutoff rows change no prediction", np.array_equal(p1, p2))
    fl = d.copy()
    fl["first_ever_listed"] = "2026-09-01"
    fl["attrs_last_seen"] = "2026-10-01"
    m, counts = guard.mask_flats(fl, C.CUTOFF)
    check("post-cutoff input dates masked", m["first_ever_listed"].isna().all() and m["attrs_last_seen"].isna().all(), counts)
    try:
        guard.check_path("x/synthetic_frozen.csv")
        ok = False
    except SystemExit:
        ok = True
    check("frozen file refused", ok)
    return A1, f1


def t_matching():
    print("matching and loading (T11.4, T11.5, T11.6, T7.4, T7.6)")
    check("'5/20 X ST' equals 'UNIT 5, 20 X STREET'", addr_key("5/20 Dunkerley Pl, Waterloo NSW 2017") ==
          addr_key("UNIT 5, 20 DUNKERLEY PLACE WATERLOO 2017"))
    flats_raw = load_flats(DATA / "rents-input-flats-city-of-sydney.csv")
    v, info = load_leases(DATA / "variant-leases-address-keyed.csv")
    check("day-first dates detected", info["date_format"] == "day-first", info["date_format"])
    M, counts = match_leases(v.head(5000), flats_raw)
    check("address-keyed lease file matches", counts.get("address", 0) > 4500, counts)
    tmp = Path(tempfile.mkdtemp())
    raw = pd.read_csv(DATA / "rents-past-leases-city-of-sydney.csv", nrows=50).rename(columns={"weekly_rent": "Weekly Amount"})
    raw.to_csv(tmp / "x.csv", index=False)
    try:
        load_leases(tmp / "x.csv")
        ok = False
    except SystemExit as e:
        ok = "Weekly Amount" in str(e)
    check("unknown rent column fails with columns listed", ok)
    L, _ = load_leases(tmp / "x.csv", {"rent": "Weekly Amount"})
    check("--col override works", len(L) > 0)
    shutil.rmtree(tmp)
    d = prep_flats(guard.mask_flats(flats_raw, C.CUTOFF)[0])
    check("bedroom outliers grouped as 4+", d.loc[d["bedrooms"] > 6, "bed_group"].eq(4).all())
    check("studios from bedrooms, not property_type", d.loc[d["property_type"] == "studio", "is_studio"].eq(0).any())


def t_api():
    print("API client and pull (T2.5, T11.7, T11.8)")
    for p in ("/v1/suburbs/X/listings/for-rent", "/v1/suburbs/X/market/rent-map", "/v1/properties/G/basics/all",
              "/v1/properties/G/valuation/avm", "/v1/suburbs/X/market/median-rent", "/v1/properties/G/rent-history-latest"):
        try:
            Client(mock_dir=DATA / "mock_api", cache_dir=tempfile.mkdtemp()).get(p)
            ok = False
        except BlockedEndpoint:
            ok = True
        check(f"blocked: {p.split('/', 4)[-1]}", ok)
    check("allowed: median-rent-series", endpoint_allowed("/v1/suburbs/X/market/median-rent-series"))
    check("allowed: listings/rent-history", endpoint_allowed("/v1/suburbs/X/listings/rent-history"))
    import api_pull
    tmp = Path(tempfile.mkdtemp())
    cl = Client(mock_dir=DATA / "mock_api", cache_dir=tmp, log=lambda m: None)
    old = C.CUTOFF
    C.CUTOFF = "2026-12-31"                      # make the mock return late rows, then filter with the real cutoff
    subs = sorted(cl.t.covered)[:2] + sorted(cl.t.uncovered)[:1]
    C.CUTOFF = old
    real_get = cl.get

    def get_late(path, params=None, use_cache=True):   # simulate a server that ignores to_date
        if params and "to_date" in params:
            params = {**params, "to_date": "2026-12-31"}
        return real_get(path, params, use_cache)
    cl.get = get_late
    man = api_pull.pull_leases(cl, subs, cache_dir=tmp)
    rows = sum(v["rows"] for v in man["leases"].values())
    dropped = sum(v["dropped_after_cutoff"] for v in man["leases"].values())
    check("API rows after cutoff dropped by guard", dropped > 0, f"{dropped} dropped, {rows} kept")
    df = pd.concat(pd.read_csv(tmp / v["file"]) for v in man["leases"].values())
    check("no API row after cutoff in cache", pd.to_datetime(df["listing_date"]).max() <= pd.Timestamp(C.CUTOFF))
    check("'leased' column removed", "leased" not in df.columns)
    check("429 honoured and 5xx retried", cl.stats["429"] >= 1 and cl.stats["retries"] >= 1, cl.stats)
    check("403 recorded for uncovered suburb", cl.stats["403"] >= 1)
    check("manifest written with to_date", all(v["to_date"] == C.CUTOFF for v in man["leases"].values()))
    shutil.rmtree(tmp)


def t_merge(sup):
    print("merge and campaigns (T11.7, T9.4)")
    api = sup.head(500).copy()
    api["src"] = "api"
    api["advert_date"] = api["advert_date"] + pd.Timedelta(days=2)
    extra = sup.head(100).copy()
    extra["src"], extra["advert_date"] = "api", extra["advert_date"] - pd.Timedelta(days=400)
    out, info = merge_leases(sup, pd.concat([api, extra]))
    check("overlapping API rows deduped, supplied kept", info["overlap"] == 500 and info["added"] == 100, info)
    L = pd.DataFrame({"gnaf_id": ["A", "A", "A"], "bkey": ["b"] * 3, "rent": [500, 480, 520],
                      "advert_date": pd.to_datetime(["2025-01-01", "2025-01-20", "2026-01-01"])})
    L2, n = make_campaigns(L)
    check("relist within 45 days collapsed", n == 1 and L2["campaign_id"].nunique() == 2)


def t_leakage(d, A1, fit):
    print("leakage (T8.1)")
    P = fit.P
    g = P[P["in_input"]]["gnaf_id"].value_counts()
    gid = g.index[g >= 2][0]
    T = d[d["gnaf_id"] == gid].assign(gid=gid)
    asof = np.full(1, fit.ref + pd.Timedelta(days=1), dtype="datetime64[ns]")
    F1 = make_features(T, P, asof, fit.fallback)
    P2 = P.copy()
    P2.loc[P2["gnaf_id"] == gid, "la"] += 1.0
    F2 = make_features(T, P2, asof, fit.fallback)
    nb = [c for c in F1.columns if c.startswith(("bld_", "mb_", "knn", "ob_knn", "sub"))]
    same = all((np.isclose(F1[c], F2[c]) | (F1[c].isna() & F2[c].isna())).all() for c in nb)
    check("own adverts never enter neighbour features", same)
    check("own history moves (positive control)", not np.isclose(F1["la_prev"], F2["la_prev"]).all())
    tr = P[P["in_input"]].head(300).copy()
    Tt = tr[["gnaf_id", "t", "bedrooms", "car_space", "car_spaces"]].merge(
        d.drop(columns=["bedrooms", "car_space", "car_spaces"]), on="gnaf_id")
    Tt["gid"] = Tt["gnaf_id"]
    P3 = P.copy()
    late = P3["t"] >= Tt["t"].max()
    P3.loc[late, "la"] += 5.0
    Fa = make_features(Tt, P, Tt["t"], fit.fallback)
    Fb = make_features(Tt, P3, Tt["t"], fit.fallback)
    nb2 = [c for c in Fa.columns if c.startswith(("bld_", "mb_", "knn", "ob_knn", "sub", "la_prev"))]
    same2 = all((np.isclose(Fa[c], Fb[c]) | (Fa[c].isna() & Fb[c].isna())).all() for c in nb2)
    check("no feature uses adverts on or after the row's date", same2)


def t_outputs(fr, d, A1, fit):
    print("submission, repro, outage (T1.x, T11.2, T11.8)")
    from submission import assemble
    tmp = Path(tempfile.mkdtemp())
    F, w, wo, _ = predict(fit, d)
    p = assemble(fr, d["gnaf_id"].to_numpy(), w, wo, "Test Team", tmp, fit.P["rent"])
    check("valid CSV written and checked", p.exists() and validate(p, fr))
    h1 = hashlib.sha1(p.read_bytes()).hexdigest()
    F, w, wo, _ = predict(fit, d)
    p2 = assemble(fr, d["gnaf_id"].to_numpy(), w, wo, "Test Team", tmp, fit.P["rent"])
    check("same inputs give byte-identical CSV", hashlib.sha1(p2.read_bytes()).hexdigest() == h1)
    s = pd.read_csv(p, dtype={"gnaf_id": str})
    bad = [("drop an id", s.iloc[1:]), ("blank", s.assign(rent_with_car_space=lambda x: x.rent_with_car_space.where(x.index > 0))),
           ("with < without", s.assign(rent_with_car_space=s.rent_without_car_space - 10)),
           ("cents", s.assign(rent_without_car_space=s.rent_without_car_space + 0.25))]
    print("  (the next five 'SUBMISSION CHECK FAILED' messages are expected - the checker must catch them)")
    for name, df in bad:
        q = tmp / "rents_test-team.csv"
        df.to_csv(q, index=False)
        check(f"checker catches: {name}", not validate(q, fr))
    q = tmp / "Rents_Test Team.csv"
    s.to_csv(q, index=False)
    check("checker catches: bad file name", not validate(q, fr))
    old = C.CACHE_DIR
    C.CACHE_DIR = tmp / "no_cache_here"
    A, info = curate(d, A1[A1["src"] == "supplied"][[c for c in ("gnaf_id", "address", "building_address", "suburb",
                     "advert_date", "rent", "bedrooms", "bathrooms", "car_spaces", "car_space", "lat", "lng", "src")]], "merged")
    C.CACHE_DIR = old
    check("API outage: merged run falls back to supplied only", len(A) > 0 and "merge" not in info)
    check("premium >= 0 and varies by bedroom group", (w - wo >= 0).all() and len(set(np.round(list(
        v for k, v in fit.premium.items() if isinstance(k, float)), 4))) > 1, {k: round(v, 3) for k, v in fit.premium.items() if isinstance(k, float)})
    shutil.rmtree(tmp)


if __name__ == "__main__":
    t0 = time.time()
    seeds = C.SEEDS
    fr, d, sup = small()
    A1, fit = t_guard_and_cutoff(d, sup)
    t_matching()
    t_api()
    t_merge(sup)
    t_leakage(d, A1, fit)
    t_outputs(fr, d, A1, fit)
    C.SEEDS = seeds
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if n_fail else 0)
