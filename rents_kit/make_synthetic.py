"""Rehearsal data in the published layout, with planted traps (design.md sec. 10).
Accuracy on this data proves plumbing and leak-safety only - not hackathon performance.
Usage:  python make_synthetic.py --out ../data_synth"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STREETS = ["DUNKERLEY PLACE", "GADIGAL AVENUE", "BOURKE STREET", "CROWN STREET", "HARRIS STREET", "MACLEAY STREET",
           "BOTANY ROAD", "ELIZABETH STREET", "GEORGE STREET", "PITT STREET", "JONES STREET", "MCEVOY STREET",
           "BAPTIST STREET", "ARCHIBALD AVENUE", "SAM SING STREET", "POWELL STREET", "GLEBE POINT ROAD"]
N_FLATS, N_MISSING, N_BLANK_CAR, N_FLOOR_KNOWN, N_NEVER = 65_777, 4_246, 5_793, 34_396, 16_094


def month_index():
    months = pd.period_range("2015-01", "2026-12", freq="M")
    g = []
    for p in months:
        y, m = p.year, p.month
        if y < 2020 or (y == 2020 and m < 4):
            g.append(0.002)
        elif y == 2020:
            g.append(-0.010)
        elif y <= 2023:
            g.append(0.008)
        elif y <= 2025:
            g.append(0.003)
        else:
            g.append(0.010)
    c = pd.Series(np.cumsum(g), index=months.astype(str))
    return c - c["2026-07"]


def main(out, seed=7):
    rng = np.random.default_rng(seed)
    out = Path(out)
    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "mock_api").mkdir(parents=True, exist_ok=True)

    # ---------------- suburbs: 29 covered by the API, 4 not (hold the 4,246 "missing" addresses)
    names = [f"SUBURB{i:02d}" for i in range(33)]
    covered = names[:29]
    sub = pd.DataFrame({"suburb": [n.title() + (" (NSW)" if i % 5 == 0 else "") for i, n in enumerate(names)],
                        "clean": names, "cx": rng.uniform(-3500, 3500, 33), "cy": rng.uniform(-4000, 3000, 33),
                        "base": rng.uniform(430, 700, 33), "pc": rng.integers(2000, 2020, 33)})
    sub["covered"] = sub["clean"].isin(covered)

    # ---------------- buildings and flats
    rows, bid, filled_missing = [], 0, 0
    total = 0
    while total < N_FLATS:
        if filled_missing < N_MISSING:
            s = int(rng.integers(29, 33))
        else:
            s = int(rng.integers(0, 29))
        n = int(np.clip(rng.lognormal(2.6, 0.8), 2, 300))
        n = min(n, N_FLATS - total)
        if s >= 29:
            n = min(n, N_MISSING - filled_missing)
            filled_missing += n
        if n <= 0:
            continue
        per_floor = int(rng.integers(3, 9))
        floors = int(np.ceil(n / per_floor))
        bx, by = sub.cx[s] + rng.normal(0, 500), sub.cy[s] + rng.normal(0, 500)
        street = STREETS[bid % len(STREETS)]
        num = int(rng.integers(1, 400))
        beff = rng.normal(0, 0.08)
        mb = f"112{s:03d}{(bid % 97):05d}0"
        for k in range(n):
            fl = k // per_floor + 1
            unit = fl * 100 + (k % per_floor) + 1
            rows.append((bid, s, street, num, fl, unit, floors, bx + rng.normal(0, 8), by + rng.normal(0, 8), beff, mb))
        bid += 1
        total += n
    f = pd.DataFrame(rows, columns=["bid", "s", "street", "num", "floor", "unit", "floors", "x", "y", "beff", "mb"])
    f["gnaf_id"] = ["GANSW7%08d" % i for i in rng.choice(10**8, len(f), replace=False)]
    f["suburb"] = sub.suburb.to_numpy()[f.s]
    f["building_address"] = f.num.astype(str) + " " + f.street
    f["address"] = ("UNIT " + f.unit.astype(str) + ", " + f.building_address + " " + sub.clean.to_numpy()[f.s] + " "
                    + sub.pc.to_numpy()[f.s].astype(str))
    lat0, lon0 = -33.8731, 151.2065
    f["latitude"] = lat0 + f.y / 110574.0
    f["longitude"] = lon0 + f.x / (111320.0 * np.cos(np.radians(lat0)))
    beds = rng.choice([0, 1, 2, 3, 4], len(f), p=[0.08, 0.42, 0.38, 0.10, 0.02])
    f["bedrooms_true"] = beds
    f["bathrooms"] = np.where(beds >= 2, rng.choice([1, 2], len(f), p=[0.4, 0.6]), 1)
    f["car_true"] = (rng.random(len(f)) < 0.55).astype(int)
    f["flats_in_building"] = f.groupby("bid")["gnaf_id"].transform("size")
    top = f.groupby("bid").floors.first()
    # floors above neighbours (true view effect)
    bx = f.groupby("bid")[["x", "y"]].first()
    from scipy.spatial import cKDTree
    tr = cKDTree(bx.to_numpy())
    nb = tr.query_ball_point(bx.to_numpy(), 100)
    nmax = pd.Series([max([top.iat[j] for j in ix if j != i], default=0) for i, ix in enumerate(nb)], index=bx.index)
    f["above"] = f.floor - f.bid.map(nmax)
    f["true_log"] = (np.log(sub.base.to_numpy()[f.s]) + 0.28 * np.clip(beds, 0, 4) - 0.1 * (beds == 0)
                     + 0.07 * (f.bathrooms == 2) + 0.010 * f.floor + 0.05 * np.clip(f.above, 0, 10) / 5
                     + f.beff + rng.normal(0, 0.05, len(f)))
    # input view of the flats (with blanks and errors)
    f["bedrooms"] = beds.astype(float)
    err = rng.choice(len(f), 30, replace=False)
    f.loc[err, "bedrooms"] = rng.integers(10, 16, 30)                         # planted outliers (T7.6)
    f["property_type"] = np.where(beds == 0, "studio", rng.choice(["apartment", "unit"], len(f)))
    st = rng.choice(np.where(beds == 1)[0], 200, replace=False)
    f.loc[st, "property_type"] = "studio"                                      # studios with 1 bedroom (T7.4)
    f["car_space"] = f.car_true.astype(float)
    f.loc[rng.choice(len(f), N_BLANK_CAR, replace=False), "car_space"] = np.nan
    f["car_spaces"] = f.car_space
    f["floor_level"] = np.nan
    known = rng.choice(len(f), N_FLOOR_KNOWN, replace=False)
    f.loc[known, "floor_level"] = f.floor.to_numpy()[known]
    miss = f.s >= 29
    blk = f.groupby("mb").size()
    f["block_homes_2026"] = f.mb.map(blk * 1.6).round()
    f["block_new_homes_2021_2026"] = f.mb.map((blk * rng.uniform(0, 0.3, len(blk))).round())
    f.loc[miss, ["block_homes_2026", "block_new_homes_2021_2026"]] = np.nan

    # ---------------- adverts (2016 .. Oct 2026)
    idx = month_index()
    never = rng.choice(len(f), N_NEVER, replace=False)
    adv = np.ones(len(f), bool)
    adv[never] = False
    recs = []
    for i in np.where(adv)[0]:
        t = pd.Timestamp("2015-06-01") + pd.Timedelta(days=int(rng.integers(0, 900)))
        while t < pd.Timestamp("2026-10-15"):
            lr = f.true_log.iat[i] + idx[str(t.to_period("M"))] + 0.10 * f.car_true.iat[i] + rng.normal(0, 0.06)
            rent = float(np.round(np.exp(lr) / 10) * 10)
            recs.append((i, t, rent))
            if rng.random() < 0.12:                                            # relist at a lower rent (T9.4)
                recs.append((i, t + pd.Timedelta(days=int(rng.integers(10, 40))), float(np.round(rent * 0.95 / 5) * 5)))
            t = t + pd.Timedelta(days=int(rng.integers(330, 760)))
    A = pd.DataFrame(recs, columns=["i", "listing_date", "weekly_rent"])
    A["gnaf_pid"] = f.gnaf_id.to_numpy()[A.i]
    A["building_address"] = f.building_address.to_numpy()[A.i]
    A["suburb"] = f.suburb.to_numpy()[A.i]
    A["address"] = f.address.to_numpy()[A.i]
    A["bedrooms"] = f.bedrooms_true.to_numpy()[A.i].astype(float)
    A["bathrooms"] = f.bathrooms.to_numpy()[A.i]
    A["car_spaces"] = f.car_true.to_numpy()[A.i].astype(float)
    gap = rng.random(len(A)) < 0.04
    A.loc[gap, "bedrooms"] = np.nan
    A.loc[rng.random(len(A)) < 0.04, "car_spaces"] = np.nan
    A["leased"] = rng.random(len(A)) < 0.8
    A["lat"] = f.latitude.to_numpy()[A.i]
    A["lng"] = f.longitude.to_numpy()[A.i]
    A["property_type"] = "Unit"

    # first_ever_listed / attrs_last_seen / ever_advertised from FULL history (planted post-cutoff values, T2.2)
    first = A.groupby("i").listing_date.min()
    f["first_ever_listed"] = pd.Series(f.index.map(first)).dt.strftime("%Y-%m-%d")
    f["ever_advertised_for_lease"] = adv.astype(int)
    f["attrs_last_seen"] = (pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 1400, len(f)), "D")).strftime("%Y-%m-%d")
    f.loc[rng.random(len(f)) < 0.1, "attrs_last_seen"] = "0001-01-01"

    cols = ["gnaf_id", "address", "building_address", "suburb", "mb", "latitude", "longitude", "property_type",
            "bedrooms", "bathrooms", "car_space", "car_spaces", "floor_level", "flats_in_building",
            "block_new_homes_2021_2026", "block_homes_2026", "ever_advertised_for_lease", "first_ever_listed",
            "attrs_last_seen"]
    f[cols].to_csv(out / "rents-input-flats-city-of-sydney.csv", index=False)

    # supplied lease file: 1 May 2025 .. 31 Jul 2026 plus 300 planted post-cutoff rows (T2.1)
    lcols = ["gnaf_pid", "building_address", "suburb", "listing_date", "weekly_rent", "bedrooms", "bathrooms",
             "car_spaces", "leased"]
    sup = A[(A.listing_date >= "2025-05-01") & (A.listing_date <= "2026-07-31")]
    late = A[A.listing_date > "2026-07-31"].sample(300, random_state=seed)
    s_out = pd.concat([sup, late]).sort_values("listing_date")
    s_out.assign(listing_date=s_out.listing_date.dt.strftime("%Y-%m-%d"))[lcols].to_csv(
        out / "rents-past-leases-city-of-sydney.csv", index=False)
    # address-keyed variant (T11.5)
    s_out.assign(listing_date=s_out.listing_date.dt.strftime("%d/%m/%Y"))[
        ["address", "suburb", "listing_date", "weekly_rent", "bedrooms", "car_spaces"]].to_csv(
        out / "variant-leases-address-keyed.csv", index=False)

    # blocks file (City blocks only)
    b = f[~miss].groupby("mb").agg(persons_2021=("gnaf_id", "size")).reset_index()
    b["persons_2021"] = b["persons_2021"] * 2
    b["dwellings_2021"] = (b["persons_2021"] / 1.9).round()
    b.to_csv(out / "blocks-homes-and-people-2021-2026.csv", index=False)

    # private frozen set (Aug-Sep, first advert per campaign, about 1,700 rows) - local scorer only
    fz = A[(A.listing_date >= "2026-08-01") & (A.listing_date <= "2026-09-30")].drop_duplicates("i").sample(
        frac=1, random_state=seed).head(1700)
    fz[["gnaf_pid", "listing_date", "weekly_rent", "car_spaces"]].to_csv(out / "private" / "synthetic_frozen.csv", index=False)

    # mock API: every advert per covered suburb, including post-cutoff rows; late-July rows carry an updated price (T11.9)
    api = A[A.suburb.isin(sub.suburb[sub.covered])].copy()
    jul = (api.listing_date >= "2026-07-01") & (api.listing_date <= "2026-07-31")
    api.loc[jul, "weekly_rent"] = (api.loc[jul, "weekly_rent"] * 0.97 / 5).round() * 5
    api["listing_date"] = api.listing_date.dt.strftime("%Y-%m-%d")
    acols = ["address", "bathrooms", "bedrooms", "car_spaces", "gnaf_pid", "lat", "leased", "listing_date", "lng",
             "property_type", "weekly_rent"]
    for s_name, g in api.groupby("suburb"):
        g.sort_values("listing_date", ascending=False)[acols].to_csv(out / "mock_api" / f"{s_name}.csv", index=False)
    json.dump({"covered": list(sub.suburb[sub.covered]), "uncovered": list(sub.suburb[~sub.covered])},
              open(out / "mock_api" / "_suburbs.json", "w", encoding="utf-8"))
    print(f"wrote {out}: flats {len(f):,} (missing-file flats {int(miss.sum()):,}), adverts total {len(A):,}, "
          f"supplied {len(s_out):,}, frozen {len(fz):,}, mock API rows {len(api):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data_synth"))
    main(ap.parse_args().out)
