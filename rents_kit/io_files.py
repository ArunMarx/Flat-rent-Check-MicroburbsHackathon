"""Loading, column matching, lease-to-flat matching, campaigns and merging (design.md sec. 6, 8.2).
T11.4, T11.5, T11.6, T11.7, T9.4."""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import guard


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _norm(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")


# --------------------------------------------------------------------------- address keys
_SUFFIX = re.compile(r"\b(NSW|AUSTRALIA)\b|\b\d{4}\s*$")


def clean_suburb(s):
    return (pd.Series(s, dtype=object).astype(str).str.replace(r"\s*\(.*\)\s*$", "", regex=True)
            .str.strip().str.upper())


def addr_key(a):
    """'UNIT 5, 20 X STREET WATERLOO 2017' and '5/20 X ST, WATERLOO NSW 2017' -> '5|20|X'."""
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return None
    s = str(a).upper().replace(",", " ")
    s = _SUFFIX.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    unit = ""
    m = re.match(r"^(?:UNIT|APARTMENT|APT|FLAT|SUITE|SHOP|LEVEL \d+ UNIT)\s+([A-Z]?\d+[A-Z]?)\s+(.*)$", s)
    if m:
        unit, s = m.group(1), m.group(2)
    else:
        m = re.match(r"^([A-Z]?\d+[A-Z]?)\s*/\s*(.*)$", s)
        if m:
            unit, s = m.group(1), m.group(2)
    m = re.match(r"^(\d+[A-Z]?(?:-\d+[A-Z]?)?)\s+([A-Z']+)", s)
    if not m:
        return None
    return f"{unit}|{m.group(1)}|{m.group(2)}"


def bldg_key(building_address, suburb):
    b = (pd.Series(building_address, dtype=object).astype(str).str.upper()
         .str.replace(r"[^A-Z0-9 \-]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())
    return (b + "|" + clean_suburb(suburb).to_numpy()).where(pd.Series(building_address).notna().to_numpy(), None)


def building_from_address(a):
    k = addr_key(a)
    if k is None:
        return None
    _, num, street = k.split("|")
    return f"{num} {street}"


# --------------------------------------------------------------------------- flats
def load_flats(path):
    guard.check_path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], low_memory=False)
    norm = {_norm(c): c for c in df.columns}
    df = df.rename(columns={norm[k]: k for k in C.FLAT_COLUMNS if k in norm})
    bad = [c for c in C.FLAT_REQUIRED if c not in df.columns]
    if bad:
        raise SystemExit(f"Input flats file is missing {bad}. Columns found: {list(df.columns)}")
    for c in C.FLAT_COLUMNS:
        if c not in df.columns:
            log(f"WARNING: flats file has no '{c}' column - filled with blanks")
            df[c] = np.nan
    df["gnaf_id"] = df["gnaf_id"].astype(str).str.strip()
    if df["gnaf_id"].duplicated().any():
        log(f"WARNING: {int(df['gnaf_id'].duplicated().sum())} duplicate gnaf_id rows in flats file")
    num = ["latitude", "longitude", "bedrooms", "bathrooms", "car_space", "car_spaces", "floor_level",
           "flats_in_building", "block_new_homes_2021_2026", "block_homes_2026", "ever_advertised_for_lease"]
    for c in num:
        df[c] = pd.to_numeric(df[c].replace({"None": "0"}) if c == "car_spaces" else df[c], errors="coerce")
    return df


# --------------------------------------------------------------------------- leases
def pick(df, key, overrides, required=True):
    if key in overrides:
        if overrides[key] not in df.columns:
            raise SystemExit(f"--col {key}={overrides[key]!r} is not a column. Columns: {list(df.columns)}")
        return overrides[key]
    norm = {_norm(c): c for c in df.columns}
    for a in C.LEASE_ALIASES[key]:
        if a in norm:
            return norm[a]
    if required:
        raise SystemExit(f"Cannot find a '{key}' column in the lease file. Columns: {list(df.columns)}.\n"
                         f"Re-run with  --col {key}=\"<column name>\"  (or add an alias in config.LEASE_ALIASES).")
    return None


def parse_dates(s):
    """Detect day-first vs month-first (T11.6). ISO dates parse either way."""
    s = s.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    if iso.mean() > 0.9:
        return pd.to_datetime(s, errors="coerce", format="mixed"), "iso"
    parts = s.str.extract(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})")
    a, b = pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")
    if (b > 12).any() and not (a > 12).any():
        return pd.to_datetime(s, errors="coerce", dayfirst=False), "month-first"
    if not (a > 12).any():
        log("WARNING: ambiguous date format - assuming day-first (Australian)")
    return pd.to_datetime(s, errors="coerce", dayfirst=True), "day-first"


def standardise_leases(raw, overrides=None, src="supplied"):
    overrides = overrides or {}
    raw = guard.strip(raw)
    out = pd.DataFrame(index=raw.index)
    dc, rc = pick(raw, "advert_date", overrides), pick(raw, "rent", overrides)
    keys = {k: pick(raw, k, overrides, required=False) for k in ("gnaf_id", "address", "building_address")}
    if not any(keys.values()):
        raise SystemExit(f"Lease file needs one of gnaf_id / address / building_address. Columns: {list(raw.columns)}")
    for k, c in keys.items():
        out[k] = raw[c].astype(str).str.strip().where(raw[c].notna()) if c else None
    sc = pick(raw, "suburb", overrides, required=False)
    out["suburb"] = raw[sc] if sc else None
    out["advert_date"], fmt = parse_dates(raw[dc])
    out["rent"] = pd.to_numeric(raw[rc].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    for k in ("bedrooms", "bathrooms", "car_spaces", "car_space", "lat", "lng"):
        c = pick(raw, k, overrides, required=False)
        out[k] = pd.to_numeric(raw[c], errors="coerce") if c else np.nan
    cs = out["car_space"].copy()
    out["car_space"] = cs.where(cs.notna(), (out["car_spaces"] > 0).astype(float).where(out["car_spaces"].notna()))
    out["src"] = src
    n0 = len(out)
    out = out[out["rent"].between(C.RENT_MIN, C.RENT_MAX) & out["advert_date"].notna()]
    info = {"rows_in": n0, "rows_kept": len(out), "date_format": fmt,
            "columns_used": {"date": dc, "rent": rc, **keys}}
    return out.reset_index(drop=True), info


def load_leases(path, overrides=None):
    guard.check_path(path)
    raw = pd.read_csv(path, dtype=str, low_memory=False)
    out, info = standardise_leases(raw, overrides, "supplied")
    med = out["rent"].median()
    if med > 1500:
        log(f"WARNING: median rent {med:.0f} looks monthly - check units (divide by 4.345?)")
    log(f"lease file: {info['rows_in']:,} rows -> {info['rows_kept']:,} usable; dates {info['date_format']}; "
        f"columns {info['columns_used']}")
    return out, info


def load_api_leases(cache_dir=None):
    cache_dir = Path(cache_dir or C.CACHE_DIR)
    man = cache_dir / "manifest.json"
    if not man.exists():
        return None
    files = json.loads(man.read_text(encoding="utf-8")).get("leases", {})
    frames = [pd.read_csv(cache_dir / f["file"], dtype=str) for f in files.values() if (cache_dir / f["file"]).exists()]
    if not frames:
        return None
    out, _ = standardise_leases(pd.concat(frames, ignore_index=True), {}, "api")
    return out


# --------------------------------------------------------------------------- matching (T11.5)
def match_leases(leases, flats):
    L = leases.copy()
    fl = flats[["gnaf_id", "address", "building_address", "suburb"]].copy()
    ids = set(fl["gnaf_id"])
    L["match"] = "none"
    m = L["gnaf_id"].isin(ids) if L["gnaf_id"].notna().any() else pd.Series(False, index=L.index)
    L.loc[m, "match"] = "gnaf_id"
    need = ~m & L["address"].notna()
    if need.any():
        fl["akey"] = fl["address"].map(addr_key)
        fl["skey"] = fl["akey"] + "|" + clean_suburb(fl["suburb"]).to_numpy()
        k = L.loc[need, "address"].map(addr_key)
        hit = pd.Series(np.nan, index=k.index, dtype=object)
        if L.loc[need, "suburb"].notna().any():                 # address + suburb first (same street in two suburbs)
            smap = fl.dropna(subset=["skey"]).drop_duplicates("skey", keep=False).set_index("skey")["gnaf_id"]
            hit = (k + "|" + clean_suburb(L.loc[need, "suburb"]).to_numpy()).map(smap)
        amap = fl.dropna(subset=["akey"]).drop_duplicates("akey", keep=False).set_index("akey")["gnaf_id"]
        hit = hit.where(hit.notna(), k.map(amap))
        L.loc[hit.dropna().index, "gnaf_id"] = hit.dropna()
        L.loc[hit.dropna().index, "match"] = "address"
    # building key for every row (lease building_address, else from address, else from the matched flat)
    fmap = fl.set_index("gnaf_id")
    b = L["building_address"].copy()
    b = b.where(b.notna(), L["address"].map(building_from_address) if L["address"].notna().any() else None)
    b = b.where(b.notna(), L["gnaf_id"].map(fmap["building_address"]))
    sub = L["suburb"].where(L["suburb"].notna(), L["gnaf_id"].map(fmap["suburb"]))
    L["bkey"] = bldg_key(b.to_numpy(), sub.to_numpy()).to_numpy()
    L["suburb_clean"] = clean_suburb(sub.to_numpy()).to_numpy()
    nb = (L["match"] == "none") & L["bkey"].notna()
    L.loc[nb, "match"] = "building"
    L.loc[L["match"] == "building", "gnaf_id"] = None
    counts = L["match"].value_counts().to_dict()
    return L, counts


# --------------------------------------------------------------------------- campaigns (T9.4)
def make_campaigns(L):
    L = L.copy()
    flat = L["gnaf_id"].fillna("B:" + L["bkey"].fillna("") + ":" + L["rent"].astype(str))
    L["_k"] = flat
    L = L.sort_values(["_k", "advert_date"]).reset_index(drop=True)
    gap = L.groupby("_k")["advert_date"].diff().dt.days
    new = gap.isna() | (gap > C.CAMPAIGN_DAYS)
    L["campaign_id"] = new.cumsum()
    L["campaign_n"] = L.groupby("campaign_id")["rent"].transform("size")
    L["campaign_first"] = new.to_numpy()
    return L.drop(columns="_k"), int((~new).sum())


# --------------------------------------------------------------------------- merge (T11.7)
def merge_leases(sup, api):
    if api is None or api.empty:
        return sup.copy(), {"api_rows": 0, "added": 0}
    a = api.reset_index(drop=True)
    s = sup[sup["gnaf_id"].notna()][["gnaf_id", "advert_date", "rent"]]
    m = a.reset_index().merge(s, on="gnaf_id", how="inner", suffixes=("", "_s"))
    dup = m[((m["advert_date"] - m["advert_date_s"]).dt.days.abs() <= 3) & ((m["rent"] - m["rent_s"]).abs() <= 5)]
    a = a.drop(index=dup["index"].unique())
    out = pd.concat([sup, a], ignore_index=True)
    return out, {"api_rows": int(len(api)), "added": int(len(a)), "overlap": int(dup["index"].nunique())}


# --------------------------------------------------------------------------- blocks
def load_blocks(path):
    if path is None or not Path(path).exists():
        return None
    guard.check_path(path)
    b = pd.read_csv(path, dtype=str)
    norm = {_norm(c): c for c in b.columns}
    mbc = next((norm[k] for k in ("mb", "mb_code", "mb_code_2021", "mesh_block") if k in norm), None)
    if mbc is None:
        log(f"WARNING: blocks file has no mesh-block column; ignored. Columns: {list(b.columns)}")
        return None
    b = b.rename(columns={mbc: "mb"})
    keep = ["mb"]
    for c in b.columns:
        if c != "mb":
            v = pd.to_numeric(b[c], errors="coerce")
            if v.notna().mean() > 0.5:                 # keep numeric columns only (text like suburb/mb_cat dropped)
                b[c] = v
                keep.append(c)
    return b[keep].drop_duplicates("mb")
