"""API pulls (design.md sec. 10; T11.7, T7.7, S10).
  python api_pull.py leases   [--mock ../data_synth/mock_api]   per-suburb rent-history up to the cutoff
  python api_pull.py coverage --flats F [--leases L] [--mock ...] which missing addresses the API can reach
Run 'leases' in the background at 11:00; it is resumable and stops at config.API_DEADLINE."""
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

import config as C
import guard
from api_client import Client
from io_files import load_flats, load_leases, log


def _past_deadline():
    h, m = map(int, C.API_DEADLINE.split(":"))
    now = dt.datetime.now()
    return now.hour * 60 + now.minute >= h * 60 + m and now.date() == dt.date(2026, 11, 7)


def covered_suburbs(cl):
    """Exact SAL names. /v1/suburbs/list without a filter returns 1,000 Australian suburbs, so we filter by LGA and
    sanity-check the count; otherwise use the 29 names from the API docs (config.API_SUBURBS)."""
    r = cl.get("/v1/suburbs/list", {"lga": "Sydney"})
    names = [row["suburb"] for row in (r["body"].get("data") or [])] if r["status"] == 200 else []
    if 20 <= len(names) <= 40:
        return names
    log(f"suburb list returned {len(names)} names - using the 29 from the API docs")
    return C.API_SUBURBS


def pull_leases(cl, suburbs=None, cache_dir=None):
    cache_dir = Path(cache_dir or C.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    man_path = cache_dir / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {"leases": {}}
    me = cl.get("/v1/me", use_cache=False)
    if me["status"] != 200:
        log(f"API check failed ({me['status']}); continue with supplied data only")
        return man
    suburbs = suburbs or covered_suburbs(cl)
    log(f"pulling rent-history for {len(suburbs)} suburbs up to {C.CUTOFF}")
    for s in suburbs:
        if s in man["leases"] and man["leases"][s].get("complete"):
            continue
        if _past_deadline():
            log("API deadline reached - stopping; partial pull is still usable")
            break
        rows, off, total = [], 0, None
        while True:
            r = cl.get(f"/v1/suburbs/{s}/listings/rent-history",
                       {"from_date": C.API_FROM_DATE, "to_date": C.CUTOFF, "limit": C.API_PAGE, "offset": off})
            if r["status"] == 403:
                log(f"  {s}: outside API area (403)")
                break
            if r["status"] != 200:
                log(f"  {s}: failed with {r['status']} after retries - skipped")
                break
            data = r["body"].get("data")
            if data is None:                       # "no data" is HTTP 200 with data: null
                log(f"  {s}: no data ({r['body'].get('reason')})")
                break
            res = data.get("results", [])
            rows.extend(res)
            total = data.get("meta", {}).get("total", len(rows))
            off += len(res)
            if len(res) < C.API_PAGE or off >= total:
                break
        if not rows:
            continue
        df = pd.DataFrame(rows)
        n0 = len(df)
        df = guard.admit(guard.strip(df), "listing_date", C.CUTOFF, "api")   # post-filter (T2.5)
        f = f"leases_{hashlib.sha1(s.encode()).hexdigest()[:8]}.csv"
        df.to_csv(cache_dir / f, index=False)
        man["leases"][s] = {"file": f, "rows": len(df), "dropped_after_cutoff": n0 - len(df), "total_reported": total,
                            "to_date": C.CUTOFF, "pulled": dt.datetime.now().isoformat(timespec="seconds"),
                            "sha1": hashlib.sha1((cache_dir / f).read_bytes()).hexdigest(), "complete": True}
        man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
        log(f"  {s}: {len(df):,} adverts (dropped {n0 - len(df):,} after cutoff)")
    log(f"API stats {cl.stats}")
    return man


def coverage(cl, flats_path, leases_path=None):
    flats = load_flats(flats_path)
    miss = flats[flats["block_homes_2026"].isna()]
    cov = set(covered_suburbs(cl))
    by = miss.groupby("suburb").size().sort_values(ascending=False)
    rep = pd.DataFrame({"flats": by, "in_api": [s in cov for s in by.index]})
    share = None
    if leases_path:
        L, _ = load_leases(leases_path)
        if L["gnaf_id"].notna().any():
            share = float(miss["gnaf_id"].isin(set(L["gnaf_id"])).mean())
    print(rep.to_string())
    n_cov = int(rep.loc[rep["in_api"], "flats"].sum())
    verdict = ("NOT POSSIBLE (outside API coverage)" if n_cov == 0 else
               ("OPEN" if share is not None and share < C.API_GATE_SHARE else "NOT NEEDED"))
    print(f"\nmissing-file flats: {len(miss):,}; in API suburbs: {n_cov:,}; with lease rows: "
          f"{'unknown' if share is None else f'{share:.0%}'}  ->  API fill {verdict}  (T7.7)")
    return verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["leases", "coverage", "smoke"])
    ap.add_argument("--mock", help="serve the synthetic mock API instead of the real one")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--flats")
    ap.add_argument("--leases")
    a = ap.parse_args()
    cl = Client(mock_dir=a.mock, cache_dir=a.cache, log=log)
    if a.what == "smoke":                          # 11:00 check: key works, field names as expected (T11.8)
        me = cl.get("/v1/me", use_cache=False)
        print("me:", me["status"], me["body"])
        sub = covered_suburbs(cl)[0]
        print("first covered suburb:", sub)
        r = cl.get(f"/v1/suburbs/{sub}/listings/rent-history",
                   {"to_date": C.CUTOFF, "limit": 3, "offset": 0}, use_cache=False)
        d = r["body"].get("data") or {}
        print("rent-history:", r["status"], "total", d.get("meta", {}).get("total"))
        for row in d.get("results", []):
            print("  ", {k: row.get(k) for k in ("gnaf_pid", "listing_date", "weekly_rent", "bedrooms", "car_spaces")})
        sys.exit(0)
    if a.what == "leases":
        pull_leases(cl, cache_dir=a.cache)
    else:
        if not a.flats:
            sys.exit("--flats is required for coverage")
        coverage(cl, a.flats, a.leases)
