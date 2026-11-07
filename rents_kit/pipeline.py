"""Orchestration: curated data -> index -> features -> model -> eval / final (design.md sec. 7, 8, 11)."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import guard
from baselines import own_last, sample_method, suburb_bedroom_median, suburb_median
from features import ANCHOR_NAMES, bed_group, feature_list, make_features, prep_flats, to_xy, with_parking
from index import build_rent_index
from io_files import load_api_leases, log, make_campaigns, match_leases, merge_leases
from model import RentModel


# --------------------------------------------------------------------------- curated adverts
def attach_attributes(L, d):
    """Give every advert the attributes features need; advert-time bedrooms/parking win over the flat's."""
    A = L.copy()
    f = d.set_index("gnaf_id")
    for c in ("x", "y", "mb", "floor", "bathrooms"):
        A[c] = A["gnaf_id"].map(f[c]) if c in f.columns else np.nan
    A["bedrooms"] = A["bedrooms"].where(A["bedrooms"].notna(), A["gnaf_id"].map(f["bedrooms"]))
    A["car_space"] = A["car_space"].where(A["car_space"].notna(), A["gnaf_id"].map(f["car_space"]))
    A["car_spaces"] = A["car_spaces"].where(A["car_spaces"].notna(), A["gnaf_id"].map(f["car_spaces"]))
    # building-only rows: building centroid / mesh block from the input flats, else advert lat/lng
    b = d.groupby("bkey").agg(bx=("x", "mean"), by=("y", "mean"), bmb=("mb", lambda s: s.mode().iat[0] if s.notna().any() else None))
    A["x"] = A["x"].fillna(A["bkey"].map(b["bx"]))
    A["y"] = A["y"].fillna(A["bkey"].map(b["by"]))
    A["mb"] = A["mb"].where(A["mb"].notna(), A["bkey"].map(b["bmb"]))
    if A["x"].isna().any() and A["lat"].notna().any():
        lx, ly = to_xy(A["lat"], A["lng"])
        A["x"], A["y"] = A["x"].fillna(pd.Series(lx, index=A.index)), A["y"].fillna(pd.Series(ly, index=A.index))
    sub_xy = d.groupby("suburb_clean")[["x", "y"]].mean()
    A["x"] = A["x"].fillna(A["suburb_clean"].map(sub_xy["x"])).fillna(d["x"].mean())
    A["y"] = A["y"].fillna(A["suburb_clean"].map(sub_xy["y"])).fillna(d["y"].mean())
    A["bed_group"] = bed_group(A["bedrooms"])
    A["in_input"] = A["gnaf_id"].isin(set(d["gnaf_id"]))
    return A


def curate(d, sup, sources="supplied", late_api=None):
    """Merge sources, match, campaigns, attributes. Returns adverts table + info for the report."""
    info = {}
    if sources == "merged":
        api = load_api_leases()
        if api is not None:
            api = guard.admit(api, "advert_date", C.CUTOFF, "api")
            if (late_api or C.LATE_API) == "drop":
                api = api[api["advert_date"] < pd.Timestamp(C.LATE_API_FROM)]
            L, mi = merge_leases(sup, api)
            info["merge"] = mi
        else:
            log("no API cache found - using supplied adverts only")
            L = sup.copy()
    else:
        L = sup.copy()
    L, counts = match_leases(L, d)
    L = L[L["match"] != "none"]
    L, n_collapsed = make_campaigns(L)
    info.update({"match_counts": counts, "relists_collapsed": n_collapsed, "adverts": int(len(L)),
                 "by_source": L["src"].value_counts().to_dict()})
    return attach_attributes(L, d), info


# --------------------------------------------------------------------------- training
class Fitted:
    pass


def parking_premium(P):
    """Log parking premium per bedroom group, measured WITHIN the same building (T6.4, T10.2):
    median over buildings of (median log rent with parking - median without), same bedroom group."""
    p = P[P["car_space"].notna() & P["bkey"].notna()]
    g = p.groupby(["bkey", "bed_group", "car_space"])["la"].median().unstack("car_space")
    out = {}
    if 0.0 in g.columns and 1.0 in g.columns:
        diff = (g[1.0] - g[0.0]).dropna()
        overall = float(diff.median()) if len(diff) else 0.08
        for bg, v in diff.groupby(level="bed_group"):
            out[float(bg)] = float(v.median()) if len(v) >= 20 else overall
        out["overall"], out["n_buildings"] = overall, int(diff.index.get_level_values(0).nunique())
    else:
        out["overall"], out["n_buildings"] = 0.08, 0
    for bg in range(C.BED_GROUPS_MAX + 1):
        out[float(bg)] = max(0.0, out.get(float(bg), out["overall"]))
    return out


def train(d, A, pool_end_excl, use_views=None, verbose=True):
    """Use only adverts dated before pool_end_excl; express rents in dollars of the day before it."""
    ref = pool_end_excl - pd.Timedelta(days=1)
    Acut = guard.admit(A, "advert_date", ref, "train")
    idx = build_rent_index(Acut, ref)
    P = Acut.copy()
    P["t"] = P["advert_date"]
    P["la"] = np.log(P["rent"].to_numpy()) + idx.log_at([ref], "flat")[0] - idx.log_at(P["t"], "flat")
    P = P.sort_values(["gnaf_id", "t"]).reset_index(drop=True)
    P["n_before"] = P.groupby("gnaf_id").cumcount()
    fallback = float(np.median(P["la"]))
    tr = P[P["in_input"] & (P["t"] >= pd.Timestamp(C.TRAIN_FROM))].copy()
    T = tr[["gnaf_id", "t", "bedrooms", "car_space", "car_spaces"]].merge(
        d.drop(columns=["bedrooms", "car_space", "car_spaces"]), on="gnaf_id", how="left")
    T["gid"] = T["gnaf_id"]
    if verbose:
        log(f"index: {idx.kind}, {idx.n_pairs:,} repeat pairs, last 12 months {idx.change_last_12m():+.1%}; "
            f"training rows {len(T):,} from {len(P):,} adverts")
    F = make_features(T, P, T["t"], fallback, use_views=use_views)
    cols = feature_list(d, use_views)
    model = RentModel(cols).fit(F, tr["la"].to_numpy(), (ref - tr["t"]).dt.days.to_numpy() / 365.25,
                                tr["campaign_n"].to_numpy())
    fit = Fitted()
    fit.premium = parking_premium(P)
    fit.ref, fit.P, fit.idx, fit.model, fit.fallback, fit.use_views, fit.n_train = ref, P, idx, model, fallback, use_views, len(T)
    return fit


def predict(fit, T, target_date=None):
    """Both parking columns for target flats. Returns (F, rent_with, rent_without, anchor-based baselines)."""
    T = T.reset_index(drop=True).copy()
    T["gid"] = T["gnaf_id"]
    asof = np.full(len(T), fit.ref + pd.Timedelta(days=1), dtype="datetime64[ns]")
    F = make_features(T, fit.P, asof, fit.fallback, use_views=fit.use_views)
    shift = 0.0
    if target_date is not None and C.INDEX_END_MODE == "trend":
        shift = float(fit.idx.log_at([target_date], "trend")[0] - fit.idx.log_at([fit.ref], "flat")[0])
    # Predict at the flat's own known parking (its history already reflects it), then derive the other
    # column with the within-building premium. Unknown parking: model at "without" plus premium. (T6.1-T6.5)
    cs = pd.to_numeric(T["car_space"], errors="coerce").to_numpy()
    prem = np.array([fit.premium[float(b)] if np.isfinite(b) else fit.premium["overall"] for b in F["bed_group"]])
    la1 = fit.model.predict_la(with_parking(F, 1))
    la0 = fit.model.predict_la(with_parking(F, 0))
    base_wo = np.where(cs == 1, la1 - prem, la0)
    wo = np.exp(base_wo + shift)
    w = np.exp(base_wo + prem + shift)                                    # T1.5 holds since prem >= 0
    return F, w, wo, shift


# --------------------------------------------------------------------------- metrics
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(p) & (p > 0)
    if not ok.any():
        return {"n": int(len(y)), "n_missing": int(len(y)), "mdae": np.nan, "within100": np.nan, "r2": np.nan, "bias": np.nan}
    e = p[ok] - y[ok]
    r2 = 1 - (e ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum() if ok.sum() > 1 else np.nan
    return {"n": int(len(y)), "n_missing": int((~ok).sum()), "mdae": float(np.median(np.abs(e))),
            "within100": float(np.mean(np.abs(e) <= C.SCORE_WITHIN)), "r2": float(r2), "bias": float(np.median(e))}


def scored_column(te, flats_cs, rw, rwo):
    """T3.2: advert parking first, then the input's car_space; excluded if both unknown."""
    cs = te["adv_car_space"].where(te["adv_car_space"].notna(), flats_cs)
    pred = np.where(cs == 1, rw, np.where(cs == 0, rwo, np.nan))
    return pred, cs.notna().to_numpy()


def append_result(name, fold, res, extra=None):
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "fold": fold, **{k: res.get(k) for k in ("mdae", "within100", "r2", "bias", "n")}}
    if extra:
        row.update(extra)
    with open(C.OUT_DIR / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------- evaluation
def run_validation(d, A, train_end, test_range, label="tune", use_views=None, log_results=True):
    split = pd.Timestamp(train_end)
    t0, t1 = pd.Timestamp(test_range[0]), pd.Timestamp(test_range[1])
    fit = train(d, A, split, use_views)
    te = A[A["in_input"] & (A["advert_date"] >= t0) & (A["advert_date"] <= t1)].copy()
    te = te.rename(columns={"car_space": "adv_car_space"})
    T = d[d["gnaf_id"].isin(te["gnaf_id"].unique())]
    F, rw, rwo, shift = predict(fit, T, t0 + (t1 - t0) / 2)
    F.index = T["gnaf_id"].to_numpy()
    rw, rwo = pd.Series(rw, index=F.index), pd.Series(rwo, index=F.index)
    fcs = te["gnaf_id"].map(d.set_index("gnaf_id")["car_space"])
    pred, eligible = scored_column(te, fcs, te["gnaf_id"].map(rw).to_numpy(), te["gnaf_id"].map(rwo).to_numpy())
    te["pred"], te["eligible"] = pred, eligible
    te = te[te["eligible"]].copy()
    # baselines on the same adverts (all built only from adverts before the split)
    hist = fit.P
    te["b_suburb"] = suburb_median(hist, te)
    te["b_subbed"] = suburb_bedroom_median(hist, te)
    te["b_sample"] = sample_method(hist, te, split - pd.Timedelta(days=365), split)
    te["b_ownlast"] = own_last(F, te, fit.fallback, te["b_subbed"])
    te["had_prior"] = te["gnaf_id"].map(F["la_prev"].notna()).to_numpy()
    first = te["campaign_first"].to_numpy()
    y = te["rent"].to_numpy()
    res = {"label": label, "train_end": str(split.date()), "test": [str(t0.date()), str(t1.date())],
           "n_train_rows": fit.n_train, "index_kind": fit.idx.kind, "index_pairs": fit.idx.n_pairs,
           "index_last12m": fit.idx.change_last_12m(), "trend_shift": shift, "views": fit.use_views}
    for name, col in (("model", "pred"), ("suburb median", "b_suburb"), ("suburb x bedrooms", "b_subbed"),
                      ("sample method", "b_sample"), ("own last rent x index", "b_ownlast")):
        res[name] = {"all": metrics(y, te[col]), "one_per_campaign": metrics(y[first], te[col].to_numpy()[first])}
    m = te["had_prior"].to_numpy()
    res["model | prior advert"] = metrics(y[m], te["pred"].to_numpy()[m])
    res["model | no prior advert"] = metrics(y[~m], te["pred"].to_numpy()[~m])
    # blend with own-last if the model does not beat it (T4.5)
    best_w, best = 1.0, res["model"]["all"]["mdae"]
    if res["own last rent x index"]["all"]["mdae"] <= best:
        lp, lb = np.log(te["pred"].to_numpy()), np.log(te["b_ownlast"].to_numpy())
        for w in np.round(np.arange(0, 1.01, 0.1), 1):
            mm = metrics(y, np.exp(w * lp + (1 - w) * lb))["mdae"]
            if mm < best:
                best_w, best = float(w), mm
    res["blend_weight"], res["blend_mdae"] = best_w, best
    res["bias_by_bed"] = te.assign(e=te["pred"] - te["rent"]).groupby("bed_group")["e"].median().round(1).to_dict()
    res["anchor_src"] = {ANCHOR_NAMES[int(k)]: int(v) for k, v in F["anchor_src"].value_counts().sort_index().items()}
    prem = (rw - rwo)
    res["premium_median"] = float(prem.median())
    if log_results:
        append_result("model", label, res["model"]["all"], {"one_per_campaign_mdae": res["model"]["one_per_campaign"]["mdae"]})
        for b in ("suburb median", "suburb x bedrooms", "sample method", "own last rent x index"):
            append_result(b, label, res[b]["all"])
    return res, te, fit, F


def print_validation(res):
    def row(n, m):
        a, o = m["all"], m["one_per_campaign"]
        return (f"  {n:<24} MdAE ${a['mdae']:6.1f}  within$100 {a['within100']:5.1%}  R2 {a['r2']:.3f}  bias {a['bias']:+6.1f}"
                f"  n {a['n']:5d} | one/campaign MdAE ${o['mdae']:6.1f} n {o['n']}")
    print(f"\n{res['label']}: train before {res['train_end']}, test {res['test'][0]}..{res['test'][1]}; "
          f"index {res['index_kind']} ({res['index_pairs']} pairs, 12m {res['index_last12m']:+.1%}), end {C.INDEX_END_MODE}, shift {res['trend_shift']:+.3f}")
    for n in ("suburb median", "suburb x bedrooms", "sample method", "own last rent x index", "model"):
        print(row(n, res[n]))
    print(f"  model | prior advert    MdAE ${res['model | prior advert']['mdae']:.1f} (n {res['model | prior advert']['n']}); "
          f"no prior MdAE ${res['model | no prior advert']['mdae']:.1f} (n {res['model | no prior advert']['n']})")
    print(f"  blend weight {res['blend_weight']} (1.0 = model only) -> MdAE ${res['blend_mdae']:.1f}")
    print(f"  bias by bedroom group {res['bias_by_bed']}; median parking premium ${res['premium_median']:.0f}")
    print(f"  anchor sources {res['anchor_src']}\n")


# --------------------------------------------------------------------------- calibration (T3.5)
def calibrate(sup_adverts, d):
    A = sup_adverts
    tr = A[(A["advert_date"] >= pd.Timestamp(C.SAMPLE_TRAIN[0])) & (A["advert_date"] <= pd.Timestamp(C.SAMPLE_TRAIN[1]))]
    te = A[(A["advert_date"] >= pd.Timestamp(C.SAMPLE_TEST[0])) & (A["advert_date"] <= pd.Timestamp(C.SAMPLE_TEST[1]))]
    tr = tr[tr["bedrooms"].notna() & tr["car_space"].notna()]
    te = te[te["bedrooms"].notna() & te["car_space"].notna()].copy()
    pred = sample_method(tr, te, None, None)
    m = metrics(te["rent"], pred)
    ref = C.SAMPLE_REF
    checks = {"mdae": abs(m["mdae"] - ref["mdae"]) <= 3, "within100": abs(m["within100"] - ref["within100"]) <= 0.02,
              "r2": abs(m["r2"] - ref["r2"]) <= 0.03, "n": abs(m["n"] - ref["n"]) <= 0.02 * ref["n"]}
    return m, checks


# --------------------------------------------------------------------------- final
def run_final(d, flats_raw, A, team, outdir=None, blend_weight=None):
    from submission import assemble
    outdir = Path(outdir or C.OUT_DIR)
    fit = train(d, A, pd.Timestamp(C.CUTOFF) + pd.Timedelta(days=1))
    F, rw, rwo, shift = predict(fit, d, pd.Timestamp("2026-08-31"))
    bw = C.BLEND_WEIGHT if blend_weight is None else blend_weight
    if bw is not None and bw < 1.0:
        base = own_last(F.set_index(d["gnaf_id"].to_numpy()), pd.DataFrame({"gnaf_id": d["gnaf_id"]}), fit.fallback,
                        None, d=d)
        # blend in log space, keeping the model's parking premium
        prem = np.log(rw) - np.log(rwo)
        lb = np.log(base)
        lo = bw * np.log(rwo) + (1 - bw) * (lb - np.where(d["car_space"].to_numpy() == 1, prem, 0))
        rwo, rw = np.exp(lo), np.exp(lo + prem)
    train_rents = fit.P["rent"]
    path = assemble(flats_raw, d["gnaf_id"].to_numpy(), rw, rwo, team, outdir, train_rents)
    counts = {ANCHOR_NAMES[int(k)]: int(v) for k, v in F["anchor_src"].value_counts().sort_index().items()}
    log(f"wrote {path}; anchor sources {counts}; trend shift {shift:+.3f}")
    return path, fit, F, rw, rwo
