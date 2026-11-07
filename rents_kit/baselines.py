"""Baselines (design.md sec. 8.6): suburb median, suburb x bedrooms, the sample report's method (T3.5, T4.2)
and own last rent x index (T4.5). All use only adverts before the split."""
import numpy as np
import pandas as pd

import config as C


def _recent(hist, months=24):
    end = hist["t"].max()
    return hist[hist["t"] > end - pd.DateOffset(months=months)]


def suburb_median(hist, te):
    h = _recent(hist)
    m = np.exp(h.groupby("suburb_clean")["la"].median())
    return te["suburb_clean"].map(m).fillna(float(np.exp(h["la"].median()))).to_numpy()


def suburb_bedroom_median(hist, te):
    h = _recent(hist)
    key_h = h["suburb_clean"].astype(str) + "|" + h["bed_group"].astype(str)
    key_t = te["suburb_clean"].astype(str) + "|" + te["bed_group"].astype(str)
    m = np.exp(h.groupby(key_h)["la"].median())
    return pd.Series(key_t.map(m).to_numpy()).fillna(pd.Series(suburb_median(hist, te))).to_numpy()


def sample_method(tr, te, date_from, date_to):
    """Median raw rent by suburb x bedrooms x parking (groups of >= 10), then suburb x bedrooms, then bedrooms."""
    t = tr
    if date_from is not None:
        t = t[(t["advert_date"] >= date_from) & (t["advert_date"] < date_to)]
    t = t[t["bed_group"].notna() & t["car_space"].notna()]
    park_t = te["adv_car_space"] if "adv_car_space" in te.columns else te["car_space"]
    keys = [["suburb_clean", "bed_group", "car_space"], ["suburb_clean", "bed_group"], ["bed_group"]]
    out = pd.Series(np.nan, index=te.index)
    tt = te.assign(car_space=park_t)
    for k in keys:
        g = t.groupby(k)["rent"].agg(["median", "size"])
        g = g[g["size"] >= C.SAMPLE_MIN_GROUP]["median"]
        idx = pd.MultiIndex.from_frame(tt[k]) if len(k) > 1 else tt[k[0]]
        vals = pd.Series(g.reindex(idx).to_numpy(), index=te.index)
        out = out.fillna(vals)
    return out.fillna(t["rent"].median()).to_numpy()


def own_last(F, te, fallback_la, fallback_vals=None, d=None):
    """The flat's last advert (<= 3 years old) restated with the index; else suburb x bedrooms (T4.5)."""
    la = F["la_prev"].where(F["prev_age"] <= C.OWN_LAST_MAX_AGE_YEARS)
    v = te["gnaf_id"].map(np.exp(la)).to_numpy(float)
    if fallback_vals is None:
        fb = np.exp(F["subbed_mean"].fillna(F["sub_mean"]).fillna(fallback_la))
        fallback_vals = te["gnaf_id"].map(fb).to_numpy(float)
    return np.where(np.isfinite(v), v, np.asarray(fallback_vals, float))
