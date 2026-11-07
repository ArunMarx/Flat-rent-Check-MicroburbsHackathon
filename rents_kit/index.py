"""Monthly citywide rent index (US9). Repeat-advert pairs of the same flat, robust penalised least squares;
mix-adjusted monthly median when repeat pairs are scarce (T9.1). Adapted from the valuation kit."""
import warnings

import numpy as np
import pandas as pd

import config as C


def month_number(dates):
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    return np.asarray((d.year - 2000) * 12 + (d.month - 1))


class RentIndex:
    def __init__(self, m0, levels, n_pairs, kind):
        self.m0, self.levels, self.n_pairs, self.kind = int(m0), np.asarray(levels, float), int(n_pairs), kind
        self.slope = 0.0
        if len(self.levels) > C.TREND_MONTHS:
            self.slope = float((self.levels[-1] - self.levels[-1 - C.TREND_MONTHS]) / C.TREND_MONTHS)

    def log_at(self, dates, mode=None):
        mode = mode or C.INDEX_END_MODE
        m = month_number(dates) - self.m0
        last = len(self.levels) - 1
        base = self.levels[np.clip(m, 0, last)]
        if mode == "trend":
            base = base + np.clip(m - last, 0, None) * self.slope
        return base

    def table(self):
        start = pd.Period("2000-01", freq="M") + self.m0
        months = pd.period_range(start=start, periods=len(self.levels), freq="M")
        return pd.DataFrame({"month": months.astype(str), "index_100": 100 * np.exp(self.levels - self.levels[-1])})

    def change(self, d1, d2):
        return float(np.exp(self.log_at([d2], "flat")[0] - self.log_at([d1], "flat")[0]) - 1)

    def change_last_12m(self):
        return float(np.exp(self.levels[-1] - self.levels[-13]) - 1) if len(self.levels) >= 13 else np.nan


def pairs_from_adverts(A):
    """Consecutive campaigns (first advert of each) of the same flat."""
    a = A[A["gnaf_id"].notna() & A["campaign_first"]].sort_values(["gnaf_id", "advert_date"])
    g = a.groupby("gnaf_id")
    d1, r1 = g["advert_date"].shift(), g["rent"].shift()
    ok = d1.notna()
    out = pd.DataFrame({"d1": d1[ok].to_numpy(), "r1": r1[ok].to_numpy(),
                        "d2": a["advert_date"][ok].to_numpy(), "r2": a["rent"][ok].to_numpy()})
    gap = (pd.to_datetime(out["d2"]) - pd.to_datetime(out["d1"])).dt.days
    lr = np.log(out["r2"] / out["r1"])
    keep = (gap >= C.PAIR_MIN_GAP_DAYS) & (lr.abs() <= C.PAIR_MAX_ABS_LOG)
    out = out[keep.to_numpy()].reset_index(drop=True)
    out["gap_days"], out["lr"] = gap[keep].to_numpy(), lr[keep].to_numpy()
    return out


def _penalty(T, lam2, lam1):
    D1 = np.diff(np.eye(T), axis=0)
    D2 = np.diff(np.eye(T), 2, axis=0)
    return lam2 * (D2.T @ D2) + lam1 * (D1.T @ D1) + 1e-6 * np.eye(T)


def build_rent_index(A, last_date, lam2=None, lam1=None, force_fallback=False):
    """A: adverts (gnaf_id, advert_date, rent, campaign_first, suburb_clean, bed_group), already cut at last_date."""
    lam2 = C.INDEX_LAM2 if lam2 is None else lam2
    lam1 = C.INDEX_LAM1 if lam1 is None else lam1
    m0 = int(month_number([A["advert_date"].min()])[0])
    T = int(month_number([last_date])[0]) - m0 + 1
    if T < 3:
        return RentIndex(m0, np.zeros(max(T, 1)), 0, "flat")
    P = _penalty(T, lam2, lam1)
    pairs = pairs_from_adverts(A)
    if len(pairs) >= C.INDEX_MIN_PAIRS and not force_fallback:
        a, b = month_number(pairs["d1"]) - m0, month_number(pairs["d2"]) - m0
        y = pairs["lr"].to_numpy()
        w0 = 1.0 / (1.0 + pairs["gap_days"].to_numpy() / (365.0 * 3))
        w = w0.copy()
        for _ in range(3):
            M = np.zeros((T, T))
            r = np.zeros(T)
            np.add.at(M, (a, a), w)
            np.add.at(M, (b, b), w)
            np.add.at(M, (a, b), -w)
            np.add.at(M, (b, a), -w)
            np.add.at(r, a, -w * y)
            np.add.at(r, b, w * y)
            lev = np.linalg.solve(M + P, r)
            res = y - (lev[b] - lev[a])
            w = w0 / (1.0 + (res / 0.10) ** 2)
        return RentIndex(m0, lev - lev[-1], len(pairs), "repeat_adverts")

    warnings.warn(f"Only {len(pairs)} repeat pairs - using mix-adjusted monthly median index")
    a = A[A["campaign_first"]].copy()
    grp = a["suburb_clean"].astype(str) + "|" + a["bed_group"].astype(str)
    lr = np.log(a["rent"].to_numpy())
    rel = lr - pd.Series(lr).groupby(grp.to_numpy()).transform("median").to_numpy()
    m = month_number(a["advert_date"]) - m0
    med = pd.Series(rel).groupby(m).median()
    cnt = pd.Series(rel).groupby(m).size()
    mu, c = np.zeros(T), np.zeros(T)
    mu[med.index.to_numpy()] = med.to_numpy()
    c[cnt.index.to_numpy()] = cnt.to_numpy()
    lev = np.linalg.solve(np.diag(c) + P, c * mu)
    return RentIndex(m0, lev - lev[-1], len(pairs), "mix_adjusted_fallback")
