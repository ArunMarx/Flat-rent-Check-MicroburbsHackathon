"""Gradient-boosted correction model around the anchor (design.md sec. 8.5). scikit-learn only."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import config as C
from features import CATEGORICAL


class RentModel:
    """target='resid': learn log rent - anchor. loss='absolute_error' targets the median (scored metric).
    car_space has a +1 monotone constraint so adding parking can never lower the estimate (T6.2)."""

    def __init__(self, cols, loss=None, target=None, halflife=None, seeds=None):
        self.cols = list(cols) + ["anchor_src"]
        self.loss = loss or C.LOSS
        self.target = target or C.TARGET
        self.halflife = C.HALFLIFE_YEARS if halflife is None else halflife
        self.seeds = C.SEEDS if seeds is None else seeds

    def fit(self, F, la, yrs_old, campaign_n):
        y = la - F["anchor"].to_numpy() if self.target == "resid" else la
        w = 1.0 / np.asarray(campaign_n, float)
        if self.halflife and self.halflife > 0:
            w = w * 0.5 ** (np.clip(np.asarray(yrs_old, float), 0, None) / self.halflife)
        cat = [i for i, c in enumerate(self.cols) if c in CATEGORICAL]
        mono = [1 if c == "car_space" else 0 for c in self.cols]
        usable = [c for c in self.cols if F[c].nunique(dropna=True) >= 2 or c in ("car_space", "anchor_src")]
        dropped = sorted(set(self.cols) - set(usable))
        if dropped:
            print(f"  model: skipping constant/empty features {dropped}")
        self.cols = usable
        cat = [i for i, c in enumerate(self.cols) if c in CATEGORICAL]
        mono = [1 if c == "car_space" else 0 for c in self.cols]
        X = F[self.cols].to_numpy(float)
        self.models = []
        for s in self.seeds:
            m = HistGradientBoostingRegressor(
                loss=self.loss, max_iter=600, learning_rate=0.05, max_leaf_nodes=48, min_samples_leaf=30,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
                random_state=s, categorical_features=cat or None, monotonic_cst=mono)
            m.fit(X, y, sample_weight=w)
            self.models.append(m)
        self.n_iter_ = [m.n_iter_ for m in self.models]
        return self

    def predict_la(self, F):
        X = F[self.cols].to_numpy(float)
        p = np.mean([m.predict(X) for m in self.models], axis=0)
        return F["anchor"].to_numpy() + p if self.target == "resid" else p
