"""Preprocessing and per-class information-gain feature selection (thesis step 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

from .data import CATEGORICAL, FEATURES


class KDDEncoder(BaseEstimator, TransformerMixin):
    """Ordinal-encode the 3 symbolic fields (protocol, service, flag); unseen values map to -1.

    Decision trees split on thresholds, so ordinal codes are enough and keep the 41 original features.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.enc_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(X[CATEGORICAL])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X[FEATURES].copy()
        X[CATEGORICAL] = self.enc_.transform(X[CATEGORICAL])
        return X.to_numpy(dtype=float)


def information_gain(X: np.ndarray, y_binary: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Information gain of each feature for a one-vs-rest target (mutual information, in nats)."""
    discrete = np.zeros(X.shape[1], dtype=bool)
    discrete[[FEATURES.index(c) for c in CATEGORICAL]] = True
    return mutual_info_classif(X, y_binary, discrete_features=discrete, random_state=random_state)


def top_features_per_class(X: np.ndarray, y: np.ndarray, classes, k: int = 15, sample: int = 40000, random_state: int = 0):
    """For each class, rank features by information gain against 'this class vs the rest' and keep the top k.

    Low-frequency classes (U2R, R2L) get their own feature subsets, which is the point of the thesis's
    per-class selection: features that matter for rare attacks are not drowned out by DoS traffic.
    """
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(y))
    if len(idx) > sample:
        # Keep every rare-class row, subsample the rest, so MI estimates stay fast but still see rare attacks.
        rare = np.isin(y, [c for c in classes if np.mean(y == c) < 0.02])
        idx = np.concatenate([idx[rare], rng.choice(idx[~rare], size=sample - rare.sum(), replace=False)])
    out = {}
    for c in classes:
        gain = information_gain(X[idx], (y[idx] == c).astype(int), random_state)
        order = np.argsort(gain)[::-1][:k]
        out[c] = {"idx": order, "names": [FEATURES[i] for i in order], "gain": gain[order]}
    return out
