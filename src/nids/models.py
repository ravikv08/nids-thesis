"""Models compared in this repo: the thesis model plus the baselines it was compared against."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from .features import top_features_per_class
from .multiboost import MultiBoostClassifier


def _log1p(a):
    return np.log1p(np.maximum(a, 0))


def c45(random_state=0, **kw):
    """C4.5-style tree: entropy (information-gain) splits with a minimum leaf size as light pruning.

    scikit-learn grows CART-style binary trees; with criterion='entropy' this is the closest built-in match to
    C4.5 / Weka J48 (which uses gain ratio and error-based pruning).
    """
    return DecisionTreeClassifier(criterion="entropy", min_samples_leaf=kw.pop("min_samples_leaf", 5), random_state=random_state, **kw)


class PerClassMultiBoostNIDS(ClassifierMixin, BaseEstimator):
    """The thesis model.

    1. Per-class feature selection by information gain (top-k features for each attack family).
    2. One binary C4.5 classifier per class ("this class vs the rest"), each trained with MultiBoosting.
    3. Combine: every binary committee votes for its class; the class with the strongest vote wins.
    """

    def __init__(self, k_features=15, n_estimators=20, random_state=0):
        self.k_features = k_features
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        self.selected_ = top_features_per_class(X, y, self.classes_, k=self.k_features, random_state=self.random_state)
        self.committees_ = {}
        for c in self.classes_:
            cols = self.selected_[c]["idx"]
            m = MultiBoostClassifier(c45(self.random_state), n_estimators=self.n_estimators, random_state=self.random_state)
            self.committees_[c] = m.fit(X[:, cols], (y == c).astype(int))
        return self

    def decision_scores(self, X):
        X = np.asarray(X)
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for j, c in enumerate(self.classes_):
            m = self.committees_[c]
            p = m.predict_proba(X[:, self.selected_[c]["idx"]])
            scores[:, j] = p[:, list(m.classes_).index(1)] if 1 in m.classes_ else 0.0
        return scores

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_scores(X), axis=1)]


def build_models(n_estimators=20, random_state=0, k_features=15):
    """Name -> unfitted model. Order = how they appear in the results table."""
    T = n_estimators
    return {
        # log1p tames the heavy-tailed byte counts, which otherwise break the Gaussian assumption.
        "Naive Bayes (classic baseline)": make_pipeline(FunctionTransformer(_log1p), GaussianNB()),
        "C4.5 decision tree": c45(random_state),
        "Bagging (C4.5)": BaggingClassifier(c45(random_state), n_estimators=T, random_state=random_state, n_jobs=-1),
        "AdaBoost (C4.5)": AdaBoostClassifier(c45(random_state, max_depth=10), n_estimators=T, random_state=random_state),
        "MultiBoosting (C4.5)": MultiBoostClassifier(c45(random_state), n_estimators=T, random_state=random_state),
        "Thesis: per-class IG + C4.5 + MultiBoosting": PerClassMultiBoostNIDS(k_features=k_features, n_estimators=T, random_state=random_state),
        "Random Forest (modern baseline)": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "Gradient Boosting (modern baseline)": HistGradientBoostingClassifier(random_state=random_state),
    }
