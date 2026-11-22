"""MultiBoosting (Webb, 2000): AdaBoost combined with wagging.

Faithful to the algorithm in G. I. Webb, "MultiBoosting: A Technique for Combining Boosting and Wagging",
Machine Learning 40, 159-196 (2000) - the same algorithm the thesis used through Weka's MultiBoostAB.

AdaBoost reduces bias; wagging (bagging with random continuous-Poisson weights) reduces variance. MultiBoosting
splits the T boosting rounds into ~sqrt(T) sub-committees and resets the instance weights to random Poisson
weights at the start of each sub-committee, so it gets both effects.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


class MultiBoostClassifier(ClassifierMixin, BaseEstimator):
    """MultiBoosting meta-classifier.

    Parameters
    ----------
    estimator : base learner supporting ``sample_weight`` (default: entropy decision tree, i.e. C4.5-style).
    n_estimators : total number of boosting rounds T.
    n_subcommittees : number of wagging sub-committees; ``None`` uses Webb's default ``floor(sqrt(T))``.
    max_retries : how many times a round may be retried after a weak learner with error > 0.5.
    random_state : seed for the Poisson weight resets.
    """

    def __init__(self, estimator=None, n_estimators=20, n_subcommittees=None, max_retries=25, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.n_subcommittees = n_subcommittees
        self.max_retries = max_retries
        self.random_state = random_state

    def _poisson_weights(self, rng, n):
        # Continuous Poisson (exponential) weights, standardised to sum to n, as in Webb (2000).
        w = -np.log(rng.uniform(1e-3, 1.0, size=n))
        return w * (n / w.sum())

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = unique_labels(y)
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        T = int(self.n_estimators)
        n_sub = self.n_subcommittees or max(1, int(math.floor(math.sqrt(T))))
        # Round index at which each sub-committee ends (I_k in the paper).
        boundaries = [math.ceil(i * T / n_sub) for i in range(1, n_sub + 1)]
        base = self.estimator if self.estimator is not None else DecisionTreeClassifier(criterion="entropy", min_samples_leaf=5)

        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float) * (n / np.sum(sample_weight))
        self.estimators_, self.estimator_weights_, self.estimator_errors_ = [], [], []
        t, retries = 0, 0
        while t < T:
            est = clone(base)
            if hasattr(est, "random_state"):
                est.set_params(random_state=int(rng.integers(1 << 31)))
            est.fit(X, y, sample_weight=w)
            wrong = est.predict(X) != y
            eps = float(np.dot(w, wrong) / w.sum())

            if eps > 0.5:
                # Discard this learner, reset to random weights and start a new sub-committee.
                w = self._poisson_weights(rng, n)
                retries += 1
                if retries > self.max_retries:
                    break
                continue
            if eps == 0.0:
                beta = 1e-10
                self.estimators_.append(est); self.estimator_weights_.append(math.log(1 / beta)); self.estimator_errors_.append(eps)
                w = self._poisson_weights(rng, n)
            else:
                beta = eps / (1 - eps)
                self.estimators_.append(est); self.estimator_weights_.append(math.log(1 / beta)); self.estimator_errors_.append(eps)
                # AdaBoost re-weighting: misclassified mass -> 1/2, correct mass -> 1/2.
                w = np.where(wrong, w / (2 * eps), w / (2 * (1 - eps)))
                w = np.maximum(w, 1e-8)
            t += 1
            retries = 0
            # Scheduled sub-committee boundary -> wagging reset.
            if t in boundaries and t < T:
                w = self._poisson_weights(rng, n)
        if not self.estimators_:
            raise RuntimeError("MultiBoosting could not fit any base learner with error <= 0.5")
        return self

    def _votes(self, X):
        check_is_fitted(self, "estimators_")
        X = np.asarray(X)
        idx = {c: i for i, c in enumerate(self.classes_)}
        votes = np.zeros((X.shape[0], len(self.classes_)))
        for est, a in zip(self.estimators_, self.estimator_weights_):
            pred = est.predict(X)
            votes[np.arange(len(pred)), [idx[p] for p in pred]] += a
        return votes

    def predict_proba(self, X):
        v = self._votes(X)
        s = v.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        return v / s

    def predict(self, X):
        return self.classes_[np.argmax(self._votes(X), axis=1)]
