import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids.data import ATTACK_FAMILY, CLASSES  # noqa: E402
from nids.evaluate import ids_metrics  # noqa: E402
from nids.multiboost import MultiBoostClassifier  # noqa: E402


def test_multiboost_beats_chance_and_builds_committee():
    X, y = make_classification(n_samples=600, n_features=10, n_informative=5, n_classes=3, random_state=0)
    m = MultiBoostClassifier(n_estimators=9, random_state=0).fit(X[:400], y[:400])
    assert 1 <= len(m.estimators_) <= 9
    assert (m.predict(X[400:]) == y[400:]).mean() > 0.6
    p = m.predict_proba(X[400:])
    assert np.allclose(p.sum(axis=1), 1)


def test_multiboost_is_reproducible():
    X, y = make_classification(n_samples=300, random_state=1)
    a = MultiBoostClassifier(n_estimators=6, random_state=7).fit(X, y).predict(X)
    b = MultiBoostClassifier(n_estimators=6, random_state=7).fit(X, y).predict(X)
    assert (a == b).all()


def test_attack_families_cover_all_classes():
    assert set(ATTACK_FAMILY.values()) == set(CLASSES)


def test_ids_metrics():
    y = np.array(["normal", "normal", "DoS", "R2L"])
    p = np.array(["normal", "DoS", "DoS", "normal"])
    m = ids_metrics(y, p)
    assert m["detection_rate"] == 0.5        # 1 of 2 attacks flagged
    assert m["false_alarm_rate"] == 0.5      # 1 of 2 normals flagged
    assert m["accuracy"] == 0.5
