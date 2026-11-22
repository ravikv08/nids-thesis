"""Metrics used in the thesis (detection rate) plus standard ones."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from .data import CLASSES


def ids_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    attack, flagged = y_true != "normal", y_pred != "normal"
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        # Detection rate: share of real attacks flagged as any attack (the thesis's headline metric).
        "detection_rate": flagged[attack].mean(),
        # False alarm rate: share of normal traffic wrongly flagged as an attack.
        "false_alarm_rate": flagged[~attack].mean(),
        "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
    }
    rec = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    out.update({f"recall_{c}": r for c, r in zip(CLASSES, rec)})
    return out


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=CLASSES)
