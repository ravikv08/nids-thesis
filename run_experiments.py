"""Train every model and compare them on NSL-KDD.

Two evaluations:
  A. "Known attacks"  - stratified 80/20 split of KDDTrain+ (same attack types in train and test, like the
                         thesis's KDD Cup 99 setting).
  B. "New attacks"    - train on all of KDDTrain+, test on KDDTest+, which contains 17 attack types never seen
                         in training. This is the harder, more realistic test.

Usage:  python run_experiments.py [--estimators 20] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))
from nids.data import CLASSES, load  # noqa: E402
from nids.evaluate import confusion, ids_metrics  # noqa: E402
from nids.features import KDDEncoder  # noqa: E402
from nids.models import build_models  # noqa: E402

OUT = Path(__file__).parent / "results"


def run(setting, X_tr, y_tr, X_te, y_te, models):
    enc = KDDEncoder().fit(X_tr)
    A_tr, A_te = enc.transform(X_tr), enc.transform(X_te)
    rows, preds = [], {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(A_tr, y_tr.to_numpy())
        fit_s = time.time() - t0
        p = model.predict(A_te)
        m = ids_metrics(y_te, p)
        rows.append({"setting": setting, "model": name, **m, "train_seconds": fit_s})
        preds[name] = p
        print(f"[{setting}] {name:48s} acc={m['accuracy']:.4f} DR={m['detection_rate']:.4f} "
              f"FAR={m['false_alarm_rate']:.4f} F1={m['macro_f1']:.3f} ({fit_s:.0f}s)", flush=True)
    return rows, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimators", type=int, default=20, help="boosting/bagging rounds T")
    ap.add_argument("--quick", action="store_true", help="use 20%% of the training data (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X_tr, y_tr, X_te, y_te = load(Path(__file__).parent / "data")
    if args.quick:
        X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=0.2, stratify=y_tr, random_state=args.seed)
    print(f"KDDTrain+: {len(y_tr):,} rows  |  KDDTest+: {len(y_te):,} rows")
    print("Train class counts:", y_tr.value_counts().to_dict())

    OUT.mkdir(exist_ok=True)
    all_rows, conf = [], {}

    # A. Known attacks
    a_tr, a_te, b_tr, b_te = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=args.seed)
    rows, preds = run("A. Known attacks (KDDTrain+ 80/20)", a_tr, b_tr, a_te, b_te, build_models(args.estimators, args.seed))
    all_rows += rows
    conf["A"] = {k: confusion(b_te, v).tolist() for k, v in preds.items()}

    # B. New attacks
    models = build_models(args.estimators, args.seed)
    rows, preds = run("B. New attacks (KDDTrain+ -> KDDTest+)", X_tr, y_tr, X_te, y_te, models)
    all_rows += rows
    conf["B"] = {k: confusion(y_te, v).tolist() for k, v in preds.items()}

    thesis = models["Thesis: per-class IG + C4.5 + MultiBoosting"]
    feats = {c: list(zip(v["names"], [round(float(g), 4) for g in v["gain"]])) for c, v in thesis.selected_.items()}

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "metrics.csv", index=False)
    json.dump({"classes": CLASSES, "confusion": conf, "top_features": feats, "estimators": args.estimators,
               "quick": args.quick, "seed": args.seed}, open(OUT / "details.json", "w"), indent=1)
    print(f"\nSaved {OUT/'metrics.csv'} and {OUT/'details.json'}. Run `python make_report.py` for tables and charts.")


if __name__ == "__main__":
    main()
