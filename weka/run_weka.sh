#!/usr/bin/env bash
# Re-run the thesis setup in Weka itself (Java): J48 (Weka's C4.5) inside MultiBoostAB, plus baselines.
# Usage: bash weka/run_weka.sh /path/to/weka.jar      (Weka 3.8: https://waikato.github.io/weka-wiki/downloading_weka/)
set -euo pipefail
WEKA_JAR=${1:?"pass the path to weka.jar"}
cd "$(dirname "$0")/.."
[ -f data/train_family.arff ] || python weka/export_arff.py
TR=data/train_family.arff; TE=data/test_family.arff
run() { echo "=== $1"; shift; java -Xmx4g -cp "$WEKA_JAR" "$@" -t "$TR" -T "$TE" -o | sed -n '/Error on test data/,/Kappa/p'; }
run "C4.5 (J48)"                 weka.classifiers.trees.J48
run "AdaBoost.M1 + J48"          weka.classifiers.meta.AdaBoostM1 -I 20 -W weka.classifiers.trees.J48
run "Bagging + J48"              weka.classifiers.meta.Bagging -I 20 -W weka.classifiers.trees.J48
run "MultiBoostAB + J48 (thesis)" weka.classifiers.meta.MultiBoostAB -I 20 -W weka.classifiers.trees.J48
