"""Export NSL-KDD with 5 attack-family labels to ARFF, so the original Weka experiment can be re-run.

    python weka/export_arff.py         # writes data/train_family.arff and data/test_family.arff
    bash weka/run_weka.sh /path/to/weka.jar
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nids.data import CATEGORICAL, CLASSES, FEATURES, download, load_split  # noqa: E402


def write_arff(X, y, path, nominal):
    with open(path, "w") as f:
        f.write("@relation nsl_kdd_family\n\n")
        for c in FEATURES:
            f.write(f"@attribute {c} {{{','.join(nominal[c])}}}\n" if c in CATEGORICAL else f"@attribute {c} numeric\n")
        f.write(f"@attribute family {{{','.join(CLASSES)}}}\n\n@data\n")
        for row, label in zip(X.itertuples(index=False), y):
            f.write(",".join(str(v) for v in row) + f",{label}\n")


if __name__ == "__main__":
    paths = download(ROOT / "data")
    (Xtr, ytr), (Xte, yte) = load_split(paths["train"]), load_split(paths["test"])
    nominal = {c: sorted(set(Xtr[c]) | set(Xte[c])) for c in CATEGORICAL}
    write_arff(Xtr, ytr, ROOT / "data/train_family.arff", nominal)
    write_arff(Xte, yte, ROOT / "data/test_family.arff", nominal)
    print("Wrote data/train_family.arff and data/test_family.arff")
