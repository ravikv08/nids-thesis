"""Turn results/metrics.csv + details.json into results/RESULTS.md and charts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = Path(__file__).parent / "results"
THESIS = "Thesis: per-class IG + C4.5 + MultiBoosting"
pct = lambda x: f"{100 * x:.2f}%"  # noqa: E731


def table(df):
    cols = ["model", "accuracy", "detection_rate", "false_alarm_rate", "macro_f1", "recall_DoS", "recall_Probe", "recall_R2L", "recall_U2R", "train_seconds"]
    head = "| Model | Accuracy | Detection rate | False alarms | Macro-F1 | DoS | Probe | R2L | U2R | Train (s) |"
    lines = [head, "|" + "---|" * len(cols)]
    best = {c: df[c].max() for c in ["accuracy", "detection_rate", "macro_f1"]}
    for _, r in df[cols].iterrows():
        cell = lambda c: (f"**{pct(r[c])}**" if c in best and np.isclose(r[c], best[c]) else pct(r[c]))  # noqa: E731
        f1 = f"**{r.macro_f1:.3f}**" if np.isclose(r.macro_f1, best["macro_f1"]) else f"{r.macro_f1:.3f}"
        lines.append(f"| {r.model} | {cell('accuracy')} | {cell('detection_rate')} | {pct(r.false_alarm_rate)} | {f1} | "
                     f"{pct(r.recall_DoS)} | {pct(r.recall_Probe)} | {pct(r.recall_R2L)} | {pct(r.recall_U2R)} | {r.train_seconds:.0f} |")
    return "\n".join(lines)


def bar_chart(df, path):
    settings = df["setting"].unique()
    fig, axes = plt.subplots(1, len(settings), figsize=(13, 5.2), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), settings):
        d = df[df.setting == s].iloc[::-1]
        y = np.arange(len(d))
        ax.barh(y + 0.2, d.accuracy * 100, height=0.4, label="Accuracy", color="#2b6cb0")
        ax.barh(y - 0.2, d.macro_f1 * 100, height=0.4, label="Macro-F1 (rare attacks count)", color="#dd8452")
        ax.set_yticks(y, [m.replace(" (modern baseline)", "*").replace(" (classic baseline)", "") for m in d.model])
        for lbl in ax.get_yticklabels():
            if lbl.get_text().startswith("Thesis"):
                lbl.set_fontweight("bold")
        ax.set_xlim(0, 100); ax.set_title(s, fontsize=11); ax.grid(axis="x", alpha=0.3)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle("NSL-KDD: model comparison (* = modern baseline)", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(path, dpi=150); plt.close(fig)


def confusion_chart(cm, classes, title, path):
    cm = np.array(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(classes)), classes); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    df = pd.read_csv(R / "metrics.csv")
    det = json.load(open(R / "details.json"))
    bar_chart(df, R / "model_comparison.png")
    classes = det["classes"]
    for key, label in [("A", "known attacks"), ("B", "new attacks")]:
        for name, fname in [(THESIS, f"confusion_thesis_{key}.png"), ("MultiBoosting (C4.5)", f"confusion_multiboost_{key}.png")]:
            confusion_chart(det["confusion"][key][name], classes, f"{name.split(':')[0]} - {label}", R / fname)

    md = ["# Results", "",
          f"NSL-KDD, {det['estimators']} boosting/bagging rounds, seed {det['seed']}"
          + (" (QUICK run on 20% of the data)" if det["quick"] else "") + ". Best value per column in **bold**.", ""]
    for s in df.setting.unique():
        md += [f"## {s}", "", table(df[df.setting == s]), ""]
    md += ["![Model comparison](model_comparison.png)", "",
           "| Thesis model, known attacks | Thesis model, new attacks |", "|---|---|",
           "| ![](confusion_thesis_A.png) | ![](confusion_thesis_B.png) |", "",
           "## Features chosen by information gain (thesis step 1)", "",
           "Top 8 of the 15 features kept for each one-vs-rest classifier:", "",
           "| Class | Top features (information gain) |", "|---|---|"]
    for c, feats in det["top_features"].items():
        md.append(f"| {c} | " + ", ".join(f"`{n}` ({g:.3f})" for n, g in feats[:8]) + " |")
    (R / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("Wrote", R / "RESULTS.md")


if __name__ == "__main__":
    main()
