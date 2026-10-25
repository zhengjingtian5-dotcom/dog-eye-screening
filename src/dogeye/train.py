"""Train and evaluate the baseline: logistic regression on frozen DINOv2 features.

Model selection and the decision threshold use the validation split only.
The threshold is the highest one that still catches at least --target-recall
of see_vet eyes on validation; the test split is touched once, at the end.

Usage:
    python -m dogeye.train --manifest data/manifest_split.csv --features data/features.npz \
        --out runs/baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dogeye.metrics import binary_report, per_condition_rates, save_plots, threshold_for_recall

C_GRID = [0.001, 0.01, 0.1, 1.0]


def fit_baseline(X_train, y_train, X_val, y_val, c_grid=C_GRID):
    """Fit one model per C and keep the one with the best validation AUPRC."""
    best = None
    for c in c_grid:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))
        model.fit(X_train, y_train)
        score = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
        print(f"C={c:<6} val AUPRC={score:.3f}")
        if best is None or score > best[0]:
            best = (score, c, model)
    return best[2], best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="data/manifest_split.csv")
    parser.add_argument("--features", default="data/features.npz")
    parser.add_argument("--out", default="runs/baseline")
    parser.add_argument("--target-recall", type=float, default=0.95)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    features = np.load(args.features, allow_pickle=True)["features"]
    X = features[manifest["feature_row"].to_numpy()]
    y = (manifest["label"] == "see_vet").to_numpy().astype(int)
    parts = {name: (manifest["split"] == name).to_numpy() for name in ["train", "val", "test"]}

    model, c = fit_baseline(X[parts["train"]], y[parts["train"]], X[parts["val"]], y[parts["val"]])
    val_probs = model.predict_proba(X[parts["val"]])[:, 1]
    threshold = threshold_for_recall(y[parts["val"]], val_probs, args.target_recall)

    test_probs = model.predict_proba(X[parts["test"]])[:, 1]
    report = {
        "model": "StandardScaler + LogisticRegression on DINOv2 features",
        "C": c,
        "target_recall": args.target_recall,
        "val": binary_report(y[parts["val"]], val_probs, threshold),
        "test": binary_report(y[parts["test"]], test_probs, threshold),
    }

    test_rows = manifest[parts["test"]].copy()
    test_rows["prob_see_vet"] = test_probs
    test_rows["flagged"] = (test_probs >= threshold).astype(int)
    by_condition = per_condition_rates(test_rows["condition"], test_rows["label"], test_rows["flagged"].to_numpy())

    joblib.dump({"model": model, "threshold": threshold, "C": c}, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    by_condition.to_csv(out / "per_condition.csv", index=False)
    test_rows.to_csv(out / "test_predictions.csv", index=False)
    save_plots(y[parts["test"]], test_probs, threshold, out)

    t = report["test"]
    print(f"\nthreshold {threshold:.3f} (set on val for recall >= {args.target_recall})")
    print(f"test recall {t['recall']:.3f}  specificity {t['specificity']:.3f}  "
          f"AUROC {t['auroc']:.3f}  AUPRC {t['auprc']:.3f}  ECE {t['ece']:.3f}")
    print(by_condition.to_string(index=False))
    print(f"saved model, metrics and plots to {out}")


if __name__ == "__main__":
    main()
