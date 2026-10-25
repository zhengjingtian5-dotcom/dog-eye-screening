"""Screening metrics. The positive class is "see_vet": a miss is the costly error."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def threshold_for_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """Highest threshold whose recall on (y_true, scores) is at least target_recall."""
    positives = np.sort(scores[y_true == 1])[::-1]
    if len(positives) == 0:
        raise ValueError("no positive examples to set a threshold on")
    needed = int(np.ceil(target_recall * len(positives)))
    return float(positives[max(needed, 1) - 1])


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            error += in_bin.mean() * abs(probs[in_bin].mean() - y_true[in_bin].mean())
    return float(error)


def binary_report(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "recall": float(recall),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "precision": float(precision),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "auroc": float(roc_auc_score(y_true, probs)),
        "auprc": float(average_precision_score(y_true, probs)),
        "ece": expected_calibration_error(y_true, probs),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def per_condition_rates(conditions: pd.Series, labels: pd.Series, pred: np.ndarray) -> pd.DataFrame:
    """Share of each condition flagged see_vet: recall for diseases, false-alarm rate for healthy."""
    frame = pd.DataFrame({"condition": conditions.to_numpy(), "label": labels.to_numpy(), "flagged": pred})
    return (
        frame.groupby(["condition", "label"])["flagged"]
        .agg(n="size", flagged_rate="mean")
        .reset_index()
        .sort_values("flagged_rate")
    )


def save_plots(y_true: np.ndarray, probs: np.ndarray, threshold: float, out_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["normal", "see_vet"])
    ax.set_yticks([0, 1], ["normal", "see_vet"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Test confusion matrix (t={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.close(fig)

    edges = np.linspace(0, 1, 11)
    bins = np.minimum((probs * 10).astype(int), 9)
    centers, observed = [], []
    for b in range(10):
        in_bin = bins == b
        if in_bin.any():
            centers.append(probs[in_bin].mean())
            observed.append(y_true[in_bin].mean())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    ax.plot(centers, observed, marker="o", label="model")
    ax.set_xticks(edges[::2])
    ax.set_xlabel("predicted probability of see_vet")
    ax.set_ylabel("observed share see_vet")
    ax.set_title("Reliability (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/reliability.png", dpi=150)
    plt.close(fig)
