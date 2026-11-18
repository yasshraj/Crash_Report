# Evaluation used by all three models so they're scored the same way.
# Accuracy is misleading here (79% of crashes are not severe), so the main ones
# to look at are recall/precision/F1 for the severe class, ROC-AUC and PR-AUC.

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve

from data_prep import COLORS, OUTPUT_DIR, set_plot_style


def evaluate_model(model_key, display_name, y_true, y_prob, threshold=0.5,
                   cv_auc_mean=None, cv_auc_std=None, extra=None):
    # scores the model on the test set, saves a chart and the results
    # (04_compare_models.py reads the saved files later)
    plt = set_plot_style()
    OUTPUT_DIR.mkdir(exist_ok=True)

    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    results = {
        "model": display_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_severe": precision_score(y_true, y_pred, zero_division=0),
        "recall_severe": recall_score(y_true, y_pred, zero_division=0),
        "f1_severe": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "cv_roc_auc_mean": cv_auc_mean,
        "cv_roc_auc_std": cv_auc_std,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if extra is not None:
        results.update(extra)

    n_severe = int(y_true.sum())
    print("\nTest set:", len(y_true), "crashes,", n_severe, "severe")
    print(f"  not severe -> correct: {tn}, false alarm: {fp}")
    print(f"  severe     -> missed: {fn}, caught: {tp}")
    print(f"Accuracy:          {results['accuracy']:.3f}")
    print(f"Balanced accuracy: {results['balanced_accuracy']:.3f}")
    print(f"Recall:            {results['recall_severe']:.3f}")
    print(f"Precision:         {results['precision_severe']:.3f}")
    print(f"F1:                {results['f1_severe']:.3f}")
    print(f"ROC-AUC:           {results['roc_auc']:.3f}")
    print(f"PR-AUC:            {results['pr_auc']:.3f}")
    if cv_auc_mean is not None:
        print(f"CV ROC-AUC:        {cv_auc_mean:.3f} +/- {cv_auc_std:.3f}")

    # left: confusion matrix, right: ROC curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    cm = np.array([[tn, fp], [fn, tp]])
    ax1.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    names = [["correct", "false alarm"], ["missed", "caught"]]
    for i in range(2):
        for j in range(2):
            # white text on the darker squares
            if cm[i, j] > cm.max() * 0.55:
                text_color = "#ffffff"
            else:
                text_color = COLORS["ink"]
            ax1.text(j, i, str(cm[i, j]) + "\n" + names[i][j], ha="center", va="center",
                     fontsize=12, color=text_color)
    ax1.set_xticks([0, 1], ["Not severe", "Severe"])
    ax1.set_yticks([0, 1], ["Not severe", "Severe"])
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.grid(False)
    ax1.set_title("Confusion matrix (test set)")

    if model_key in COLORS:
        line_color = COLORS[model_key]
    else:
        line_color = COLORS["ink"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax2.plot([0, 1], [0, 1], ls="--", lw=1.2, color=COLORS["muted"])
    ax2.text(0.62, 0.55, "coin flip (0.50)", color=COLORS["muted"], fontsize=9, rotation=33)
    ax2.plot(fpr, tpr, lw=2, color=line_color)
    ax2.set_xlabel("False alarm rate (non-severe flagged as severe)")
    ax2.set_ylabel("Catch rate (severe caught)")
    ax2.set_title(f"ROC curve, AUC = {results['roc_auc']:.3f}")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.01)
    fig.suptitle(display_name, fontweight="bold", color=COLORS["ink"])
    fig.savefig(OUTPUT_DIR / f"{model_key}_evaluation.png")
    plt.close(fig)

    # save the numbers and predictions for the comparison script
    with open(OUTPUT_DIR / f"{model_key}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    preds = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    preds.to_csv(OUTPUT_DIR / f"{model_key}_test_predictions.csv", index=False)
    print("Saved chart, metrics and predictions to outputs/")
    return results
