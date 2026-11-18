# Compares the models using the results they saved in outputs/.
# Run this after 01, 02 and 03. Any model that hasn't been run is left out.

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from data_prep import COLORS, OUTPUT_DIR, RANDOM_STATE, set_plot_style

MODELS = ["logistic_regression", "decision_tree", "catboost"]
N_BOOT = 2000  # number of bootstrap samples


def main():
    # load the saved metrics and test predictions for each model
    metrics = {}
    preds = {}
    for key in MODELS:
        mfile = OUTPUT_DIR / f"{key}_metrics.json"
        pfile = OUTPUT_DIR / f"{key}_test_predictions.csv"
        if mfile.exists() and pfile.exists():
            metrics[key] = json.loads(mfile.read_text())
            preds[key] = pd.read_csv(pfile)
        else:
            print("No results for", key, "yet, skipping it")
    if len(metrics) < 2:
        raise SystemExit("Need results from at least two models to compare.")

    # every model should have been tested on the same crashes
    first_key = list(preds.keys())[0]
    y_true = preds[first_key]["y_true"].to_numpy()
    for key in preds:
        if not np.array_equal(preds[key]["y_true"].to_numpy(), y_true):
            raise SystemExit(key + " was tested on different crashes - rerun all the scripts.")

    # comparison table
    rows = {
        "Recall (severe caught)": "recall_severe",
        "Precision (severe)": "precision_severe",
        "F1 (severe)": "f1_severe",
        "Balanced accuracy": "balanced_accuracy",
        "Accuracy": "accuracy",
        "ROC-AUC (test)": "roc_auc",
        "PR-AUC (test)": "pr_auc",
        "ROC-AUC (5-fold CV, train)": "cv_roc_auc_mean",
        "ROC-AUC (training data)": "train_roc_auc",
    }
    columns = {}
    for key in metrics:
        m = metrics[key]
        col = {}
        for label in rows:
            value = m.get(rows[label])
            if value is None:
                col[label] = "-"
            else:
                col[label] = f"{value:.3f}"
        col["Severe caught"] = f"{m['tp']} of {m['tp'] + m['fn']}"
        col["False alarms"] = str(m["fp"])
        columns[m["model"]] = col

    table = pd.DataFrame(columns)
    print("Test set comparison:")
    print(table.to_string())
    table.to_csv(OUTPUT_DIR / "model_comparison.csv")

    # bootstrap: resample the test set N_BOOT times and recompute the AUC each time
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    boot = {}
    for key in preds:
        boot[key] = []

    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        # skip samples that only have one class (AUC can't be calculated)
        if y_true[idx].min() == y_true[idx].max():
            continue
        for key in preds:
            probs = preds[key]["y_prob"].to_numpy()
            boot[key].append(roc_auc_score(y_true[idx], probs[idx]))

    for key in boot:
        boot[key] = np.array(boot[key])

    print("\nTest ROC-AUC with 95% bootstrap CI:")
    for key in preds:
        lo, hi = np.percentile(boot[key], [2.5, 97.5])
        print(f"  {metrics[key]['model']:<20} {metrics[key]['roc_auc']:.3f}  [{lo:.3f}, {hi:.3f}]")

    # compare each pair of models. if the CI of the difference includes 0
    # we can't say one is really better than the other
    print("\nDifference in ROC-AUC between models:")
    keys = list(preds)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            name_a = metrics[a]["model"]
            name_b = metrics[b]["model"]
            diff = boot[a] - boot[b]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            if lo <= 0 <= hi:
                verdict = "no clear winner (CI includes 0)"
            elif diff.mean() > 0:
                verdict = name_a + " is better"
            else:
                verdict = name_b + " is better"
            test_diff = metrics[a]["roc_auc"] - metrics[b]["roc_auc"]
            print(f"  {name_a} minus {name_b}:{test_diff:+.3f}  [{lo:+.3f}, {hi:+.3f}] -> {verdict}")

    # chart: ROC curves on the left, bar chart of metrics on the right
    plt = set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    ax1.plot([0, 1], [0, 1], ls="--", lw=1.2, color=COLORS["muted"])
    for key in preds:
        fpr, tpr, _ = roc_curve(y_true, preds[key]["y_prob"])
        label = f"{metrics[key]['model']} (AUC {metrics[key]['roc_auc']:.3f})"
        ax1.plot(fpr, tpr, lw=2, color=COLORS[key], label=label)
    ax1.set_xlabel("False alarm rate")
    ax1.set_ylabel("Catch rate (severe caught)")
    ax1.set_title("ROC curves on the test set")
    ax1.legend(loc="lower right", frameon=False)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.01)

    bar_labels = ["Recall\n(severe)", "Precision\n(severe)", "F1\n(severe)",
                  "Balanced\naccuracy", "ROC-AUC", "PR-AUC"]
    bar_fields = ["recall_severe", "precision_severe", "f1_severe",
                  "balanced_accuracy", "roc_auc", "pr_auc"]
    x = np.arange(len(bar_fields))
    width = 0.8 / len(metrics)
    if len(metrics) < 3:
        font = 7.5
    else:
        font = 6.5

    model_keys = list(metrics)
    for i in range(len(model_keys)):
        key = model_keys[i]
        vals = []
        for f in bar_fields:
            vals.append(metrics[key][f])
        pos = x - 0.4 + width * (i + 0.5)
        ax2.bar(pos, vals, width * 0.92, color=COLORS[key], label=metrics[key]["model"])
        for j in range(len(vals)):
            ax2.text(pos[j], vals[j] + 0.01, f"{vals[j]:.2f}", ha="center",
                     fontsize=font, color=COLORS["ink_2"])
    ax2.set_xticks(x, bar_labels)
    ax2.set_ylim(0, 1)
    ax2.set_title("Key test-set metrics (higher is better)")
    ax2.legend(frameon=False, loc="upper left", ncol=len(metrics))
    ax2.grid(axis="x", visible=False)
    fig.savefig(OUTPUT_DIR / "model_comparison.png")
    plt.close(fig)
    print("\nSaved comparison chart and table to outputs/")


if __name__ == "__main__":
    main()
