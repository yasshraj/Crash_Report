# Model 3: CatBoost for predicting SEVERE_INJURY (needs: pip install catboost).
# CatBoost handles text categories and blank numbers by itself, so there's no
# one-hot encoding or median filling here. Takes a few minutes to run.

import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    sys.exit("CatBoost is not installed, run: pip install catboost")

from data_prep import COLORS, NUMERIC_COLS, OUTPUT_DIR, RANDOM_STATE
from data_prep import get_categorical_cols, get_cv, load_clean_data, set_plot_style, split_data
from evaluation import evaluate_model

MODEL_KEY = "catboost"
NAME = "CatBoost"

MAX_TREES = 800  # CV picks the best number of trees below this

# every combination of these settings gets tried
PARAM_GRID = []
for d in [4, 6]:
    for lr in [0.03, 0.08]:
        for l2 in [3, 10]:
            PARAM_GRID.append({"depth": d, "learning_rate": lr, "l2_leaf_reg": l2})


def make_model(iterations, depth, learning_rate, l2_leaf_reg):
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,  # stops it making a catboost_info folder
    )
    return model


def validation_auc_curve(model):
    # AUC on the validation fold after each tree
    results = model.get_evals_result()
    for key in results:
        if key != "learn":
            val_key = key
            break
    for key in results[val_key]:
        if key.startswith("AUC"):
            auc_key = key
            break
    return np.array(results[val_key][auc_key])


def prepare_for_catboost(X, cat_cols):
    # catboost wants the category columns as plain text
    X = X.copy()
    for col in cat_cols:
        X[col] = X[col].astype(str).astype(object)
    return X


def main():
    df = load_clean_data()
    cat_cols = get_categorical_cols(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)} crashes | Test: {len(X_test)} crashes ({int(y_test.sum())} severe)")

    X_train = prepare_for_catboost(X_train, cat_cols)
    X_test = prepare_for_catboost(X_test, cat_cols)

    # tune with 5-fold CV. each fold grows MAX_TREES trees and records the AUC
    # after every tree, so this also finds the best number of trees
    print(f"\nTuning CatBoost ({len(PARAM_GRID)} settings x 5 folds)...")
    cv = get_cv()
    folds = list(cv.split(X_train, y_train))
    results = []
    all_mean_curves = []
    all_std_curves = []
    start = time.time()

    for i in range(len(PARAM_GRID)):
        params = PARAM_GRID[i]
        curves = []
        for train_idx, val_idx in folds:
            model = make_model(MAX_TREES, params["depth"], params["learning_rate"], params["l2_leaf_reg"])
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
                      cat_features=cat_cols,
                      eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
                      use_best_model=False)
            curves.append(validation_auc_curve(model))

        # stack the 5 fold curves (cut to the same length first)
        shortest = min(len(c) for c in curves)
        trimmed = []
        for c in curves:
            trimmed.append(c[:shortest])
        trimmed = np.vstack(trimmed)
        mean_curve = trimmed.mean(axis=0)
        std_curve = trimmed.std(axis=0)

        # smooth the curve before taking the peak so one lucky spike doesn't win
        smooth = pd.Series(mean_curve).rolling(25, center=True, min_periods=1).mean()
        best_it = int(np.argmax(smooth.values)) + 1

        row = params.copy()
        row["best_trees"] = best_it
        row["cv_auc"] = mean_curve[best_it - 1]
        row["cv_std"] = std_curve[best_it - 1]
        results.append(row)
        all_mean_curves.append(mean_curve)
        all_std_curves.append(std_curve)

        seconds = time.time() - start
        print(f"  [{i + 1}/{len(PARAM_GRID)}] depth={params['depth']} lr={params['learning_rate']} "
              f"l2={params['l2_leaf_reg']} -> {best_it} trees, "
              f"CV ROC-AUC {row['cv_auc']:.3f} +/- {row['cv_std']:.3f}  ({seconds:.0f}s)")

    res = pd.DataFrame(results)
    res.to_csv(OUTPUT_DIR / f"{MODEL_KEY}_cv_results.csv", index=False)
    best_i = int(res["cv_auc"].idxmax())
    best = res.loc[best_i]
    best_params = {"depth": int(best["depth"]), "learning_rate": float(best["learning_rate"]),
                   "l2_leaf_reg": float(best["l2_leaf_reg"])}
    best_trees = int(best["best_trees"])
    print(f"Best: {best_params}, {best_trees} trees -> CV ROC-AUC {best['cv_auc']:.3f} +/- {best['cv_std']:.3f}")

    # learning curve for the best setting
    mean_curve = all_mean_curves[best_i]
    std_curve = all_std_curves[best_i]
    print("\nLearning curve (best setting):")
    for k in [10, 50, 100, 200, 400, MAX_TREES]:
        if k <= len(mean_curve):
            print(f"  {k} trees: CV ROC-AUC {mean_curve[k - 1]:.3f}")

    plt = set_plot_style()
    xs = np.arange(1, len(mean_curve) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve,
                    color=COLORS[MODEL_KEY], alpha=0.15, lw=0)
    ax.plot(xs, mean_curve, color=COLORS[MODEL_KEY], lw=2)
    ax.axvline(best_trees, color=COLORS["ink_2"], ls="--", lw=1)
    ax.text(best_trees, ax.get_ylim()[0], f"  best: {best_trees} trees",
            color=COLORS["ink_2"], va="bottom", fontsize=9)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Cross-validated ROC-AUC")
    ax.set_title("CatBoost learning curve (band = +/- 1 std across folds)")
    fig.savefig(OUTPUT_DIR / f"{MODEL_KEY}_learning_curve.png")
    plt.close(fig)

    # train the final model on the whole training set
    final = make_model(best_trees, best_params["depth"], best_params["learning_rate"],
                       best_params["l2_leaf_reg"])
    final.fit(X_train, y_train, cat_features=cat_cols)

    # final check on the test set
    test_prob = final.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train, final.predict_proba(X_train)[:, 1])
    print(f"\nTrain ROC-AUC: {train_auc:.3f}")
    params_to_save = best_params.copy()
    params_to_save["iterations"] = best_trees
    evaluate_model(MODEL_KEY, NAME, y_test, test_prob,
                   cv_auc_mean=float(best["cv_auc"]), cv_auc_std=float(best["cv_std"]),
                   extra={"best_params": params_to_save, "train_roc_auc": train_auc})

    # feature importance (one score per original column, adds up to 100)
    imp = pd.Series(final.get_feature_importance(), index=final.feature_names_)
    imp = imp.sort_values(ascending=False)
    imp.to_csv(OUTPUT_DIR / f"{MODEL_KEY}_feature_importance.csv", header=["importance"])
    print("\nTop 15 features (full list in the csv):")
    print(imp.head(15).round(2).to_string())

    top = imp.head(15).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(top.index, top.values, color=COLORS[MODEL_KEY], height=0.6)
    for i in range(len(top)):
        v = top.values[i]
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", color=COLORS["ink_2"], fontsize=9)
    ax.set_xlabel("Importance (sums to 100 over all columns)")
    ax.set_title("CatBoost: 15 most important columns")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, top.max() * 1.15)
    fig.savefig(OUTPUT_DIR / f"{MODEL_KEY}_feature_importance.png")
    plt.close(fig)

    # SHAP: how much an "Unknown" value pushes the prediction towards (+) or away from (-) severe
    shap = final.get_feature_importance(Pool(X_train, y_train, cat_features=cat_cols),
                                        type="ShapValues")
    shap = pd.DataFrame(shap[:, :-1], columns=final.feature_names_, index=X_train.index)

    rows = []
    for col in final.feature_names_:
        if col in NUMERIC_COLS:
            is_unknown = X_train[col].isna()
        else:
            is_unknown = X_train[col] == "Unknown"

        # only columns with at least 10 unknowns
        if is_unknown.sum() >= 10:
            rows.append({
                "column": col,
                "n_unknown": int(is_unknown.sum()),
                "avg_push_if_unknown": shap[is_unknown][col].mean(),
                "avg_push_if_known": shap[~is_unknown][col].mean(),
            })

    unk = pd.DataFrame(rows)
    unk = unk.sort_values("avg_push_if_unknown")
    unk.to_csv(OUTPUT_DIR / f"{MODEL_KEY}_unknown_shap.csv", index=False)
    print("\nAverage SHAP push of Unknown vs known values:")
    print(unk.to_string(index=False, float_format="%+.3f"))
    print("Saved all CatBoost charts and results to outputs/")


if __name__ == "__main__":
    main()
