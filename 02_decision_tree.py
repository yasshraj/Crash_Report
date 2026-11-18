# Model 2: decision tree for predicting SEVERE_INJURY.
# A tree with no limits memorises the training data, so max_depth and
# min_samples_leaf are tuned with cross-validation.

import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from data_prep import COLORS, NUMERIC_COLS, OUTPUT_DIR, RANDOM_STATE
from data_prep import get_categorical_cols, get_cv, load_clean_data, set_plot_style, split_data
from evaluation import evaluate_model

warnings.filterwarnings("ignore", category=UserWarning)
MODEL_KEY = "decision_tree"
NAME = "Decision tree"


def make_preprocessor(num_cols, cat_cols):
    # same as logistic regression but without scaling (trees don't need it)
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=10, sparse_output=False), cat_cols),
    ])
    return preprocessor


def make_pipeline(num_cols, cat_cols):
    model = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    pipe = Pipeline([
        ("prep", make_preprocessor(num_cols, cat_cols)),
        ("model", model),
    ])
    return pipe


def clean_names(names):
    # shorter names, e.g. "cat__WEATHER_Clear" -> "WEATHER_Clear"
    new_names = []
    for n in names:
        n = n.replace("num__", "").replace("cat__", "")
        n = n.replace("missingindicator_", "MISSING_")
        new_names.append(n)
    return new_names


def main():
    df = load_clean_data()
    cat_cols = get_categorical_cols(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)} crashes | Test: {len(X_test)} crashes ({int(y_test.sum())} severe)")

    prep = make_preprocessor(NUMERIC_COLS, cat_cols)
    prep.fit(X_train)
    print("Columns the tree sees:", len(prep.get_feature_names_out()))

    # a tree with no limits, to show how much it overfits
    free = make_pipeline(NUMERIC_COLS, cat_cols)
    cv_free = cross_validate(free, X_train, y_train, cv=get_cv(), scoring="roc_auc",
                             return_train_score=True)
    free.fit(X_train, y_train)
    tree = free.named_steps["model"]
    print(f"\nUnlimited tree: depth {tree.get_depth()}, {tree.get_n_leaves()} leaves")
    print(f"  train ROC-AUC: {cv_free['train_score'].mean():.3f}")
    print(f"  CV ROC-AUC:    {cv_free['test_score'].mean():.3f}")

    # tune max_depth and min_samples_leaf with 5-fold CV
    print("\nTuning the tree...")
    param_grid = {
        "model__max_depth": [2, 3, 4, 5, 6, 8, 10],
        "model__min_samples_leaf": [5, 10, 20, 30, 40, 60],
    }
    grid = GridSearchCV(make_pipeline(NUMERIC_COLS, cat_cols), param_grid,
                        scoring="roc_auc", cv=get_cv(), n_jobs=-1,
                        return_train_score=True)
    grid.fit(X_train, y_train)

    res = pd.DataFrame(grid.cv_results_)
    table = res.pivot_table(index="param_model__max_depth",
                            columns="param_model__min_samples_leaf",
                            values="mean_test_score")
    table.index.name = "max_depth"
    table.columns.name = "min_samples_leaf"
    print("CV ROC-AUC (rows = max_depth, columns = min_samples_leaf):")
    print(table.round(3).to_string())

    best_depth = grid.best_params_["model__max_depth"]
    best_leaf = grid.best_params_["model__min_samples_leaf"]
    cv_mean = res.loc[grid.best_index_, "mean_test_score"]
    cv_std = res.loc[grid.best_index_, "std_test_score"]
    cv_train = res.loc[grid.best_index_, "mean_train_score"]
    print(f"Best: max_depth={best_depth}, min_samples_leaf={best_leaf} "
          f"(train {cv_train:.3f}, CV {cv_mean:.3f})")

    # final check on the test set
    best_model = grid.best_estimator_
    test_prob = best_model.predict_proba(X_test)[:, 1]
    train_auc = grid.score(X_train, y_train)
    print(f"\nTrain ROC-AUC: {train_auc:.3f}")
    best_params = {"max_depth": best_depth, "min_samples_leaf": best_leaf}
    evaluate_model(MODEL_KEY, NAME, y_test, test_prob,
                   cv_auc_mean=cv_mean, cv_auc_std=cv_std,
                   extra={"best_params": best_params, "train_roc_auc": train_auc})
    # note: the tree only has a few leaves, so it only gives a few different
    # probabilities - that's why its ROC curve looks like straight steps
    tree = best_model.named_steps["model"]

    # rules the tree learned. on 0/1 columns "<= 0.5" means NO and "> 0.5" means YES
    names = clean_names(best_model.named_steps["prep"].get_feature_names_out())
    rules = export_text(tree, feature_names=names, show_weights=False, decimals=1)
    rules_file = OUTPUT_DIR / f"{MODEL_KEY}_rules.txt"
    rules_file.write_text(rules)
    print("\nTree rules (class 1 = severe):")
    print(rules)

    # picture of the first few levels of the tree
    plt = set_plot_style()
    show_depth = min(best_depth, 3)
    fig, ax = plt.subplots(figsize=(18, 7.5))
    plot_tree(tree, feature_names=names, class_names=["Not severe", "Severe"],
              filled=True, rounded=True, impurity=False, proportion=True,
              max_depth=show_depth, fontsize=8, ax=ax)
    ax.set_title(f"Decision tree (first {show_depth} levels). 'value' = share of "
                 "[not severe, severe] after balancing class weights")
    fig.savefig(OUTPUT_DIR / f"{MODEL_KEY}_tree.png")
    plt.close(fig)
    print("Saved tree chart and rules to outputs/")

    # feature importance = how much each column was used for the splits
    imp = pd.Series(tree.feature_importances_, index=names)
    imp = imp.sort_values(ascending=False)
    imp = imp[imp > 0]
    imp.to_csv(OUTPUT_DIR / f"{MODEL_KEY}_feature_importance.csv", header=["importance"])
    print("\nFeature importance:")
    print(imp.round(3).to_string())
    print(f"Only {len(imp)} of {len(names)} columns were used")

    top = imp.head(15).sort_values()
    height = max(3, 0.42 * len(top) + 1)
    fig, ax = plt.subplots(figsize=(8.5, height))
    ax.barh(top.index, top.values, color=COLORS[MODEL_KEY], height=0.6)
    for i in range(len(top)):
        v = top.values[i]
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", color=COLORS["ink_2"], fontsize=9)
    ax.set_xlabel("Importance (share of total)")
    ax.set_title("Decision tree: columns the tree relied on")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, top.max() * 1.15)
    fig.savefig(OUTPUT_DIR / f"{MODEL_KEY}_feature_importance.png")
    plt.close(fig)
    print("Saved feature importance chart to outputs/")


if __name__ == "__main__":
    main()
