# Model 1: logistic regression for predicting SEVERE_INJURY.
# It only works with numbers, so blanks are filled in, categories are one-hot
# encoded and the number columns are scaled before fitting.

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import COLORS, NUMERIC_COLS, OUTPUT_DIR, RANDOM_STATE
from data_prep import get_categorical_cols, get_cv, load_clean_data, set_plot_style, split_data
from evaluation import evaluate_model

warnings.filterwarnings("ignore", category=UserWarning)  # hides the rare category warnings
MODEL_KEY = "logistic_regression"
NAME = "Logistic regression"


def make_preprocessor(num_cols, cat_cols, unknown_strategy="keep"):
    # "keep"  -> "Unknown" stays as its own category, blanks get the median + a missing flag
    # "guess" -> Unknowns/blanks get replaced with the most common value / median
    #            (only used in the experiment at the end)
    if unknown_strategy == "keep":
        add_flag = True
    else:
        add_flag = False

    numeric = Pipeline([
        ("fill_blanks", SimpleImputer(strategy="median", add_indicator=add_flag)),
        ("scale", StandardScaler()),
    ])

    cat_steps = []
    if unknown_strategy != "keep":
        cat_steps.append(("guess_unknowns", SimpleImputer(missing_values="Unknown", strategy="most_frequent")))
    # categories with fewer than 10 rows get grouped together as "infrequent"
    cat_steps.append(("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                               min_frequency=10, sparse_output=False)))
    categorical = Pipeline(cat_steps)

    preprocessor = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])
    return preprocessor


def make_pipeline(num_cols, cat_cols, C=1.0, unknown_strategy="keep"):
    # class_weight="balanced" because severe crashes are about 4x rarer
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    pipe = Pipeline([
        ("prep", make_preprocessor(num_cols, cat_cols, unknown_strategy)),
        ("model", model),
    ])
    return pipe


def main():
    df = load_clean_data()
    cat_cols = get_categorical_cols(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)} crashes ({y_train.mean():.1%} severe)")
    print(f"Test:  {len(X_test)} crashes ({y_test.mean():.1%} severe)")

    # baseline that always guesses "not severe"
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    print(f"Baseline accuracy (always 'not severe'): {np.mean(dummy_pred == y_test):.3f}")

    # check what the preprocessing does to the training data
    prep = make_preprocessor(NUMERIC_COLS, cat_cols)
    prep.fit(X_train)
    print("Columns after preprocessing:", len(prep.get_feature_names_out()))
    medians = prep.named_transformers_["num"].named_steps["fill_blanks"].statistics_
    median_values = {}
    for col, value in zip(NUMERIC_COLS, medians):
        median_values[col] = round(float(value), 1)
    print("Medians used to fill blanks:", median_values)

    # tune C with 5-fold CV on the training set (smaller C = simpler model)
    print("\nTuning C...")
    C_values = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
    grid = GridSearchCV(make_pipeline(NUMERIC_COLS, cat_cols),
                        param_grid={"model__C": C_values},
                        scoring="roc_auc", cv=get_cv(), n_jobs=-1)
    grid.fit(X_train, y_train)

    res = pd.DataFrame(grid.cv_results_)
    for i in range(len(res)):
        row = res.iloc[i]
        line = f"  C = {row['param_model__C']:<6}  CV ROC-AUC = {row['mean_test_score']:.3f} +/- {row['std_test_score']:.3f}"
        if row["rank_test_score"] == 1:
            line = line + "  <- best"
        print(line)

    best_C = grid.best_params_["model__C"]
    cv_mean = res.loc[grid.best_index_, "mean_test_score"]
    cv_std = res.loc[grid.best_index_, "std_test_score"]
    print("Best C:", best_C)

    # final check on the test set
    best_model = grid.best_estimator_
    test_prob = best_model.predict_proba(X_test)[:, 1]
    train_auc = grid.score(X_train, y_train)
    print(f"\nTrain ROC-AUC: {train_auc:.3f}")
    evaluate_model(MODEL_KEY, NAME, y_test, test_prob,
                   cv_auc_mean=cv_mean, cv_auc_std=cv_std,
                   extra={"best_params": {"C": best_C}, "train_roc_auc": train_auc})

    # coefficients: positive = raises the risk, negative = lowers it
    feature_names = best_model.named_steps["prep"].get_feature_names_out()
    names = []
    for n in feature_names:
        n = n.replace("num__", "").replace("cat__", "")
        n = n.replace("missingindicator_", "MISSING_")
        names.append(n)

    coefs = best_model.named_steps["model"].coef_[0]
    # the weights are tiny when C is small, so show each one as a % of the biggest one
    strength = 100 * coefs / np.abs(coefs).max()

    coef_df = pd.DataFrame({"feature": names, "coef": coefs,
                            "strength_pct": strength, "odds_ratio": np.exp(coefs)})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    coef_df = coef_df.drop(columns=["abs_coef"])
    coef_df.to_csv(OUTPUT_DIR / f"{MODEL_KEY}_coefficients.csv", index=False)

    raises = coef_df.sort_values("coef", ascending=False).head(10)
    lowers = coef_df.sort_values("coef").head(10)
    print("\nTop 10 factors that raise the risk:")
    print(raises[["feature", "strength_pct"]].to_string(index=False, float_format="%+.0f%%"))
    print("\nTop 10 factors that lower the risk:")
    print(lowers[["feature", "strength_pct"]].to_string(index=False, float_format="%+.0f%%"))

    # bar chart of the 16 strongest factors
    plt = set_plot_style()
    top = coef_df.head(16).sort_values("coef")
    colors = []
    for c in top["coef"]:
        if c > 0:
            colors.append(COLORS["increases_risk"])
        else:
            colors.append(COLORS["decreases_risk"])

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.barh(top["feature"], top["strength_pct"], color=colors, height=0.6)
    ax.axvline(0, color=COLORS["ink_2"], lw=1)
    ax.set_xlabel("Strength, % of strongest factor  (red = raises risk, blue = lowers risk)")
    ax.set_title("Logistic regression: 16 strongest factors")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUTPUT_DIR / f"{MODEL_KEY}_coefficients.png")
    plt.close(fig)
    print("Saved coefficient chart and csv to outputs/")

    # experiment: compare ways of handling Unknowns (5-fold CV repeated 5 times)
    print("\nComparing ways to handle Unknowns (CV ROC-AUC):")
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

    # columns that are more than 40% Unknown
    mostly_unknown = []
    for c in cat_cols:
        if (X_train[c] == "Unknown").mean() > 0.40:
            mostly_unknown.append(c)
    fewer_cats = []
    for c in cat_cols:
        if c not in mostly_unknown:
            fewer_cats.append(c)

    pipe_a = make_pipeline(NUMERIC_COLS, cat_cols, best_C, "keep")
    pipe_b = make_pipeline(NUMERIC_COLS, cat_cols, best_C, "guess")
    pipe_c = make_pipeline(NUMERIC_COLS, fewer_cats, best_C, "keep")
    X_train_c = X_train.drop(columns=mostly_unknown)

    tests = [
        ["A. Keep 'Unknown' as a category + missing flags", pipe_a, X_train],
        ["B. Replace Unknown/blank with most common value", pipe_b, X_train],
        ["C. Drop columns >40% Unknown " + str(mostly_unknown), pipe_c, X_train_c],
    ]
    for label, pipe, X in tests:
        scores = cross_val_score(pipe, X, y_train, cv=rcv, scoring="roc_auc", n_jobs=-1)
        print(f"  {label}: {scores.mean():.3f} +/- {scores.std():.3f}")
    # D. deleting every row with an Unknown isn't an option, only 45 rows (8 severe) are left


if __name__ == "__main__":
    main()
