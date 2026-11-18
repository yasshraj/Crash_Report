# Shared code for loading, cleaning and splitting the data.
# All the model scripts use this so they get exactly the same data and split.
#
# Unknowns: "Unknown" text is kept as its own category, blank numbers are left
# as NaN (each model fills them in using the training data only), and a cyclist
# age of 0 is treated as blank.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

TARGET = "SEVERE_INJURY"  # 1 = severe, 0 = not severe
RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_COLS = ["HOUR", "CYCLIST_AGE", "DRIVER_AGE", "VEHICLE_AGE", "VSPD_LIM"]

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def find_csv():
    # look for the csv in this folder or in a data/ folder
    here = Path(__file__).resolve().parent
    folders = [here, here / "data"]
    for folder in folders:
        matches = sorted(folder.glob("*Rural_Cyclist_Crashes*.csv"))
        if len(matches) > 0:
            return matches[0]
    raise FileNotFoundError("Could not find the crash CSV. Put it in this folder or in a 'data' folder.")


def load_clean_data(verbose=True):
    path = find_csv()
    df = pd.read_csv(path)
    if verbose:
        print("Loaded", path.name, "-", df.shape[0], "rows,", df.shape[1], "columns")

    # remove duplicate rows
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    if verbose and n_removed > 0:
        print("Removed", n_removed, "duplicate row(s)")

    # tidy up text columns, empty cells become "Unknown"
    cat_cols = get_categorical_cols(df)
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str)
        df[col] = df[col].str.strip()
        df[col] = df[col].replace("", "Unknown")

    # a cyclist age of 0 is a typing mistake so make it blank
    bad_age = df["CYCLIST_AGE"] <= 0
    if verbose and bad_age.sum() > 0:
        print("Set", bad_age.sum(), "cyclist age(s) of 0 to blank")
    df.loc[bad_age, "CYCLIST_AGE"] = np.nan

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_categorical_cols(df):
    # every column that isn't numeric or the target
    cat_cols = []
    for col in df.columns:
        if col not in NUMERIC_COLS and col != TARGET:
            cat_cols.append(col)
    return cat_cols


def split_data(df):
    # 80/20 split, stratify keeps the same % of severe crashes in both parts
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def get_cv():
    # 5-fold cross validation, used when tuning every model
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# colours used in all the charts
COLORS = {
    "logistic_regression": "#2a78d6",
    "decision_tree": "#eb6834",
    "catboost": "#1baf7a",
    "increases_risk": "#e34948",
    "decreases_risk": "#2a78d6",
    "ink": "#0b0b0b",
    "ink_2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "surface": "#fcfcfb",
}


def set_plot_style():
    import matplotlib
    matplotlib.use("Agg")  # save charts to files without opening a window
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": COLORS["surface"],
        "axes.facecolor": COLORS["surface"],
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": COLORS["ink_2"],
        "axes.titlecolor": COLORS["ink"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "font.size": 10,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
    return plt
