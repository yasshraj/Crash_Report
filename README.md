# Predicting crash severity for rural cyclist crashes

Three models are compared on the same cleaned data and the same 80/20 train/test split:
logistic regression, a decision tree, and CatBoost.

## Why this project

Most research on cyclist crashes focuses on cities. When we looked through the existing
studies we found a big gap: very little work looks at cyclists in rural areas, and even
less tries to predict how severe their injuries will be. As far as we know, that makes
this project a fairly new contribution.

We didn't want to rely on a single model, so we tried three different ones. They go from
simple and easy to explain (logistic regression, decision tree) to more powerful but harder
to interpret (CatBoost). Training all three on exactly the same data shows whether the
more complex model is actually worth it, and `04_compare_models.py` checks whether the
differences between them are real or just down to chance.

## Setup

1. Put `Rural_Cyclist_Crashes_Final.csv` in this folder (or in a `data/` sub-folder).
   Any file name containing `Rural_Cyclist_Crashes` is found automatically.
2. Install the libraries:

   ```
   pip install -r requirements.txt
   ```

## Run order

```
python 00_explore_data.py         # understand the data and the Unknowns
python 01_logistic_regression.py  # model 1
python 02_decision_tree.py        # model 2
python 03_catboost_model.py       # model 3
python 04_compare_models.py       # side-by-side comparison + significance check
```

Every chart and result file goes into `outputs/`.

## Files

| File | What it does |
|---|---|
| `data_prep.py` | Shared loading, cleaning and the train/test split (used by every model) |
| `evaluation.py` | Shared metrics, confusion matrix and ROC chart (used by every model) |
| `00_explore_data.py` | Target balance, % Unknown per column, whether Unknown is informative |
| `01_logistic_regression.py` | Baseline model, tuning of C, coefficients, Unknown-handling experiment |
| `02_decision_tree.py` | Overfitting demo, tuning of depth/leaf size, tree rules, importance |
| `03_catboost_model.py` | Native categorical/missing handling, tuning, learning curve, SHAP on Unknowns |
| `04_compare_models.py` | Comparison table, bootstrap confidence intervals, pairwise tests |

## How Unknowns are handled (the recipe)

| Situation | What we do | Why |
|---|---|---|
| Text "Unknown" in a category | Keep it as its own category | It's informative, and deleting rows leaves only 45 of 979 crashes |
| Blank in a number column | Median (from the training set) + a 0/1 "was missing" column; CatBoost handles blanks itself | The model gets a usable number and still knows it was missing |
| Partial unknowns ("Dark Unknown Lighting", "Grade Unknown Slope") | Keep as they are | They still carry information |
| "Non-trafficway/Driveway" | Not an unknown: a real location | Don't treat it as missing |
| Categories with fewer than 10 rows | Grouped as "infrequent" | Too rare to learn from |
| Impossible values (cyclist age 0) | Set to blank | Data-entry error |
| Columns that are mostly Unknown | Keep them, but test dropping them (step 8 of the logistic regression script) | Dropping made no measurable difference here |

All filling-in is learned from the training data only (inside a scikit-learn `Pipeline`),
so no information leaks from the test set.

## Why not accuracy?

Only about 21% of crashes are severe, so always guessing "not severe" already gets about 79% accuracy.
Use recall, precision and F1 for the severe class, plus ROC-AUC and PR-AUC.
