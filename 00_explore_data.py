# Look at the data before building any models.
# Checks how balanced the target is and how many "Unknown" values each column has.

import pandas as pd

from data_prep import NUMERIC_COLS, OUTPUT_DIR, TARGET, COLORS
from data_prep import get_categorical_cols, load_clean_data, set_plot_style

plt = set_plot_style()
OUTPUT_DIR.mkdir(exist_ok=True)
pd.set_option("display.width", 120)

df = load_clean_data()
cat_cols = get_categorical_cols(df)
print("Rows after cleaning:", len(df))
print("Numeric columns:", len(NUMERIC_COLS), "| Category columns:", len(cat_cols))

# target balance - about 79% are not severe, so accuracy on its own is misleading
counts = df[TARGET].value_counts().sort_index()
print("\nTarget balance:")
print(f"  not severe (0): {counts[0]}  ({counts[0] / len(df):.1%})")
print(f"  severe (1):     {counts[1]}  ({counts[1] / len(df):.1%})")

# count unknowns per column and compare the severe rate for known vs unknown
rows = []
for col in cat_cols + NUMERIC_COLS:
    if col in NUMERIC_COLS:
        is_unknown = df[col].isna()
        kind = "blank"
    else:
        is_unknown = df[col] == "Unknown"
        kind = "'Unknown'"

    n_unknown = int(is_unknown.sum())
    if n_unknown == 0:
        continue

    rows.append({
        "column": col,
        "type": kind,
        "n_unknown": n_unknown,
        "pct_unknown": n_unknown / len(df),
        "severe_rate_if_unknown": df[is_unknown][TARGET].mean(),
        "severe_rate_if_known": df[~is_unknown][TARGET].mean(),
    })

report = pd.DataFrame(rows)
report = report.sort_values("pct_unknown", ascending=False)
report.to_csv(OUTPUT_DIR / "unknowns_report.csv", index=False)

show = report.copy()
for c in ["pct_unknown", "severe_rate_if_unknown", "severe_rate_if_known"]:
    show[c] = (show[c] * 100).round(1)
print("\nUnknowns per column (rates in %):")
print(show.to_string(index=False))
print(f"Overall severe rate: {df[TARGET].mean():.1%}")

# if we dropped every row with an unknown, there would be almost nothing left
cat_unknown = (df[cat_cols] == "Unknown").any(axis=1)
num_missing = df[NUMERIC_COLS].isna().any(axis=1)
has_unknown = cat_unknown | num_missing
kept = df[~has_unknown]
print("\nRows with at least one unknown:", has_unknown.sum(), "of", len(df))
print("Rows left if we drop them:", len(kept), "| severe:", int(kept[TARGET].sum()))

# bar chart of % unknown per column
plot_df = report.sort_values("pct_unknown")
pct = plot_df["pct_unknown"] * 100

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.barh(plot_df["column"], pct, color=COLORS["logistic_regression"], height=0.6)

for i in range(len(pct)):
    v = pct.iloc[i]
    if v < 1:
        label = "<1%"
    else:
        label = f"{v:.0f}%"
    ax.text(v + 0.8, i, label, va="center", color=COLORS["ink_2"], fontsize=9)

ax.set_xlabel("% of crashes where the value is Unknown or blank")
ax.set_title("How much is unknown in each column")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, pct.max() + 8)
fig.savefig(OUTPUT_DIR / "unknowns_by_column.png")
print("\nSaved chart:", OUTPUT_DIR / "unknowns_by_column.png")
