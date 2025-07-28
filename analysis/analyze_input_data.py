import pandas as pd
import numpy as np

# Path to your CSV
FILE = "../data/merged_data.csv"

# 1. Load data
df = pd.read_csv(FILE)

# 2. Coerce everything that looks numeric to numbers (non‑numeric -> NaN)
df_num = df.apply(pd.to_numeric, errors="coerce")

# 3. Count totals
total_numeric_vals = df_num.notna().sum().sum()
total_zeros        = (df_num == 0).sum().sum()
total_nonzeros     = ((df_num != 0) & df_num.notna()).sum().sum()

# 4. Percentages
pct_zeros    = (total_zeros / total_numeric_vals) * 100 if total_numeric_vals else np.nan
pct_nonzeros = (total_nonzeros / total_numeric_vals) * 100 if total_numeric_vals else np.nan

print(f"Total numeric values: {total_numeric_vals}")
print(f"Total zeros:          {total_zeros} ({pct_zeros:.2f}%)")
print(f"Total nonzeros:       {total_nonzeros} ({pct_nonzeros:.2f}%)")

# Optional: save a quick summary CSV
summary = pd.DataFrame({
    "Total_Numeric_Values": [total_numeric_vals],
    "Total_Zeros": [total_zeros],
    "Pct_Zeros": [pct_zeros],
    "Total_Nonzeros": [total_nonzeros],
    "Pct_Nonzeros": [pct_nonzeros]
})
summary.to_csv("../data/zero_nonzero_summary.csv", index=False)
print("Saved summary to ../data/zero_nonzero_summary.csv")



# Masks
train_mask = df["Year"] <= 2019
val_mask   = df["Year"].between(2020, 2021, inclusive="both")

# Counts
train_count = train_mask.sum()
val_count   = val_mask.sum()

print(f"Training rows (<=2019): {train_count}")
print(f"Validation rows (2020-2021): {val_count}")
