import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

def add_correct_swing_column_and_fisher(input_csv: str, output_csv: str):
    """
    Reads the input CSV, computes whether the model predicts
    the correct up/down swing from the previous year's baseline,
    and then performs a Fisher's Exact Test on the contingency table.
    Writes a new CSV with an added column for correctness and
    prints the Fisher's test results.
    """

    # 1. Read the CSV
    df = pd.read_csv(input_csv)

    # 2. Classify actual swing and predicted swing
    #    We'll say "Up" if difference > 0, "Down" if difference <= 0
    #    (Adjust as you like to handle the == 0 case)
    def swing_direction(diff):
        return "Up" if diff > 0 else "Down"

    df["Actual_Change"] = df["Suicide Rate"] - df["Baseline_Predictions"]
    df["Predicted_Change"] = df["Model_Predictions"] - df["Baseline_Predictions"]

    df["Actual_Swing"] = df["Actual_Change"].apply(swing_direction)
    df["Predicted_Swing"] = df["Predicted_Change"].apply(swing_direction)

    # 3. Determine if the swing is correct
    def direction_correct(row):
        return "Correct" if row["Actual_Swing"] == row["Predicted_Swing"] else "Incorrect"

    df["Model_Correct_Swing"] = df.apply(direction_correct, axis=1)

    # 4. Build the 2x2 table for fisher exact test:
    #    Let's define:
    #       a = # of (Actual Up, Predicted Up)
    #       b = # of (Actual Up, Predicted Down)
    #       c = # of (Actual Down, Predicted Up)
    #       d = # of (Actual Down, Predicted Down)
    a = len(df[(df["Actual_Swing"] == "Up") & (df["Predicted_Swing"] == "Up")])
    b = len(df[(df["Actual_Swing"] == "Up") & (df["Predicted_Swing"] == "Down")])
    c = len(df[(df["Actual_Swing"] == "Down") & (df["Predicted_Swing"] == "Up")])
    d = len(df[(df["Actual_Swing"] == "Down") & (df["Predicted_Swing"] == "Down")])

    # Our 2x2 contingency table
    contingency_table = np.array([[a, b],
                                  [c, d]])

    # 5. Fisher's Exact Test
    odds_ratio, p_value = fisher_exact(contingency_table, alternative='two-sided')

    # 6. Print results
    print("Contingency table:")
    print(f"               Predicted Up   Predicted Down")
    print(f"Actual Up          {a}               {b}")
    print(f"Actual Down        {c}               {d}")
    print("\nFisher's Exact Test Results:")
    print(f"Odds ratio = {odds_ratio}")
    print(f"P-value     = {p_value}")

    # 6a. Percent of swings predicted correctly
    total = a + b + c + d
    correct = a + d

    up_total = a + b
    down_total = c + d

    overall_acc = correct / total if total else np.nan
    up_acc = a / up_total if up_total else np.nan          # P(predicted Up | actual Up)
    down_acc = d / down_total if down_total else np.nan    # P(predicted Down | actual Down)

    # Balanced accuracy (macro-average of the two class recalls)
    balanced_acc = np.nanmean([up_acc, down_acc])

    print("\nDirection accuracy:")
    print(f"Overall correct: {correct}/{total} ({overall_acc*100:.2f}%)")
    print(f"Up correct:      {a}/{up_total} ({up_acc*100:.2f}%)")
    print(f"Down correct:    {d}/{down_total} ({down_acc*100:.2f}%)")
    print(f"Balanced acc:    {balanced_acc*100:.2f}%")

    # Optional: save a compact summary CSV
    summary = pd.DataFrame({
        "a_actualUp_predUp": [a],
        "b_actualUp_predDown": [b],
        "c_actualDown_predUp": [c],
        "d_actualDown_predDown": [d],
        "overall_correct": [correct],
        "overall_total": [total],
        "overall_pct": [overall_acc * 100],
        "up_total": [up_total],
        "up_pct_correct": [up_acc * 100],
        "down_total": [down_total],
        "down_pct_correct": [down_acc * 100],
        "balanced_accuracy_pct": [balanced_acc * 100],
        "fisher_odds_ratio": [odds_ratio],
        "fisher_p_value": [p_value],
    })
    # summary.to_csv("swing_direction_summary.csv", index=False)


    # 7. Write out to a new CSV with the extra columns
    #df.to_csv(output_csv, index=False)
    #print(f"\nProcessed file saved to: {output_csv}")

    # contingency_table = np.array([[8, 11], [4, 5]])
    # odds_ratio, p_value = fisher_exact(contingency_table, alternative='two-sided')

    # # 6. Print results
    # print("Contingency table:")
    # print(f"               Predicted Up   Predicted Down")
    # print(f"Actual Up          {a}               {b}")
    # print(f"Actual Down        {c}               {d}")
    # print("\nFisher's Exact Test Results:")
    # print(f"Odds ratio = {odds_ratio}")
    # print(f"P-value     = {p_value}")

if __name__ == "__main__":
    input_file = "predictions.csv"
    output_file = "suicide_data_labeled.csv"
    add_correct_swing_column_and_fisher(input_file, output_file)
