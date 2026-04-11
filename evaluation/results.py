import pandas as pd

def compare_toxicity_scores(df_clean, df_pert, label="toxicity"):
    diffs = df_clean[label] - df_pert[label]
    summary = {
        "mean_drop": diffs.mean(),
        "median_drop": diffs.median(),
        "min_drop": diffs.min(),
        "max_drop": diffs.max(),
    }
    return pd.DataFrame([summary])
