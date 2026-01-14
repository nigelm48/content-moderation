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

if __name__ == "__main__":
    df_clean = pd.DataFrame({"toxicity": [0.8, 0.6, 0.9]})
    df_pert = pd.DataFrame({"toxicity": [0.5, 0.4, 0.6]})
    print(compare_toxicity_scores(df_clean, df_pert))
