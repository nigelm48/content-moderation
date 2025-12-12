import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_bar(results_dict, metric="mean_drop", save_path="results_bar.png"):
    """
    Bar chart showing mean drop in toxicity scores across different conditions.
    """
    labels, means = [], []
    for name, df in results_dict.items():
        if metric in df.columns:
            labels.append(name)
            means.append(float(df[metric].iloc[0]))

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, means, color='skyblue')
    plt.ylabel(f'{metric} (clean - perturbed)')
    plt.title('Mean drop in toxicity score by condition')
    plt.xticks(rotation=30, ha='right')

    for b in bars:
        h = b.get_height()
        plt.annotate(f'{h:.3f}', xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_box(results_dict, metric="drop_values", save_path="results_box.png"):
    """
    Boxplot showing distribution of drops.
    """
    all_data = []
    for name, df in results_dict.items():
        if metric in df.columns:
            for val in df[metric].dropna():
                all_data.append({"condition": name, metric: val})

    if not all_data:
        print("No data for boxplot.")
        return

    plot_df = pd.DataFrame(all_data)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="condition", y=metric, data=plot_df)
    plt.xticks(rotation=30, ha='right')
    plt.title(f'Distribution of {metric} by condition')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_scatter(scores_x, scores_y, label_x="Model X", label_y="Model Y", save_path="results_scatter.png"):
    plt.figure(figsize=(8, 8))
    
    all_x, all_y, labels = [], [], []

    for name in scores_x:
        if name in scores_y:

            x_vals = scores_x[name]["toxicity"].tolist()
            y_vals = scores_y[name]["toxicity"].tolist()

            
            n = min(len(x_vals), len(y_vals))

            all_x.extend(x_vals[:n])
            all_y.extend(y_vals[:n])
            labels.extend([name] * n)


    df = pd.DataFrame({
        "x": all_x,
        "y": all_y,
        "condition": labels
    })

    sns.scatterplot(data=df, x="x", y="y", hue="condition", alpha=0.6)

    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title("Raw Toxicity Score Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



def plot_label_changes(results_dict, save_path="label_changes.png"):

    rows = []
    for name, df in results_dict.items():
        rows.append({
            "scenario": name,
            "normal_to_toxic_rate": df["normal_to_toxic_rate"].iloc[0],
            "toxic_to_normal_rate": df["toxic_to_normal_rate"].iloc[0],
        })

    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(12, 6))
    x = range(len(plot_df))

    plt.bar([i - 0.2 for i in x], plot_df["normal_to_toxic_rate"], width=0.4, label="Normal → Toxic")
    plt.bar([i + 0.2 for i in x], plot_df["toxic_to_normal_rate"], width=0.4, label="Toxic → Normal")

    plt.xticks(x, plot_df["scenario"], rotation=45, ha="right")
    plt.ylabel("Rate")
    plt.title("HateXplain Label Change rates")
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()
    
def plot_similarity_distributions(human_df, auto_df, save_path="similarity_boxplot.png"):
    human_df["type"] = "human"
    auto_df["type"] = "automated"

    df = pd.concat([human_df, auto_df])

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="type", y="similarity")
    plt.title("Semantic Similarity: Human vs Automated Perturbations")
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_levenshtein_box(results, save_path="levenshtein_boxplot.png"):
    plt.figure(figsize=(10, 6))

    all_df = []
    for label, df in results.items():
        temp = df.copy()
        temp["type"] = label
        all_df.append(temp)

    full = pd.concat(all_df)

    sns.boxplot(data=full, x="type", y="lev_distance")
    plt.title("Levenshtein Distance Across Perturbation Types")
    plt.xlabel("Perturbation Type")
    plt.ylabel("Character-level Levenshtein Distance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


