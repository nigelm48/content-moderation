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


def plot_scatter(results_dict_x, results_dict_y, metric="mean_drop", save_path="results_scatter.png"):
    """
    Scatter plot to compare perspective api vs detoxify.
    """
    x_vals, y_vals, labels = [], [], []
    for name in results_dict_x:
        if metric in results_dict_x[name].columns and name in results_dict_y:
            x_vals.append(float(results_dict_x[name][metric].iloc[0]))
            y_vals.append(float(results_dict_y[name][metric].iloc[0]))
            labels.append(name)

    plt.figure(figsize=(8, 8))
    plt.scatter(x_vals, y_vals)
    for i, label in enumerate(labels):
        plt.text(x_vals[i], y_vals[i], label)
    plt.xlabel(f"{metric} (set X)")
    plt.ylabel(f"{metric} (set Y)")
    plt.title(f"Scatter plot comparing {metric} between two conditions")
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
    
