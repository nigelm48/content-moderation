import matplotlib.pyplot as plt

def plot_comparison(results_dict, metric="mean_drop", save_path="results_plot.png"):
    """
    results_dict: dict of name -> pandas.DataFrame (single-row summary)
    metric: which column to plot (e.g., mean_drop)
    """
    labels = []
    means = []
    for name, df in results_dict.items():
        # Expect df to have the metric column
        if metric in df.columns:
            labels.append(name)
            # pull single value safely
            means.append(float(df[metric].iloc[0]))
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, means)
    plt.ylabel(f'{metric} (clean - perturbed)')
    plt.title('Mean drop in toxicity score by condition')
    plt.xticks(rotation=30, ha='right')

    # Add numeric labels on bars
    for b in bars:
        h = b.get_height()
        plt.annotate(f'{h:.3f}', xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
