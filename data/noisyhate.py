from datasets import load_dataset
import pandas as pd

def load_noisyhate():
    """
    Loads the NoisyHate dataset (clean and perturbed versions) 
    from Hugging Face and merges them into a single DataFrame.
    """

    # Load dataset from Hugging Face
    dataset = load_dataset("NoisyHate/Noisy_Hate_Data")

    print("Available splits:", dataset.keys())

    # Check if 'clean' and 'pert' splits exist
    if "clean" in dataset and "pert" in dataset:
        df_clean = pd.DataFrame(dataset["clean"])
        df_pert = pd.DataFrame(dataset["pert"])

        # Merge if both have an ID column
        if "id" in df_clean.columns and "id" in df_pert.columns:
            df = df_clean.merge(df_pert, on="id", suffixes=("_clean", "_pert"))
        else:
            df = pd.concat([df_clean, df_pert], axis=1)

        print("Loaded clean and perturbed splits successfully.")
        return df

    # Otherwise, fallback to single-split mode
    else:
        split_name = list(dataset.keys())[0]  # pick the first available split
        df = pd.DataFrame(dataset[split_name])
        print(f"Loaded single split: {split_name}")
        print("Columns:", df.columns.tolist())
        print(df.head())
        return df


if __name__ == "__main__":
    df = load_noisyhate()
    print(f"\n✅ Dataset loaded successfully with {len(df)} rows.")
