from datasets import load_dataset
import pandas as pd

def load_noisyhate():

    dataset = load_dataset("NoisyHate/Noisy_Hate_Data")

    print("Available splits:", dataset.keys())

    if "clean" in dataset and "pert" in dataset:
        df_clean = pd.DataFrame(dataset["clean"])
        df_pert = pd.DataFrame(dataset["pert"])

        if "id" in df_clean.columns and "id" in df_pert.columns:
            df = df_clean.merge(df_pert, on="id", suffixes=("_clean", "_pert"))
        else:
            df = pd.concat([df_clean, df_pert], axis=1)

        print("Loaded clean and perturbed splits successfully.")
        return df

    else:
        split_name = list(dataset.keys())[0]
        df = pd.DataFrame(dataset[split_name])
        print(f"Loaded single split: {split_name}")
        print("Columns:", df.columns.tolist())
        print(df.head())
        return df
