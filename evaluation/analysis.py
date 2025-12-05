from sentence_transformers import SentenceTransformer, util
import pandas as pd


model = SentenceTransformer("all-mpnet-base-v2")

def compute_similarity(clean_texts, perturbed_texts):

    clean_emb = model.encode(clean_texts, convert_to_tensor=True, show_progress_bar=True)
    pert_emb = model.encode(perturbed_texts, convert_to_tensor=True, show_progress_bar=True)

    sims = util.cos_sim(clean_emb, pert_emb)
    diagonal_sims = [float(sims[i][i]) for i in range(len(clean_texts))]

    return pd.DataFrame({
        "clean": clean_texts,
        "perturbed": perturbed_texts,
        "similarity": diagonal_sims
    })


def compare_similarity(clean, human, auto):
    
    human_df = compute_similarity(clean, human)
    auto_df = compute_similarity(clean, auto)

    summary = pd.DataFrame({
        "type": ["human", "automated"],
        "mean_similarity": [
            human_df["similarity"].mean(),
            auto_df["similarity"].mean(),
        ],
        "median_similarity": [
            human_df["similarity"].median(),
            auto_df["similarity"].median(),
        ],
        "min_similarity": [
            human_df["similarity"].min(),
            auto_df["similarity"].min(),
        ],
        "max_similarity": [
            human_df["similarity"].max(),
            auto_df["similarity"].max(),
        ]
    })

    return human_df, auto_df, summary

