from sentence_transformers import SentenceTransformer, util
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from textstat import flesch_reading_ease


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

def compute_levenshtein(clean_texts, perturbed_texts):
    distances = [
        levenshtein_distance(clean_texts[i], perturbed_texts[i])
        for i in range(len(clean_texts))
    ]

    return pd.DataFrame({
        "clean": clean_texts,
        "perturbed": perturbed_texts,
        "lev_distance": distances
    })


def summarise_levenshtein(df, label):
    return {
        "type": label,
        "mean": df["lev_distance"].mean(),
        "median": df["lev_distance"].median(),
        "min": df["lev_distance"].min(),
        "max": df["lev_distance"].max()
    }

def compute_readability(clean_texts, perturbed_texts):
    clean_flesch = [flesch_reading_ease(t if isinstance(t, str) else "") for t in clean_texts]
    pert_flesch = [flesch_reading_ease(t if isinstance(t, str) else "") for t in perturbed_texts]
    flesch_change = [pert_flesch[i] - clean_flesch[i] for i in range(len(clean_flesch))]

    return pd.DataFrame({
        "clean": clean_texts,
        "perturbed": perturbed_texts,
        "flesch_clean": clean_flesch,
        "flesch_perturbed": pert_flesch,
        "flesch_change": flesch_change
    })


def summarise_readability(df, label):
    return {
        "type": label,
        "mean": df["flesch_change"].mean(),
        "median": df["flesch_change"].median(),
        "min": df["flesch_change"].min(),
        "max": df["flesch_change"].max()
    }