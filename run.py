from data.noisyhate import load_noisyhate
from data.automated import automated_perturbation
from models.detoxify import evaluate_toxicity
from mitigations.normalisation import normalise_text
from evaluation.results import compare_toxicity_scores
import pandas as pd

def main():
    print("Loading data...")
    df = load_noisyhate()
    
    # Example: take first 100 texts for quick testing
    texts_clean = df["clean_version"].head(100).tolist()
    texts_human = df["perturbed_version"].head(100).tolist()

    
    print("Evaluating Detoxify on clean texts...")
    clean_scores = evaluate_toxicity(texts_clean)
    
    print("Evaluating Detoxify on human perturbed texts...")
    human_scores = evaluate_toxicity(texts_human)
    
    print("Generating automated perturbations...")
    auto_texts = automated_perturbation(texts_clean, num_examples=100)
    auto_scores = evaluate_toxicity(auto_texts)
    
    print("Applying normalisation mitigation...")
    mitigated_texts = [normalise_text(t) for t in texts_human]
    mitigated_scores = evaluate_toxicity(mitigated_texts)
    
    print("Comparing results...")
    result_summary = {
        "human_drop": compare_toxicity_scores(clean_scores, human_scores),
        "auto_drop": compare_toxicity_scores(clean_scores, auto_scores),
        "mitigated_recovery": compare_toxicity_scores(clean_scores, mitigated_scores)
    }

    for k, v in result_summary.items():
        print(f"\n{k}:\n{v}")

if __name__ == "__main__":
    main()
