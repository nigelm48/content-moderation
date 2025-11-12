from data.noisyhate import load_noisyhate
from data.automated import automated_perturbation
from models.detoxify import evaluate_toxicity
from mitigations.normalisation import normalise_text
from mitigations.detection_fallback import detect_and_fallback
from evaluation.results import compare_toxicity_scores
from evaluation.visualisation import plot_comparison
import pandas as pd

def main():
    print("Loading data...")
    df = load_noisyhate()

    # Take a subset for faster testing
    texts_clean = df["clean_version"].head(100).tolist()
    texts_human = df["perturbed_version"].head(100).tolist()

    # === Step 1: Baseline evaluations ===
    print("Evaluating Detoxify on clean texts...")
    clean_scores = evaluate_toxicity(texts_clean)

    print("Evaluating Detoxify on human perturbed texts...")
    human_scores = evaluate_toxicity(texts_human)

    print("Generating automated perturbations...")
    auto_texts = automated_perturbation(texts_clean, num_examples=100)
    # Ensure auto_texts length matches expectation and are strings
    auto_texts = [str(t) if t is not None else "" for t in auto_texts]
    auto_scores = evaluate_toxicity(auto_texts)

    # === Step 2: Mitigation strategies on HUMAN obfuscations ===
    print("Applying normalisation mitigation (human)...")
    mitigated_human_texts = [normalise_text(t) for t in texts_human]
    mitigated_human_scores = evaluate_toxicity(mitigated_human_texts)

    print("Applying detection + fallback mitigation (human)...")
    fallback_human_scores = detect_and_fallback(texts_human, fallback_fn=evaluate_toxicity)

    # === Step 3: Mitigation strategies on AUTOMATED obfuscations ===
    print("Applying normalisation mitigation (automated)...")
    mitigated_auto_texts = [normalise_text(t) for t in auto_texts]
    mitigated_auto_scores = evaluate_toxicity(mitigated_auto_texts)

    print("Applying detection + fallback mitigation (automated)...")
    fallback_auto_scores = detect_and_fallback(auto_texts, fallback_fn=evaluate_toxicity)

    # === Step 4: Compare all results ===
    print("Comparing results...")
    result_summary = {
        # baseline drops
        "human_drop": compare_toxicity_scores(clean_scores, human_scores),
        "auto_drop": compare_toxicity_scores(clean_scores, auto_scores),
        # human mitigations
        "mitigated_human": compare_toxicity_scores(clean_scores, mitigated_human_scores),
        "fallback_human": compare_toxicity_scores(clean_scores, fallback_human_scores),
        # automated mitigations
        "mitigated_auto": compare_toxicity_scores(clean_scores, mitigated_auto_scores),
        "fallback_auto": compare_toxicity_scores(clean_scores, fallback_auto_scores),
    }

    for k, v in result_summary.items():
        print(f"\n{k}:\n{v}")

    # === Step 5: Visualise ===
    print("\nGenerating visualisations...")
    plot_comparison(result_summary)

    print("\n✅ Experiment complete.")

if __name__ == "__main__":
    main()
