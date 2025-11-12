from data.noisyhate import load_noisyhate
from data.automated import automated_perturbation
from models.detoxify import evaluate_toxicity
from mitigations.normalisation import normalise_text
from mitigations.detection_fallback import detect_and_fallback
from evaluation.results import compare_toxicity_scores
from evaluation.visualisation import plot_comparison
from models.perspective import evaluate_perspective  # ✅ added
import pandas as pd

def main():
    print("Loading data...")
    df = load_noisyhate()

    # Take a subset for faster testing
    texts_clean = df["clean_version"].head(100).tolist()
    texts_human = df["perturbed_version"].head(100).tolist()

    # === Step 1: Baseline (Detoxify) ===
    print("Evaluating Detoxify on clean texts...")
    clean_scores = evaluate_toxicity(texts_clean)

    print("Evaluating Detoxify on human perturbed texts...")
    human_scores = evaluate_toxicity(texts_human)

    print("Generating automated perturbations...")
    auto_texts = automated_perturbation(texts_clean, num_examples=100)
    auto_texts = [str(t) if t is not None else "" for t in auto_texts]
    auto_scores = evaluate_toxicity(auto_texts)

    # === Step 2: Mitigation strategies (Detoxify) ===
    print("Applying normalisation mitigation (human)...")
    mitigated_human_texts = [normalise_text(t) for t in texts_human]
    mitigated_human_scores = evaluate_toxicity(mitigated_human_texts)

    print("Applying detection + fallback mitigation (human)...")
    fallback_human_scores = detect_and_fallback(texts_human, fallback_fn=evaluate_toxicity)

    print("Applying normalisation mitigation (automated)...")
    mitigated_auto_texts = [normalise_text(t) for t in auto_texts]
    mitigated_auto_scores = evaluate_toxicity(mitigated_auto_texts)

    print("Applying detection + fallback mitigation (automated)...")
    fallback_auto_scores = detect_and_fallback(auto_texts, fallback_fn=evaluate_toxicity)

    # === Step 3: Perspective API ===
    try:
        print("\nEvaluating Perspective API on clean texts...")
        persp_clean = evaluate_perspective(texts_clean)

        print("Evaluating Perspective API on human perturbed texts...")
        persp_human = evaluate_perspective(texts_human)

        print("Evaluating Perspective API on automated perturbed texts...")
        persp_auto = evaluate_perspective(auto_texts)

        print("Applying normalisation mitigation (Perspective, human)...")
        persp_human_norm = evaluate_perspective(mitigated_human_texts)

        print("Applying normalisation mitigation (Perspective, auto)...")
        persp_auto_norm = evaluate_perspective(mitigated_auto_texts)

        print("\nComparing Perspective API results...")
        persp_results = {
            "perspective_human_drop": compare_toxicity_scores(persp_clean, persp_human),
            "perspective_auto_drop": compare_toxicity_scores(persp_clean, persp_auto),
            "perspective_human_norm": compare_toxicity_scores(persp_clean, persp_human_norm),
            "perspective_auto_norm": compare_toxicity_scores(persp_clean, persp_auto_norm),
        }
    except Exception as e:
        print(f"\n⚠️ Skipping Perspective API analysis due to error: {e}")
        persp_results = {}

    # === Step 4: Combine results ===
    print("\nComparing Detoxify results...")
    result_summary = {
        # Detoxify baselines
        "human_drop": compare_toxicity_scores(clean_scores, human_scores),
        "auto_drop": compare_toxicity_scores(clean_scores, auto_scores),
        # Detoxify mitigations
        "mitigated_human": compare_toxicity_scores(clean_scores, mitigated_human_scores),
        "fallback_human": compare_toxicity_scores(clean_scores, fallback_human_scores),
        "mitigated_auto": compare_toxicity_scores(clean_scores, mitigated_auto_scores),
        "fallback_auto": compare_toxicity_scores(clean_scores, fallback_auto_scores),
        # Merge Perspective results
        **persp_results
    }

    for k, v in result_summary.items():
        print(f"\n{k}:\n{v}")

    # === Step 5: Visualise ===
    print("\nGenerating visualisations...")
    plot_comparison(result_summary)

    print("\n✅ Experiment complete.")

if __name__ == "__main__":
    main()
