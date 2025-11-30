from data.noisyhate import load_noisyhate
from data.automated import automated_perturbation
from models.detoxify import evaluate_toxicity
from mitigations.normalisation import normalise_text
from mitigations.detection_fallback import detect_and_fallback
from evaluation.results import compare_toxicity_scores
from evaluation.visualisation import plot_comparison, plot_scatter, plot_box
from models.perspective import evaluate_perspective
import pandas as pd

def main():
    print("Loading data...")
    df = load_noisyhate()

    # Take a subset for faster testing
    clean_texts = df["clean_version"].head(100).tolist()
    human_texts = df["perturbed_version"].head(100).tolist()

    # Step 1: Baseline (Detoxify)
    print("Evaluating Detoxify on clean texts...")
    clean_scores = evaluate_toxicity(clean_texts)

    print("Applying normalisation mitigation (clean)...")
    mitigated_clean_texts = [normalise_text(t) for t in clean_texts]
    mitigated_clean_scores = evaluate_toxicity(mitigated_clean_texts)

    print("Applying detection + fallback mitigation (clean)...")
    fallback_clean_texts = detect_and_fallback(clean_texts, fallback_fn=lambda x: x)
    fallback_clean_scores = evaluate_toxicity(fallback_clean_texts)

    print("Evaluating Detoxify on human perturbed texts...")
    human_scores = evaluate_toxicity(human_texts)

    print("Generating automated perturbations...")
    auto_texts = automated_perturbation(clean_texts, num_examples=100)
    auto_texts = [str(t) if t is not None else "" for t in auto_texts]
    auto_scores = evaluate_toxicity(auto_texts)

    # Step 2: Mitigation strategies (Detoxify)
    print("Applying normalisation mitigation (human)...")
    mitigated_human_texts = [normalise_text(t) for t in human_texts]
    mitigated_human_scores = evaluate_toxicity(mitigated_human_texts)

    print("Applying detection + fallback mitigation (human)...")
    fallback_human_texts = detect_and_fallback(human_texts, fallback_fn=lambda x: x)
    fallback_human_scores = evaluate_toxicity(fallback_human_texts)

    print("Applying normalisation mitigation (automated)...")
    mitigated_auto_texts = [normalise_text(t) for t in auto_texts]
    mitigated_auto_scores = evaluate_toxicity(mitigated_auto_texts)

    print("Applying detection + fallback mitigation (automated)...")
    fallback_auto_texts = detect_and_fallback(auto_texts, fallback_fn=lambda x: x)
    fallback_auto_scores = evaluate_toxicity(fallback_auto_texts)

    # Step 3: Perspective API
    try:
        print("\nEvaluating Perspective API on clean texts...")
        persp_clean = evaluate_perspective(clean_texts)

        print("Applying normalisation mitigation (Perspective, clean)...")
        persp_clean_norm = evaluate_perspective(mitigated_clean_texts)

        print("Applying detection + fallback mitigation (Perspective, clean)...")
        persp_clean_fallback = evaluate_perspective(fallback_clean_texts)

        print("Evaluating Perspective API on human perturbed texts...")
        persp_human = evaluate_perspective(human_texts)

        print("Applying normalisation mitigation (Perspective, human)...")
        persp_human_norm = evaluate_perspective(mitigated_human_texts)

        print("Applying detection + fallback mitigation (Perspective, human)...")
        persp_human_fallback = evaluate_perspective(fallback_human_texts)

        print("Evaluating Perspective API on automated perturbed texts...")
        persp_auto = evaluate_perspective(auto_texts)

        print("Applying normalisation mitigation (Perspective, auto)...")
        persp_auto_norm = evaluate_perspective(mitigated_auto_texts)

        print("Applying detection + fallback mitigation (Perspective, auto)...")
        persp_auto_fallback = evaluate_perspective(fallback_auto_texts)

        persp_results = {
            "perspective_clean_norm": compare_toxicity_scores(persp_clean, persp_clean_norm),
            "perspective_clean_fallback": compare_toxicity_scores(persp_clean, persp_clean_fallback),
            "perspective_human_drop": compare_toxicity_scores(persp_clean, persp_human),
            "perspective_human_norm": compare_toxicity_scores(persp_clean, persp_human_norm),
            "perspective_human_fallback": compare_toxicity_scores(persp_clean, persp_human_fallback),
            "perspective_auto_drop": compare_toxicity_scores(persp_clean, persp_auto),
            "perspective_auto_norm": compare_toxicity_scores(persp_clean, persp_auto_norm),
            "perspective_auto_fallback": compare_toxicity_scores(persp_clean, persp_auto_fallback),
        }

    except Exception as e:
        print(f"\n⚠️ Skipping Perspective API analysis due to error: {e}")
        persp_results = {}

    # Step 4: Combine results
    print("\nComparing Detoxify results...")
    result_summary = {
        "human_drop": compare_toxicity_scores(clean_scores, human_scores),
        "auto_drop": compare_toxicity_scores(clean_scores, auto_scores),
        "clean_norm": compare_toxicity_scores(clean_scores, mitigated_clean_scores),
        "clean_fallback": compare_toxicity_scores(clean_scores, fallback_clean_scores),
        "mitigated_human": compare_toxicity_scores(clean_scores, mitigated_human_scores),
        "fallback_human": compare_toxicity_scores(clean_scores, fallback_human_scores),
        "mitigated_auto": compare_toxicity_scores(clean_scores, mitigated_auto_scores),
        "fallback_auto": compare_toxicity_scores(clean_scores, fallback_auto_scores),
        **persp_results
    }

    # Step 5: Print results
    for k, v in result_summary.items():
        print(f"\n{k}:\n{v}")

    # Step 6: Visualisations
    print("\nGenerating bar plot...")
    plot_comparison(result_summary, metric="mean_drop", save_path="results_bar.png")

    print("Generating scatter plot...")
    # For scatter, just comparing human vs auto drops as an example
    plot_scatter(
        results_dict_x=result_summary,
        results_dict_y=result_summary,
        metric="mean_drop",
        save_path="results_scatter.png"
    )

    print("Generating box plot...")
    # Box plot requires raw scores
    raw_scores_dict = {
        "clean": clean_scores,
        "human": human_scores,
        "auto": auto_scores,
        "mitigated_human": mitigated_human_scores,
        "fallback_human": fallback_human_scores,
        "mitigated_auto": mitigated_auto_scores,
        "fallback_auto": fallback_auto_scores
    }
    plot_box(raw_scores_dict, metric="toxicity", save_path="results_box.png")

    print("\n✅ Experiment complete.")

if __name__ == "__main__":
    main()