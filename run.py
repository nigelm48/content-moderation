from data.noisyhate import load_noisyhate
from data.automated import automated_perturbation
from models.detoxify_model import evaluate_toxicity
from models.hatexplain import hatexplain
from mitigations.normalisation import normalise_text
from mitigations.detection_fallback import detect_and_fallback
from evaluation.results import compare_toxicity_scores
from evaluation.label_changes import evaluate_label_changes
from evaluation.similarities import compare_similarity
from evaluation.visualisation import plot_bar, plot_scatter, plot_box, plot_label_changes, plot_similarity_distributions
from models.perspective import evaluate_perspective
import pandas as pd

def main():
    print("Loading data...")
    df = load_noisyhate()

    # Take a subset for faster testing
    clean_texts = df["clean_version"].head(100).tolist()
    human_texts = df["perturbed_version"].head(100).tolist()

    # Baseline (Detoxify)
    print("Evaluating Detoxify on clean texts...")
    clean_scores = evaluate_toxicity(clean_texts)

    print("Applying normalisation mitigation (clean)...")
    norm_clean_texts = [normalise_text(t) for t in clean_texts]
    norm_clean_scores = evaluate_toxicity(norm_clean_texts)

    print("Applying detection + fallback mitigation (clean)...")
    fallback_clean_texts = detect_and_fallback(clean_texts, fallback_fn=lambda x: x)
    fallback_clean_scores = evaluate_toxicity(fallback_clean_texts)

    print("Evaluating Detoxify on human perturbed texts...")
    human_scores = evaluate_toxicity(human_texts)

    print("Generating automated perturbations...")
    auto_texts = automated_perturbation(clean_texts, num_examples=100)
    auto_texts = [str(t) if t is not None else "" for t in auto_texts]
    auto_scores = evaluate_toxicity(auto_texts)

    # Mitigation strategies (Detoxify)
    print("Applying normalisation mitigation (human)...")
    norm_human_texts = [normalise_text(t) for t in human_texts]
    norm_human_scores = evaluate_toxicity(norm_human_texts)

    print("Applying detection + fallback mitigation (human)...")
    fallback_human_texts = detect_and_fallback(human_texts, fallback_fn=lambda x: x)
    fallback_human_scores = evaluate_toxicity(fallback_human_texts)

    print("Applying normalisation mitigation (automated)...")
    norm_auto_texts = [normalise_text(t) for t in auto_texts]
    norm_auto_scores = evaluate_toxicity(norm_auto_texts)

    print("Applying detection + fallback mitigation (automated)...")
    fallback_auto_texts = detect_and_fallback(auto_texts, fallback_fn=lambda x: x)
    fallback_auto_scores = evaluate_toxicity(fallback_auto_texts)

    # HateXplain
    print("\nRunning HateXplain on clean texts...")
    hx_clean = hatexplain(clean_texts)

    print("Running HateXplain on human-perturbed texts...")
    hx_human = hatexplain(human_texts)

    hx_human_label_stats = evaluate_label_changes(hx_clean, hx_human)

    print("Running HateXplain on automated perturbations...")
    hx_auto = hatexplain(auto_texts)

    hx_auto_label_stats = evaluate_label_changes(hx_clean, hx_auto)

    hx_human_norm = hatexplain(norm_human_texts)
    hx_human_fallback = hatexplain(fallback_human_texts)

    hx_human_norm_label_stats = evaluate_label_changes(hx_clean, hx_human_norm)
    hx_human_fallback_label_stats = evaluate_label_changes(hx_clean, hx_human_fallback)

    hx_auto_norm = hatexplain(norm_auto_texts)
    hx_auto_fallback = hatexplain(fallback_auto_texts)

    hx_auto_norm_label_stats = evaluate_label_changes(hx_clean, hx_auto_norm)
    hx_auto_fallback_label_stats = evaluate_label_changes(hx_clean, hx_auto_fallback)

    hx_results = {
    "human": hx_human_label_stats,
    "auto": hx_auto_label_stats,
    "human_norm": hx_human_norm_label_stats,
    "human_fallback": hx_human_fallback_label_stats,
    "auto_norm": hx_auto_norm_label_stats,
    "auto_fallback": hx_auto_fallback_label_stats,
    }


    # Perspective API
    try:
        print("\nEvaluating Perspective API on clean texts...")
        persp_clean = evaluate_perspective(clean_texts)

        print("Applying normalisation mitigation (Perspective, clean)...")
        persp_clean_norm = evaluate_perspective(norm_clean_texts)

        print("Applying detection + fallback mitigation (Perspective, clean)...")
        persp_clean_fallback = evaluate_perspective(fallback_clean_texts)

        print("Evaluating Perspective API on human perturbed texts...")
        persp_human = evaluate_perspective(human_texts)

        print("Applying normalisation mitigation (Perspective, human)...")
        persp_human_norm = evaluate_perspective(norm_human_texts)

        print("Applying detection + fallback mitigation (Perspective, human)...")
        persp_human_fallback = evaluate_perspective(fallback_human_texts)

        print("Evaluating Perspective API on automated perturbed texts...")
        persp_auto = evaluate_perspective(auto_texts)

        print("Applying normalisation mitigation (Perspective, auto)...")
        persp_auto_norm = evaluate_perspective(norm_auto_texts)

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

    # Combine results
    print("\nComparing Detoxify results...")
    result_summary = {
        "human_drop": compare_toxicity_scores(clean_scores, human_scores),
        "auto_drop": compare_toxicity_scores(clean_scores, auto_scores),
        "clean_norm": compare_toxicity_scores(clean_scores, norm_clean_scores),
        "clean_fallback": compare_toxicity_scores(clean_scores, fallback_clean_scores),
        "norm_human": compare_toxicity_scores(clean_scores, norm_human_scores),
        "fallback_human": compare_toxicity_scores(clean_scores, fallback_human_scores),
        "norm_auto": compare_toxicity_scores(clean_scores, norm_auto_scores),
        "fallback_auto": compare_toxicity_scores(clean_scores, fallback_auto_scores),
        **persp_results
    }

    # Print results
    for k, v in result_summary.items():
        print(f"\n{k}:\n{v}")

    # Visualisations
    print("\nGenerating bar chart for toxicity...")
    plot_bar(result_summary, metric="mean_drop", save_path="results_bar.png")

    print("\nGenerating label-change bar chart...")
    plot_label_changes(hx_results, save_path="HX_label_changes.png")

    print("Generating scatter graph...")
    plot_scatter(
        results_dict_x=result_summary,
        results_dict_y=persp_results,
        metric="mean_drop",
        save_path="results_scatter.png"
    )

    print("Generating box plot...")
    # Box plot requires raw scores
    raw_scores = {
        "clean": clean_scores,
        "human": human_scores,
        "auto": auto_scores,
        "norm_human": norm_human_scores,
        "fallback_human": fallback_human_scores,
        "norm_auto": norm_auto_scores,
        "fallback_auto": fallback_auto_scores
    }
    plot_box(raw_scores, metric="toxicity", save_path="results_box.png")

    # Testing semantic similarity
    human_df, auto_df, summary = compare_similarity(clean_texts, human_texts, auto_texts)
    print("\nSemantic similarity results:")
    print(summary)
    plot_similarity_distributions(human_df, auto_df, save_path="semantic_similarity_boxplot.png")

    print("\n✅ Experiment complete.")

if __name__ == "__main__":
    main()