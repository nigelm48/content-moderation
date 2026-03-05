from data.noisyhate import load_noisyhate
from data.automated import automated_perturbation
from models.detoxify_model import evaluate_toxicity
from models.hatexplain import hatexplain
from mitigations.normalisation import normalise_text
from mitigations.detection_fallback import detect_and_fallback
from evaluation.results import compare_toxicity_scores
from evaluation.label_changes import evaluate_label_changes
from evaluation.analysis import compare_similarity, compute_levenshtein, summarise_levenshtein, compute_readability, summarise_readability
from evaluation.visualisation import plot_bar, plot_scatter, plot_box, plot_label_changes, plot_similarity_distributions, plot_levenshtein_box, plot_readability_box
from models.perspective import evaluate_perspective
import pandas as pd

def main():
    print("Loading data...")
    df = load_noisyhate()


    clean_texts = df["clean_version"].head(100).tolist()
    human_texts = df["perturbed_version"].head(100).tolist()

    print("Evaluating Detoxify on clean texts...")
    clean_scores = evaluate_toxicity(clean_texts)

    print("Applying normalisation mitigation (clean)...")
    clean_norm_texts = [normalise_text(t) for t in clean_texts]
    clean_norm_scores = evaluate_toxicity(clean_norm_texts)

    print("Applying detection + fallback mitigation (clean)...")
    clean_fallback_texts = detect_and_fallback(clean_texts, fallback_fn=lambda x: x)
    clean_fallback_scores = evaluate_toxicity(clean_fallback_texts)

    print("Evaluating Detoxify on human perturbed texts...")
    human_scores = evaluate_toxicity(human_texts)

    print("Generating automated perturbations...")
    auto_texts = automated_perturbation(clean_texts, num_examples=len(clean_texts))
    auto_texts = [str(t) if t is not None else "" for t in auto_texts]
    auto_scores = evaluate_toxicity(auto_texts)

    print("Applying normalisation mitigation (human)...")
    human_norm_texts = [normalise_text(t) for t in human_texts]
    human_norm_scores = evaluate_toxicity(human_norm_texts)

    print("Applying detection + fallback mitigation (human)...")
    human_fallback_texts = detect_and_fallback(human_texts, fallback_fn=lambda x: x)
    human_fallback_scores = evaluate_toxicity(human_fallback_texts)

    print("Applying normalisation mitigation (automated)...")
    auto_norm_texts = [normalise_text(t) for t in auto_texts]
    auto_norm_scores = evaluate_toxicity(auto_norm_texts)

    print("Applying detection + fallback mitigation (automated)...")
    auto_fallback_texts = detect_and_fallback(auto_texts, fallback_fn=lambda x: x)
    auto_fallback_scores = evaluate_toxicity(auto_fallback_texts)

    # HateXplain
    print("\nRunning HateXplain on clean texts...")
    hx_clean = hatexplain(clean_texts)

    print("Running HateXplain on human-perturbed texts...")
    hx_human = hatexplain(human_texts)

    hx_human_label_stats = evaluate_label_changes(hx_clean, hx_human)

    print("Running HateXplain on automated perturbations...")
    hx_auto = hatexplain(auto_texts)

    hx_auto_label_stats = evaluate_label_changes(hx_clean, hx_auto)

    hx_human_norm = hatexplain(human_norm_texts)
    hx_human_fallback = hatexplain(human_fallback_texts)

    hx_human_norm_label_stats = evaluate_label_changes(hx_clean, hx_human_norm)
    hx_human_fallback_label_stats = evaluate_label_changes(hx_clean, hx_human_fallback)

    hx_auto_norm = hatexplain(auto_norm_texts)
    hx_auto_fallback = hatexplain(auto_fallback_texts)

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
    PERSPECTIVE = True

    if PERSPECTIVE:
        try:
            print("\nEvaluating Perspective API on clean texts...")
            persp_clean = evaluate_perspective(clean_texts)

            print("Applying normalisation mitigation (Perspective, clean)...")
            persp_clean_norm = evaluate_perspective(clean_norm_texts)

            print("Applying detection + fallback mitigation (Perspective, clean)...")
            persp_clean_fallback = evaluate_perspective(clean_fallback_texts)

            print("Evaluating Perspective API on human perturbed texts...")
            persp_human = evaluate_perspective(human_texts)

            print("Applying normalisation mitigation (Perspective, human)...")
            persp_human_norm = evaluate_perspective(human_norm_texts)

            print("Applying detection + fallback mitigation (Perspective, human)...")
            persp_human_fallback = evaluate_perspective(human_fallback_texts)

            print("Evaluating Perspective API on automated perturbed texts...")
            persp_auto = evaluate_perspective(auto_texts)

            print("Applying normalisation mitigation (Perspective, auto)...")
            persp_auto_norm = evaluate_perspective(auto_norm_texts)

            print("Applying detection + fallback mitigation (Perspective, auto)...")
            persp_auto_fallback = evaluate_perspective(auto_fallback_texts)

            persp_results = {
                "perspective_human_drop": compare_toxicity_scores(persp_clean, persp_human),
                "perspective_auto_drop": compare_toxicity_scores(persp_clean, persp_auto),
                "perspective_unperturbed_norm": compare_toxicity_scores(persp_clean, persp_clean_norm),
                "perspective_unperturbed_fallback": compare_toxicity_scores(persp_clean, persp_clean_fallback),
                "perspective_human_norm": compare_toxicity_scores(persp_clean, persp_human_norm),
                "perspective_human_fallback": compare_toxicity_scores(persp_clean, persp_human_fallback),
                "perspective_auto_norm": compare_toxicity_scores(persp_clean, persp_auto_norm),
                "perspective_auto_fallback": compare_toxicity_scores(persp_clean, persp_auto_fallback),
            }

        except Exception as e:
            print(f"\n⚠️ Skipping Perspective API analysis due to error: {e}")
            persp_results = {}

    print("\nComparing Detoxify results...")
    result_summary = {
        "detoxify_human_drop": compare_toxicity_scores(clean_scores, human_scores),
        "detoxify_auto_drop": compare_toxicity_scores(clean_scores, auto_scores),
        "detoxify_unperturbed_norm": compare_toxicity_scores(clean_scores, clean_norm_scores),
        "detoxify_unperturbed_fallback": compare_toxicity_scores(clean_scores, clean_fallback_scores),
        "detoxify_human_norm": compare_toxicity_scores(clean_scores, human_norm_scores),
        "detoxify_human_fallback": compare_toxicity_scores(clean_scores, human_fallback_scores),
        "detoxify_auto_norm": compare_toxicity_scores(clean_scores, auto_norm_scores),
        "detoxify_auto_fallback": compare_toxicity_scores(clean_scores, auto_fallback_scores),
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

    print("Generating box plot...")

    raw_scores = {
        "unperturbed": clean_scores,
        "human": human_scores,
        "auto": auto_scores,
        "human_norm": human_norm_scores,
        "human_fallback": human_fallback_scores,
        "auto_norm": auto_norm_scores,
        "auto_fallback": auto_fallback_scores
    }
    plot_box(raw_scores, metric="toxicity", save_path="results_box.png")

    print("Generating scatter chart to compare between detoxify and perspective...")
    plot_scatter(
        scores_x=raw_scores,
        scores_y={
            "unperturbed": persp_clean,
            "human": persp_human,
            "auto": persp_auto,
            "human_norm": persp_human_norm,
            "human_fallback": persp_human_fallback,
            "auto_norm": persp_auto_norm,
            "auto_fallback": persp_auto_fallback,
        },
        label_x="Detoxify Toxicity Score",
        label_y="Perspective Toxicity Score",
        save_path="results_scatter.png"
    )


    # Testing semantic similarity
    human_df, auto_df, summary = compare_similarity(clean_texts, human_texts, auto_texts)
    print("\nSemantic similarity results:")
    print(summary)
    plot_similarity_distributions(human_df, auto_df, save_path="semantic_similarity_boxplot.png")

    # Testing Levenshtein distance
    print("\nCalculating Levenshtein distances...")

    lev_human = compute_levenshtein(clean_texts, human_texts)
    lev_auto = compute_levenshtein(clean_texts, auto_texts)
    lev_human_norm = compute_levenshtein(clean_texts, human_norm_texts)
    lev_auto_norm = compute_levenshtein(clean_texts, auto_norm_texts)
    lev_human_fallback = compute_levenshtein(clean_texts, human_fallback_texts)
    lev_auto_fallback = compute_levenshtein(clean_texts, auto_fallback_texts)

    lev_summaries = [
        summarise_levenshtein(lev_human, "human"),
        summarise_levenshtein(lev_auto, "auto"),
        summarise_levenshtein(lev_human_norm, "human_norm"),
        summarise_levenshtein(lev_auto_norm, "auto_norm"),
        summarise_levenshtein(lev_human_fallback, "human_fallback"),
        summarise_levenshtein(lev_auto_fallback, "auto_fallback")
    ]

    print("\nLevenshtein distance summary:")
    for item in lev_summaries:
        print(item)

    # Plot boxplot
    plot_levenshtein_box({
        "human": lev_human,
        "auto": lev_auto,
        "human_norm": lev_human_norm,
        "auto_norm": lev_auto_norm,
        "human_fallback": lev_human_fallback,
        "auto_fallback": lev_auto_fallback
    }, save_path="levenshtein_boxplot.png")

    # Flesch Reading Ease Tests
    print("\nCalculating the change in Flesch Reading Ease scores...")

    flesch_human = compute_readability(clean_texts, human_texts)
    flesch_auto = compute_readability(clean_texts, auto_texts)
    flesch_human_norm = compute_readability(clean_texts, human_norm_texts)
    flesch_auto_norm = compute_readability(clean_texts, auto_norm_texts)
    flesch_human_fallback = compute_readability(clean_texts, human_fallback_texts)
    flesch_auto_fallback = compute_readability(clean_texts, auto_fallback_texts)

    flesch_results = {
        "human": flesch_human,
        "auto": flesch_auto,
        "human_norm": flesch_human_norm,
        "auto_norm": flesch_auto_norm,
        "human_fallback": flesch_human_fallback,
        "auto_fallback": flesch_auto_fallback
    }

    flesch_summaries = [
        summarise_readability(flesch_human, "human"),
        summarise_readability(flesch_auto, "auto"),
        summarise_readability(flesch_human_norm, "human_norm"),
        summarise_readability(flesch_auto_norm, "auto_norm"),
        summarise_readability(flesch_human_fallback, "human_fallback"),
        summarise_readability(flesch_auto_fallback, "auto_fallback"),
    ]

    print("\nFlesch Reading Ease change summary:")
    for item in flesch_summaries:
        print(item)

    plot_readability_box(flesch_results, save_path="flesch_change_boxplot.png")



    print("\n✅ Experiment complete.")

if __name__ == "__main__":
    main()