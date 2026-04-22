from data.noisyhate import load_noisyhate
from data.automated import automated_perturbation
from models.detoxify_model import evaluate_toxicity
from models.hatexplain import hatexplain
from mitigations.normalisation import normalise_text
from mitigations.detection_spellcheck import detect_and_spellcheck
from evaluation.results import compare_toxicity_scores
from evaluation.label_changes import evaluate_label_changes
from evaluation.analysis import compare_similarity, compute_levenshtein, summarise_levenshtein, compute_readability, summarise_readability
from evaluation.visualisation import plot_bar, plot_scatter, plot_box, plot_label_changes, plot_similarity_distributions, plot_levenshtein_box, plot_readability_box
from models.perspective import evaluate_perspective

def main():
    print("Loading dataset...")
    df = load_noisyhate()


    clean_texts = df["clean_version"].head(100).tolist()
    human_texts = df["perturbed_version"].head(100).tolist()

    print("Evaluating Detoxify on clean texts...")
    clean_scores = evaluate_toxicity(clean_texts)

    print("Applying normalisation mitigation on unperturbed texts...")
    clean_norm_texts = [normalise_text(t) for t in clean_texts]
    clean_norm_scores = evaluate_toxicity(clean_norm_texts)

    print("Applying detection + spellcheck mitigation on unperturbed texts...")
    clean_spellcheck_texts = detect_and_spellcheck(clean_texts)
    clean_spellcheck_scores = evaluate_toxicity(clean_spellcheck_texts)

    print("Evaluating Detoxify on human perturbed texts...")
    human_scores = evaluate_toxicity(human_texts)

    print("Generating and evaluating detoxify on automated perturbations...")
    auto_texts = automated_perturbation(clean_texts)
    auto_texts = [str(t) if t is not None else "" for t in auto_texts]
    auto_scores = evaluate_toxicity(auto_texts)

    print("Applying normalisation mitigation on human perturbed texts...")
    human_norm_texts = [normalise_text(t) for t in human_texts]
    human_norm_scores = evaluate_toxicity(human_norm_texts)

    print("Applying detection + spellcheck mitigation on human perturbed texts...")
    human_spellcheck_texts = detect_and_spellcheck(human_texts)
    human_spellcheck_scores = evaluate_toxicity(human_spellcheck_texts)

    print("Applying normalisation mitigation on automated perturbations...")
    auto_norm_texts = [normalise_text(t) for t in auto_texts]
    auto_norm_scores = evaluate_toxicity(auto_norm_texts)

    print("Applying detection + spellcheck mitigation on automated perturbations...")
    auto_spellcheck_texts = detect_and_spellcheck(auto_texts)
    auto_spellcheck_scores = evaluate_toxicity(auto_spellcheck_texts)

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
    hx_human_spellcheck = hatexplain(human_spellcheck_texts)

    hx_human_norm_label_stats = evaluate_label_changes(hx_clean, hx_human_norm)
    hx_human_spellcheck_label_stats = evaluate_label_changes(hx_clean, hx_human_spellcheck)

    hx_auto_norm = hatexplain(auto_norm_texts)
    hx_auto_spellcheck = hatexplain(auto_spellcheck_texts)

    hx_auto_norm_label_stats = evaluate_label_changes(hx_clean, hx_auto_norm)
    hx_auto_spellcheck_label_stats = evaluate_label_changes(hx_clean, hx_auto_spellcheck)

    hx_results = {
    "human": hx_human_label_stats,
    "auto": hx_auto_label_stats,
    "human_norm": hx_human_norm_label_stats,
    "human_spellcheck": hx_human_spellcheck_label_stats,
    "auto_norm": hx_auto_norm_label_stats,
    "auto_spellcheck": hx_auto_spellcheck_label_stats,
    }

    # Perspective API
    try:
        print("\nEvaluating Perspective API on clean texts...")
        persp_clean = evaluate_perspective(clean_texts)

        print("Evaluating normalisation on unperturbed texts with Perspective API...")
        persp_clean_norm = evaluate_perspective(clean_norm_texts)

        print("Evaluating detection + spellcheck mitigation on unperturbed texts with Perspective API...")
        persp_clean_spellcheck = evaluate_perspective(clean_spellcheck_texts)

        print("Evaluating Perspective API on human perturbed texts...")
        persp_human = evaluate_perspective(human_texts)

        print("Evaluating normalisation on human perturbed texts with Perspective API...")
        persp_human_norm = evaluate_perspective(human_norm_texts)

        print("Evaluating detection + spellcheck mitigation on human perturbed texts with Perspective API...")
        persp_human_spellcheck = evaluate_perspective(human_spellcheck_texts)

        print("Evaluating Perspective API on automated perturbed texts...")
        persp_auto = evaluate_perspective(auto_texts)

        print("Evaluating normalisation on automated perturbations with Perspective API...")
        persp_auto_norm = evaluate_perspective(auto_norm_texts)

        print("Evaluating detection + spellcheck mitigation on automated perturbations with Perspective API...")
        persp_auto_spellcheck = evaluate_perspective(auto_spellcheck_texts)

        persp_results = {
            "perspective_human_drop": compare_toxicity_scores(persp_clean, persp_human),
            "perspective_auto_drop": compare_toxicity_scores(persp_clean, persp_auto),
            "perspective_unperturbed_norm": compare_toxicity_scores(persp_clean, persp_clean_norm),
            "perspective_unperturbed_spellcheck": compare_toxicity_scores(persp_clean, persp_clean_spellcheck),
            "perspective_human_norm": compare_toxicity_scores(persp_clean, persp_human_norm),
            "perspective_human_spellcheck": compare_toxicity_scores(persp_clean, persp_human_spellcheck),
            "perspective_auto_norm": compare_toxicity_scores(persp_clean, persp_auto_norm),
            "perspective_auto_spellcheck": compare_toxicity_scores(persp_clean, persp_auto_spellcheck),
        }

    except Exception as e:
        raise RuntimeError(
            "Perspective API evaluation has failed. Please make sure that the API key is set correctly in the .env file and that you have an active internet connection."
        ) from e

    print("\nComparing Detoxify results...")
    result_summary = {
        "detoxify_human_drop": compare_toxicity_scores(clean_scores, human_scores),
        "detoxify_auto_drop": compare_toxicity_scores(clean_scores, auto_scores),
        "detoxify_unperturbed_norm": compare_toxicity_scores(clean_scores, clean_norm_scores),
        "detoxify_unperturbed_spellcheck": compare_toxicity_scores(clean_scores, clean_spellcheck_scores),
        "detoxify_human_norm": compare_toxicity_scores(clean_scores, human_norm_scores),
        "detoxify_human_spellcheck": compare_toxicity_scores(clean_scores, human_spellcheck_scores),
        "detoxify_auto_norm": compare_toxicity_scores(clean_scores, auto_norm_scores),
        "detoxify_auto_spellcheck": compare_toxicity_scores(clean_scores, auto_spellcheck_scores),
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
        "detoxify_unperturbed": clean_scores,
        "detoxify_human": human_scores,
        "detoxify_auto": auto_scores,
        "detoxify_human_norm": human_norm_scores,
        "detoxify_human_spellcheck": human_spellcheck_scores,
        "detoxify_auto_norm": auto_norm_scores,
        "detoxify_auto_spellcheck": auto_spellcheck_scores,

        "perspective_unperturbed": persp_clean,
        "perspective_human": persp_human,
        "perspective_auto": persp_auto,
        "perspective_human_norm": persp_human_norm,
        "perspective_human_spellcheck": persp_human_spellcheck,
        "perspective_auto_norm": persp_auto_norm,
        "perspective_auto_spellcheck": persp_auto_spellcheck,
    }

    plot_box(raw_scores, metric="toxicity", save_path="results_box.png")

    print("Generating scatter chart to compare between detoxify and perspective...")
    plot_scatter(
        scores_x={
            "unperturbed": clean_scores,
            "human": human_scores,
            "auto": auto_scores,
            "human_norm": human_norm_scores,
            "human_spellcheck": human_spellcheck_scores,
            "auto_norm": auto_norm_scores,
            "auto_spellcheck": auto_spellcheck_scores,
        },
        scores_y={
            "unperturbed": persp_clean,
            "human": persp_human,
            "auto": persp_auto,
            "human_norm": persp_human_norm,
            "human_spellcheck": persp_human_spellcheck,
            "auto_norm": persp_auto_norm,
            "auto_spellcheck": persp_auto_spellcheck,
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
    lev_human_spellcheck = compute_levenshtein(clean_texts, human_spellcheck_texts)
    lev_auto_spellcheck = compute_levenshtein(clean_texts, auto_spellcheck_texts)

    lev_summaries = [
        summarise_levenshtein(lev_human, "human"),
        summarise_levenshtein(lev_auto, "auto"),
        summarise_levenshtein(lev_human_norm, "human_norm"),
        summarise_levenshtein(lev_auto_norm, "auto_norm"),
        summarise_levenshtein(lev_human_spellcheck, "human_spellcheck"),
        summarise_levenshtein(lev_auto_spellcheck, "auto_spellcheck")
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
        "human_spellcheck": lev_human_spellcheck,
        "auto_spellcheck": lev_auto_spellcheck
    }, save_path="levenshtein_boxplot.png")

    # Flesch Reading Ease Tests
    print("\nCalculating the change in Flesch Reading Ease scores...")

    flesch_human = compute_readability(clean_texts, human_texts)
    flesch_auto = compute_readability(clean_texts, auto_texts)
    flesch_human_norm = compute_readability(clean_texts, human_norm_texts)
    flesch_auto_norm = compute_readability(clean_texts, auto_norm_texts)
    flesch_human_spellcheck = compute_readability(clean_texts, human_spellcheck_texts)
    flesch_auto_spellcheck = compute_readability(clean_texts, auto_spellcheck_texts)

    flesch_results = {
        "human": flesch_human,
        "auto": flesch_auto,
        "human_norm": flesch_human_norm,
        "auto_norm": flesch_auto_norm,
        "human_spellcheck": flesch_human_spellcheck,
        "auto_spellcheck": flesch_auto_spellcheck
    }

    flesch_summaries = [
        summarise_readability(flesch_human, "human"),
        summarise_readability(flesch_auto, "auto"),
        summarise_readability(flesch_human_norm, "human_norm"),
        summarise_readability(flesch_auto_norm, "auto_norm"),
        summarise_readability(flesch_human_spellcheck, "human_spellcheck"),
        summarise_readability(flesch_auto_spellcheck, "auto_spellcheck"),
    ]

    print("\nFlesch Reading Ease change summary:")
    for item in flesch_summaries:
        print(item)

    plot_readability_box(flesch_results, save_path="flesch_change_boxplot.png")



    print("\nExperiment complete.")

if __name__ == "__main__":
    main()
