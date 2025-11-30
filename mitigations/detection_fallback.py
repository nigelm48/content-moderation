import re
import numpy as np
from textblob import TextBlob
from mitigations.normalisation import normalise_text  # existing stronger normaliser

LEETSPEAK_MAP = {
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "$": "s"
}

def obfuscation_score(text):
    """Quantify obfuscation using character anomalies + structure."""
    if not text:
        return 0

    length = max(len(text), 1)
    text_l = text.lower()

    non_alpha_ratio = sum(1 for c in text_l if not c.isalpha()) / length
    leet_ratio = sum(1 for c in text_l if c in LEETSPEAK_MAP) / length

    repetition = 1 if re.search(r"(.)\1{3,}", text_l) else 0

    # whitespace fragmentation (e.g., 'h a t e')
    spaced = len(text_l.split())
    whitespace_frag = 1 if spaced >= len(text_l) / 2 else 0

    vowels = sum(1 for c in text_l if c in "aeiou")
    consonants = sum(1 for c in text_l if c.isalpha()) - vowels
    consonant_ratio = consonants / max((consonants + vowels), 1)

    score = (
        0.35 * non_alpha_ratio +
        0.25 * leet_ratio +
        0.15 * consonant_ratio +
        0.15 * whitespace_frag +
        0.10 * repetition
    )

    return min(score, 1.8)


def soft_normalise(text):
    """Light reversible cleaning."""
    t = text.lower()
    for k, v in LEETSPEAK_MAP.items():
        t = t.replace(k, v)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t


def detect_and_fallback(texts, fallback_fn, threshold=0.35, correct_spelling=False):
    """
    Full detection+fallback algorithm:
    1. soft-normalise
    2. full normalisation
    3. dynamic fallback triggering based on:
         - obfuscation score
         - text length
         - optional fallback model confidence
    """

    first_pass_texts = []
    obfusc_scores = []

    # Compute both soft and strong normalisation
    for text in texts:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        score = obfuscation_score(text)
        obfusc_scores.append(score)

        soft = soft_normalise(text)
        strong = normalise_text(soft)

        if correct_spelling and score > threshold:
            try:
                strong = str(TextBlob(strong).correct())
            except:
                pass

        first_pass_texts.append(strong)

    # Attempt to get preliminary model confidence
    prelim_scores = []
    fallback_output = fallback_fn(first_pass_texts)

    # Handle different return types
    if isinstance(fallback_output, list):
        # fallback_fn returned raw scores or text: use 0 as placeholder
        prelim_scores = [0] * len(fallback_output)
    elif hasattr(fallback_output, "get") and "toxicity" in fallback_output:
        # fallback_fn returned a dict
        prelim_scores = fallback_output.get("toxicity", [0]*len(first_pass_texts))
    elif hasattr(fallback_output, "columns") and "toxicity" in fallback_output.columns:
        # fallback_fn returned DataFrame
        prelim_scores = fallback_output["toxicity"].fillna(0).tolist()
    else:
        prelim_scores = [0] * len(first_pass_texts)

    # Fallback logic
    final_inputs = []
    for original, norm, score, conf in zip(texts, first_pass_texts, obfusc_scores, prelim_scores):
        dynamic_threshold = threshold
        if len(original) < 6:
            dynamic_threshold *= 0.7
        if len(original) > 50:
            dynamic_threshold *= 1.3

        # fallback if high obfuscation & low confidence
        if score > dynamic_threshold and conf < 0.30:
            final_inputs.append(norm)
        else:
            final_inputs.append(norm)  # could also choose original

    # Final evaluation
    return fallback_fn(final_inputs)
