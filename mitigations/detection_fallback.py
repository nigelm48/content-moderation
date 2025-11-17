import re
from textblob import TextBlob
import numpy as np

LEETSPEAK_MAP = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"
}

def obfuscation_score(text):
    """Returns a score (0–1+) estimating how obfuscated a text is."""
    if not text:
        return 0

    length = max(len(text), 1)

    # Ratio of non-alphabetic characters
    non_alpha_ratio = sum(1 for c in text if not c.isalpha()) / length

    # Leetspeak ratio
    leet_ratio = sum(1 for c in text.lower() if c in LEETSPEAK_MAP) / length

    # Repetition / stretched words
    repetition = 1 if re.search(r"(.)\1{3,}", text) else 0

    # Missing vowels (high consonant ratio)
    vowels = sum(1 for c in text.lower() if c in "aeiou")
    consonants = sum(1 for c in text.lower() if c.isalpha()) - vowels
    consonant_ratio = consonants / max(consonants + vowels, 1)

    # Combined score (0–1+)
    score = (
        0.4 * non_alpha_ratio +
        0.3 * leet_ratio +
        0.2 * consonant_ratio +
        0.1 * repetition
    )

    return min(score, 1.5)


def soft_normalise(text):
    """A light normaliser that doesn't wipe structure."""
    text = text.lower()
    for k, v in LEETSPEAK_MAP.items():
        text = text.replace(k, v)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


def detect_and_fallback(texts, fallback_fn, threshold=0.35, correct_spelling=False):
    """
    Improved detection + fallback mitigation.
    - Computes obfuscation score
    - Applies light normalisation
    - Uses fallback model only when needed
    """
    cleaned = []

    for text in texts:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        score = obfuscation_score(text)

        # INITIAL normalisation
        norm = soft_normalise(text)

        # Optional: spell correction (slow)
        if correct_spelling and score > threshold:
            try:
                norm = str(TextBlob(norm).correct())
            except Exception:
                pass  # fail-safe

        cleaned.append(norm)

    # Apply fallback model ONCE to the corrected texts
    return fallback_fn(cleaned)
