import re
from textblob import TextBlob

LEETSPEAK_MAP = {
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "$": "s"
}

def obfuscation_score(text):
    if not text:
        return 0

    length = max(len(text), 1)
    text_l = text.lower()

    non_alpha_ratio = sum(1 for c in text_l if not c.isalpha()) / length
    leet_ratio = sum(1 for c in text_l if c in LEETSPEAK_MAP) / length

    repetition = 1 if re.search(r"(.)\1{3,}", text_l) else 0

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
    t = text.lower()
    for k, v in LEETSPEAK_MAP.items():
        t = t.replace(k, v)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t


def detect_and_spellcheck(texts, threshold=0.35):

    processed_texts = []

    for text in texts:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        score = obfuscation_score(text)
        cleaned = soft_normalise(text)

        if score > threshold:
            try:
                cleaned = str(TextBlob(cleaned).correct())
            except:
                pass

        processed_texts.append(cleaned)

    return processed_texts
