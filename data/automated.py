from textattack.augmentation import WordNetAugmenter
import random
import unicodedata
import numpy as np

# Homoglyph and leetspeak mappings
HOMOGLYPHS = {
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    "A": "A", "E": "E", "O": "O", "P": "P", "C": "C"
}

LEET_MAP = {
    "a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"
}


def random_char_noise(text):
    if len(text) < 2:
        return text
    i = random.randint(0, len(text) - 1)
    ops = [
        text[:i] + text[i]*2 + text[i+1:],             # duplication
        text[:i] + "" + text[i+1:],                    # deletion
        text[:i] + random.choice("!?.*") + text[i:],   # insertion
    ]
    return random.choice(ops)


def homoglyph_attack(text):
    """Replace characters with Unicode lookalikes."""
    return "".join(HOMOGLYPHS.get(c, c) for c in text)


def leetspeak(text):
    """Replace letters with numbers/symbols."""
    return "".join(LEET_MAP.get(c.lower(), c) for c in text)


def random_spacing(text):
    """Insert random spaces inside words."""
    return "".join(c + (" " if random.random() < 0.15 else "") for c in text)


def random_casing(text):
    return "".join(c.upper() if random.random() < 0.3 else c.lower() for c in text)


def punctuation_injection(text):
    """Add random punctuation breaks."""
    punct = ["*", "-", "~", ".", "!", "…"]
    return "".join(c + (random.choice(punct) if random.random() < 0.1 else "") for c in text)

def synonym_substitution(text):
    augmenter = WordNetAugmenter()
    try:
        augmented = augmenter.augment(text)
        if augmented:
            return augmented[0]
    except Exception:
        return text
    return text


def automated_perturbation(texts, num_examples=50):
    """
    Generates randomised automated obfuscations.
    """
    augmenter = WordNetAugmenter()

    attacks = [
        lambda x: augmenter.augment(x)[0],
        synonym_substitution,
        random_char_noise,
        homoglyph_attack,
        leetspeak,
        random_spacing,
        random_casing,
        punctuation_injection,
    ]

    perturbed = []
    for t in texts[:num_examples]:
        if not t:
            perturbed.append("")
            continue

        attack = random.choice(attacks)
        try:
            out = attack(t)
        except Exception:
            out = t  # if there's an error, do nothing

        perturbed.append(out)

    return perturbed



if __name__ == "__main__":
    sample_texts = ["I hate you", "You're so dumb", "Go away"]
    perturbed = automated_perturbation(sample_texts)
    for p in perturbed:
        print(p)
