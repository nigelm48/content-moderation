from textattack.augmentation import WordNetAugmenter
import random

HOMOGLYPHS = {
    "a": "а", "e": "е", "o": "о","c": "с","p": "р","x": "х","y": "у","i": "і","s": "ѕ","h": "һ"
}

LEET_MAP = {
    "a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"
}


def random_char_noise(text):
    if len(text) < 2:
        return text
    i = random.randint(0, len(text) - 1)
    ops = [
        text[:i] + text[i]*2 + text[i+1:],
        text[:i] + "" + text[i+1:],
        text[:i] + random.choice("!?.*") + text[i:],
    ]
    return random.choice(ops)


def homoglyph_attack(text):
    return "".join(HOMOGLYPHS.get(c, c) for c in text)


def leetspeak(text):
    return "".join(LEET_MAP.get(c.lower(), c) for c in text)


def random_spacing(text):
    return "".join(c + (" " if random.random() < 0.15 else "") for c in text)


def random_casing(text):
    return "".join(c.upper() if random.random() < 0.3 else c.lower() for c in text)


def punctuation_injection(text):
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

    attacks = [
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
            out = t

        perturbed.append(out)

    return perturbed
