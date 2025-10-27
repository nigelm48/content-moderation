from textattack.augmentation import WordNetAugmenter
import pandas as pd

def automated_perturbation(texts, num_examples=5):
    """
    Generates machine-based obfuscations using synonym replacement.
    """
    augmenter = WordNetAugmenter()
    perturbed = [augmenter.augment(text)[0] if text else text for text in texts[:num_examples]]
    return perturbed

if __name__ == "__main__":
    sample_texts = ["I hate you", "You're so dumb", "Go away"]
    perturbed = automated_perturbation(sample_texts)
    for p in perturbed:
        print(p)
