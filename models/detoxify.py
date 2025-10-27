from detoxify import Detoxify
import pandas as pd

model = Detoxify('original')

def evaluate_toxicity(texts):
    """
    Runs Detoxify model and returns toxicity scores as a pandas DataFrame.
    """
    results = model.predict(texts)
    return pd.DataFrame(results)

if __name__ == "__main__":
    texts = ["I hate you", "Have a nice day!"]
    scores = evaluate_toxicity(texts)
    print(scores)
