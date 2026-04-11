from detoxify import Detoxify
import pandas as pd

model = Detoxify('original')

def evaluate_toxicity(texts):
    results = model.predict(texts)
    return pd.DataFrame(results)
