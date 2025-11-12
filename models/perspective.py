import requests
import pandas as pd

API_KEY = "AIzaSyBB7A7NthFxd3Q8czAzZNh6bUCD58bkJeo"
URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

def evaluate_perspective(texts):
    scores = []
    for text in texts:
        data = {
            'comment': {'text': text},
            'languages': ['en'],
            'requestedAttributes': {'TOXICITY': {}}
        }
        response = requests.post(f"{URL}?key={API_KEY}", json=data).json()
        score = response['attributeScores']['TOXICITY']['summaryScore']['value']
        scores.append(score)
    return pd.DataFrame({"toxicity": scores})
