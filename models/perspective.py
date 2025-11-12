import requests
import pandas as pd

API_KEY = ""
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

if __name__ == "__main__":
    sample_texts = ["I hate you", "Have a nice day!"]
    scores = evaluate_perspective(sample_texts)
    print(scores)