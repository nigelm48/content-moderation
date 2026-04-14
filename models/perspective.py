from googleapiclient import discovery
import pandas as pd
import time
from functools import lru_cache
from dotenv import load_dotenv
import os

# Perspective API Docs was used as a reference for this implementation: https://developers.perspectiveapi.com/s/docs-sample-requests?language=en_US
load_dotenv()
API_KEY = os.getenv("PERSPECTIVE_API_KEY")

client = discovery.build("commentanalyzer", "v1alpha1", developerKey=API_KEY,discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1", static_discovery=False)

@lru_cache(maxsize=None)
def _cached_perspective_single(text):

    analyze_request = {"comment": {"text": text}, "languages": ["en"], "requestedAttributes": {"TOXICITY": {}}}

    response = client.comments().analyze(body=analyze_request).execute()

    return (response.get("attributeScores", {}).get("TOXICITY", {}).get("summaryScore", {}).get("value", None))



def evaluate_perspective(texts):
    scores = []
    requests_this_minute = 0
    minute_start = time.time()

    for i, text in enumerate(texts):

        if time.time() - minute_start >= 60:
            requests_this_minute = 0
            minute_start = time.time()

        if requests_this_minute >= 55:
            sleep_time = 60 - (time.time() - minute_start)
            print(f"[INFO] Throttling: sleeping {sleep_time:.1f}s to avoid rate limits...")
            time.sleep(sleep_time)
            requests_this_minute = 0
            minute_start = time.time()

        try:
            score = _cached_perspective_single(str(text))

            if score is None:
                print(f"[Warning] No score returned for text {i}")
            scores.append(score)

        except Exception as e:
            print(f"[Error] Failed on text {i}: {e}")
            scores.append(None)

        requests_this_minute += 1

    return pd.DataFrame({"toxicity": scores})


if __name__ == "__main__":
    example = ["I hate you", "Have a nice day!", "I hate you"]
    print(evaluate_perspective(example))
