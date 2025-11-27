from openai import OpenAI

def evaluate_openai_moderation(texts):
    client = OpenAI()

    scores = []
    for t in texts:
        try:
            response = client.moderations.create(
                model="omni-moderation-latest",
                input=t
            )
            toxicity = 1.0 if response.results[0].categories['hate'] else 0.0
        except Exception:
            toxicity = 0.0
        scores.append(toxicity)

    return scores

if __name__ == "__main__":
    sample_texts = [
        "I hate you",
        "You're so dumb",
        "Have a nice day!"
    ]
    scores = evaluate_openai_moderation(sample_texts)
    for text, score in zip(sample_texts, scores):
        print(f"Text: {text} | Toxicity Score: {score}")