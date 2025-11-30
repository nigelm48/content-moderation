from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

LABEL_MAP = {
    0: "hate_speech",
    1: "normal",
    2: "offensive"
}

tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
model.eval()

def hatexplain(texts):
    """
    Runs HateXplain on a list of texts and returns LABELS ONLY.
    """
    labels = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).item()
        labels.append(LABEL_MAP[preds])
    return labels

if __name__ == "__main__":
    sample_texts = [
        "I hate you",
        "Have a nice day!",
        "how are you?",
        "You're so dumb",
        "what's wrong with you? you idiot"
    ]
    predictions = hatexplain(sample_texts)
    for text, label in zip(sample_texts, predictions):
        print(f"Text: {text} | Predicted Label: {label}")
