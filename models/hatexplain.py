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
    labels = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).item()
        labels.append(LABEL_MAP[preds])
    return labels
