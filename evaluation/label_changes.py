import pandas as pd

TOXIC_LABELS = {"offensive", "hate_speech"}
NORMAL_LABEL = "normal"

def evaluate_label_changes(clean_labels, perturbed_labels):

    assert len(clean_labels) == len(perturbed_labels)

    normal_to_toxic = 0
    toxic_to_normal = 0
    total_normal = 0
    total_toxic = 0

    for c, p in zip(clean_labels, perturbed_labels):

        if c == NORMAL_LABEL:
            total_normal += 1
            if p in TOXIC_LABELS:
                normal_to_toxic += 1

        if c in TOXIC_LABELS:
            total_toxic += 1
            if p == NORMAL_LABEL:
                toxic_to_normal += 1

    return pd.DataFrame([{
        "normal_to_toxic_rate": normal_to_toxic / total_normal if total_normal else 0,
        "toxic_to_normal_rate": toxic_to_normal / total_toxic if total_toxic else 0,
        "normal_to_toxic_count": normal_to_toxic,
        "toxic_to_normal_count": toxic_to_normal,
        "total_normal": total_normal,
        "total_toxic": total_toxic
    }])
