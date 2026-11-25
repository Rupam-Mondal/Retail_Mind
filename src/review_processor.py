import json
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _project_root():
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=4)
def _load_text_model(model_dir):
    model_path = Path(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def _predict_text(text, model_dir):
    tokenizer, model, device = _load_text_model(str(model_dir))

    encoded = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_id].item())

    id2label = model.config.id2label
    predicted_label = id2label.get(
        predicted_id,
        id2label.get(str(predicted_id), f"LABEL_{predicted_id}")
    )

    return {
        "category": str(predicted_label).lower(),
        "label": predicted_id,
        "confidence": confidence
    }


def predict_review_sentiment(review_text, model_dir=None, model_path=None):
    if model_dir is None:
        if model_path and Path(model_path).is_dir():
            model_dir = model_path
        else:
            model_dir = (
                _project_root()
                / "models"
                / "nlp"
                / "retailmind_bert_sentiment"
            )

    result = _predict_text(review_text, model_dir)

    return {
        "sentiment": result["category"],
        "label": result["label"],
        "confidence": result["confidence"]
    }


def route_support_ticket(ticket_text, model_dir=None):
    if model_dir is None:
        model_dir = (
            _project_root()
            / "models"
            / "nlp"
            / "retailmind_bert_ticket_router"
        )

    result = _predict_text(ticket_text, model_dir)

    return {
        "category": result["category"],
        "label": result["label"],
        "confidence": result["confidence"]
    }
