import html
import re
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def tokenize_review(text):
    text = html.unescape(str(text)).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z']", " ", text)
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return [token for token in tokens if len(token) > 2 and token not in STOP_WORDS]


def clean_review_text(text):
    return " ".join(tokenize_review(text))


def predict_review_sentiment(review_text, model_path=None):
    project_root = Path(__file__).resolve().parents[1]
    if model_path is None:
        model_path = project_root / "models" / "nlp" / "retailmind_sentiment_pipeline.joblib"
    pipeline = joblib.load(model_path)
    prediction = int(pipeline.predict([review_text])[0])
    probabilities = pipeline.predict_proba([review_text])[0]
    return {
        "sentiment": "positive" if prediction == 1 else "negative",
        "label": prediction,
        "confidence": float(max(probabilities))
    }
