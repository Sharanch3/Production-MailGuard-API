import os
import re
from pathlib import Path

import dagshub
import joblib
import mlflow
import spacy
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

load_dotenv()


def load_model_from_mr(model_name: str, model_version: int) -> LogisticRegression:
    """Load model from Dagshub Model Registry"""

    token = os.environ["DAGSHUB_PAT"]

    dagshub.auth.add_app_token(token)

    dagshub.init(
        repo_owner="Sharanch3", repo_name="Production-MailGuard-API", mlflow=True
    )

    model_uri = f"models:/{model_name}/{model_version}"

    return mlflow.sklearn.load_model(model_uri=model_uri)


def load_vectorizer() -> TfidfVectorizer:

    return joblib.load(
        filename=(Path(__file__).parent.parent / "artifacts" / "vectorizer.joblib")
    )


def load_nlp():
    """Load spaCy model."""

    return spacy.load(name="en_core_web_sm", disable=["parser", "ner"])


def preprocessor(text: str, nlp: spacy.language.Language) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)

    return " ".join(
        token.lemma_ for token in doc if not token.is_stop and not token.is_punct
    )
