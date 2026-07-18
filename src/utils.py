import logging
from pathlib import Path

import pandas as pd
import spacy
import yaml

ROOT_DIR = Path(__file__).parent.parent.resolve()


# LOGGER-HELPER
def setup_logger(name, file_name, level=logging.DEBUG) -> logging.Logger:

    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(filename=file_name)
    file_handler.setLevel(level=level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(hdlr=file_handler)
    logger.addHandler(hdlr=console_handler)

    return logger


# NLP PIPELINE HELPER
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lower_replace(series: pd.Series) -> pd.Series:
    output = (
        series.str.lower()  # Convert to lowercase
        .str.replace(r"http\S+|www\S+|https\S+", "", regex=True)  # Remove URLs
        .str.replace(r"\S+@\S+", "", regex=True)  # Remove email addresses
        .str.replace(r"<.*?>", "", regex=True)  # Remove HTML tags
        .str.replace(r"\[.*?\]", "", regex=True)  # Remove text inside []
        .str.replace(r"[^a-zA-Z\s]", "", regex=True)  # Remove special chars & digits
        .str.replace(r"\s+", " ", regex=True)  # Collapse multiple spaces
        .str.strip()  # Remove leading/trailing spaces
    )

    return output


def clean_and_normalize(series: pd.Series):
    cleaned = lower_replace(series)

    return pd.Series(
        [
            " ".join(
                token.lemma_ for token in doc if not token.is_stop and not token.is_punct
            )
            for doc in nlp.pipe(cleaned, batch_size=1000)
        ],
        index=series.index,
    )


# LOAD PARAMS.YAML FILE
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""

    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)

        return params

    except Exception:
        raise
