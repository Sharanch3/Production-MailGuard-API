import json
import os
from pathlib import Path

import dagshub
import joblib
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from utils import ROOT_DIR, setup_logger

load_dotenv()



logger = setup_logger(
    name="model-evaluation", file_name=(ROOT_DIR / "logs" / "model-evaluation.log")
)


def load_model(file_path: str):
    """Load the trained model from a file."""

    try:
        model = joblib.load(filename=file_path)

        logger.debug("Model Loaded form from path %s", file_path)

        return model

    except FileNotFoundError:
        logger.error("File not found", exc_info=True)

        raise

    except Exception:
        logger.error("Unexpected error occurred while loading the model", exc_info=True)

        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""

    try:
        df = pd.read_csv(filepath_or_buffer=file_path)
        logger.debug("Loaded data from %s", file_path)

        return df

    except pd.errors.ParserError:
        logger.error("Failed to parse the CSV file", exc_info=True)

        raise

    except Exception:
        logger.error("Unexpected error occurred while loading the data", exc_info=True)

        raise


def evaluate_model(
    model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
    """Evaluate the model and return the evaluation metrics."""

    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        }

        logger.debug("Model evaluation metrics calculated")

        return metrics_dict

    except Exception:
        logger.error("Error during model evaluation", exc_info=True)

        raise


def save_metrics(metrics: dict, file_path: Path) -> None:
    """Save the evaluation metrics to a JSON file."""

    try:
        REPORT_DIR = ROOT_DIR / "reports"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        with open(file=file_path, mode="w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("Metrics saved to %s", file_path)

    except Exception:
        logger.error("Error occurred while saving the metrics", exc_info=True)

        raise


def register_model(model: LogisticRegression, metrics: dict):
    """Register the model to the MLflow Model Registry."""

    try:
        # Read the token form env
        token = os.environ["DAGSHUB_PAT"]

        # Token based auth
        dagshub.auth.add_app_token(token)

        # Connect to dagshub
        dagshub.init(
            repo_owner="Sharanch3", repo_name="Production-MailGuard-API", mlflow=True
        )

        mlflow.set_experiment(experiment_name="dvc-pipeline")

        with mlflow.start_run(run_name="best-model"):
            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                name="Logistic Regression",
                registered_model_name="MailGuard-API",
            )

            # Log parameters
            mlflow.log_params(
                {
                    "C": model.C,
                    "solver": model.solver,
                    "penalty": model.penalty,
                    "random_state": model.random_state,
                }
            )

            logger.info("Model registered Successfully!")

    except Exception:
        logger.error("Failed to register model", exc_info=True)

        raise


def main():

    try:
        lr = load_model((ROOT_DIR / "artifacts" / "model.joblib").as_posix())

        test_data = load_data(
            file_path=(ROOT_DIR / "data" / "processed" / "test_tfidf.csv").as_posix()
        )

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(model=lr, X_test=X_test, y_test=y_test)

        save_metrics(metrics=metrics, file_path=(ROOT_DIR / "reports" / "metrics.json"))

        register_model(model=lr, metrics=metrics)

    except Exception:
        logger.error("Failed to complete the model evaluation process", exc_info=True)

        raise


if __name__ == "__main__":
    main()
