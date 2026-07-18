from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import ROOT_DIR, load_params, setup_logger

logger = setup_logger(
    name="model-building",
    file_name=(ROOT_DIR / "logs" / "model_building.log")
)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file"""

    try:
        df = pd.read_csv(filepath_or_buffer=file_path)
        logger.debug("Data loaded from %s with shape %s", file_path, df.shape)

        return df

    except pd.errors.ParserError:
        logger.error("Failed to Parse the csv file", exc_info=True)

        raise

    except FileNotFoundError:
        logger.error("File not found", exc_info=True)

        raise

    except Exception:
        logger.error("Unexpected error occurred while loading the data", exc_info=True)

        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:

    try:
        logger.debug("Initializing Logistic Regression Model with Parameters")

        params = load_params(params_path=(ROOT_DIR / "params.yaml").as_posix())

        C:float = params["model-building"]["C"]
        solver = params["model-building"]["solver"]
        penalty = params["model-building"]["penalty"]
        random_state = params["model-building"]["random_state"]

        lr = LogisticRegression(
                C= C,
                solver=solver,
                penalty=penalty,
                random_state= random_state
        )

        logger.debug("Model Training started with %s data size", X_train.shape)

        lr.fit(X_train, y_train)

        logger.info("Model Training Completed.")

        return lr

    except ValueError:
        logger.error("ValueError during model training", exc_info=True)

        raise

    except Exception:
        logger.error("Error during model training", exc_info=True)

        raise


def save_model(model: LogisticRegression, file_path: Path) -> None:
    """Save the trained model to a file."""

    try:
        ARTIFACTS_DIR = ROOT_DIR / "artifacts"
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, filename=file_path)

        logger.debug("Model Saved to %s", file_path)

    except FileNotFoundError:
        logger.error("File path not found", exc_info=True)

        raise

    except Exception:
        logger.error("Error occurred while saving the model", exc_info=True)

        raise


def main():

    try:
        train_data = load_data(
            file_path=(ROOT_DIR / "data" / "processed" / "train_tfidf.csv")
        )

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        lr = train_model(X_train=X_train, y_train=y_train)

        save_model(model=lr, file_path=(ROOT_DIR / "artifacts" / "model.joblib"))

    except Exception:
        logger.error("Failed to complete the model building process", exc_info=True)

        raise


if __name__ == "__main__":
    main()
