import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import ROOT_DIR, load_params, setup_logger

logger = setup_logger(
    name="feature-engineering", file_name=(ROOT_DIR / "logs" / "feature_engineering.log")
)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""

    try:
        df = pd.read_csv(filepath_or_buffer=file_path)
        logger.debug("Data Loaded from %s", file_path)

        return df

    except pd.errors.ParserError:
        logger.error("Failed to parse the CSV file", exc_info=True)

        raise

    except Exception:
        logger.error("Unexpected error occurred while loading the data", exc_info=True)

        raise


def apply_tfidf(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int,
    ngram_range: tuple[int, int],
) -> tuple:
    """Apply TfIdf to the data and save the vectorizer"""

    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data["text"].fillna("").astype(str)
        y_train = train_data["target"].values

        X_test = test_data["text"].fillna("").astype(str)
        y_test = test_data["target"].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        train_df = pd.DataFrame(
            data=X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out()
        )
        train_df["label"] = y_train

        test_df = pd.DataFrame(
            data=X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out()
        )
        test_df["label"] = y_test

        logger.info("TFIDF applied and Data Transformed")

        # save vectorizer
        vec_path = ROOT_DIR / "artifacts"
        vec_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            value=vectorizer,
            filename= vec_path/"vectorizer.joblib",
            compress=True,
        )

        return train_df, test_df

    except Exception:
        logger.error("Error occured during TFIDF transformation", exc_info=True)

        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""

    try:
        DATA_DIR = ROOT_DIR / "data" / "processed"
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        df.to_csv(path_or_buf=(DATA_DIR / file_path).as_posix(), index=False)

        logger.debug("Data saved to %s", file_path)

    except Exception:
        logger.error("Unexpected error occurred while saving the data", exc_info=True)

        raise


def main():

    try:
        params = load_params(params_path=(ROOT_DIR / "params.yaml").as_posix())
        max_features = params["feature-engineering"]["max_features"]
        ngram_range = tuple(params["feature-engineering"]["ngram_range"])

        train_data = load_data(
            file_path=(ROOT_DIR / "data" / "interim" / "train_processed.csv").as_posix()
        )
        test_data = load_data(
            file_path=(ROOT_DIR / "data" / "interim" / "test_processed.csv").as_posix()
        )

        train_df, test_df = apply_tfidf(
            train_data=train_data,
            test_data=test_data,
            max_features=max_features,
            ngram_range=ngram_range,
        )

        save_data(df=train_df, file_path="train_tfidf.csv")
        save_data(df=test_df, file_path="test_tfidf.csv")

    except Exception:
        logger.error("Failed to complete the feature engineering process", exc_info=True)

        raise


if __name__ == "__main__":
    main()
