import pandas as pd

from utils import ROOT_DIR, clean_and_normalize, setup_logger

logger = setup_logger(
    name="preprocessing",
    file_name=(ROOT_DIR / "logs" / "preprocessing.log").as_posix()
)


def transform_series(series: pd.Series) -> pd.Series:
    """
    Internally uses Spacy NLP Pipeline to transforms
    the input series by converting it to lowercase,
    tokenizing, removing stopwords and punctuation,
    and lemmatization.

    """

    return clean_and_normalize(series)


def preprocess_df(
    df: pd.DataFrame,
    text_cloumn: pd.Series = "text",
    target_column: pd.Series = "target",
) -> pd.DataFrame:
    """
    Preprocesses the DataFrame by encoding the
    target column, removing duplicates,
    and transforming the text column.

    """

    try:
        logger.debug("Starting Preprocessing the Data")

        # Remove Duplicates
        df = df.drop_duplicates(keep="first")
        logger.debug("Removed Duplicated rows.")

        # Remove nan
        df = df.dropna()
        logger.debug("Removed nan values.")

        # Encoding
        df[target_column] = df[target_column].map({"spam": 1, "ham": 0})
        logger.debug("Encoding successfully done.")

        # Apply Transformation
        df[text_cloumn] = transform_series(df[text_cloumn])
        logger.info("Text Column Transformed.")

        return df

    except KeyError:
        logger.error("Column not found.", exc_info=True)

        raise

    except Exception:
        logger.error("Error during Text Normalization", exc_info=True)

        raise


def main(text_column="text", target_column="target"):

    try:
        # Fetch the raw Data
        train_data = pd.read_csv((ROOT_DIR / "data" / "raw" / "train.csv").as_posix())
        test_data = pd.read_csv((ROOT_DIR / "data" / "raw" / "test.csv").as_posix())

        logger.debug("Data Loaded properly")

        # Transform the Data
        train_processed_data: pd.DataFrame = preprocess_df(
            df=train_data, text_cloumn=text_column, target_column=target_column
        )

        test_processed_data: pd.DataFrame = preprocess_df(
            df=test_data, text_cloumn=text_column, target_column=target_column
        )

        # Store the data inside data/interim
        DATA_DIR = ROOT_DIR / "data" / "interim"
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        train_processed_data.to_csv(
            path_or_buf=(DATA_DIR / "train_processed.csv").as_posix(), index=False
        )
        test_processed_data.to_csv(
            path_or_buf=(DATA_DIR / "test_processed.csv").as_posix(), index=False
        )

        logger.info("Processed data saved to %s", DATA_DIR)

    except FileNotFoundError:
        logger.error("File not found", exc_info=True)

        raise

    except pd.errors.EmptyDataError:
        logger.error("No data found", exc_info=True)

        raise

    except Exception:
        logger.error("Failed to Preprocess the Data", exc_info=True)

        raise


if __name__ == "__main__":
    main()
