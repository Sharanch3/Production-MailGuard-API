import pandas as pd
from sklearn.model_selection import train_test_split

from utils import ROOT_DIR, load_params, setup_logger

logger = setup_logger(
    name="ingestion",
    file_name=(ROOT_DIR / "logs" / "ingestion.log").as_posix()
)


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a URL"""

    try:
        df = pd.read_csv(
            filepath_or_buffer= data_url,
            sep= ","

        ).rename(columns= {
            "text": "text",
            "label": "target"
        })

        logger.debug("Data Loaded from %s", data_url)

        return df

    except pd.errors.ParserError:
        logger.error("Failed to parse the File.", exc_info=True)

        raise

    except Exception:
        logger.error("Unexpected error ocurred during Loading the data", exc_info=True)

        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save the raw train and test dataset"""

    try:
        DATA_DIR = ROOT_DIR / "data" / "raw"

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        train_data.to_csv((DATA_DIR / "train.csv").as_posix(), index=False)
        test_data.to_csv((DATA_DIR / "test.csv").as_posix(), index=False)

        logger.info("Train and Test data saved to %s", DATA_DIR.as_posix())

    except Exception:
        logger.error("Unexpected error occur during saving the data", exc_info=True)

        raise


def main():

    try:
        params = load_params(params_path=(ROOT_DIR / "params.yaml").as_posix())

        test_size = params["data-ingestion"]["test_size"]
        url = "https://raw.githubusercontent.com/Sharanch3/Datasets/refs/heads/main/emails.csv"

        df = load_data(data_url=url)

        train_data, tests_data = train_test_split(
            df, test_size=test_size, random_state=42
        )

        save_data(train_data=train_data, test_data=tests_data)

    except Exception as e:
        logger.error("Failed to complete the data ingestion process", exc_info=True)

        print(f"ERROR: {str(e)}")


if __name__ == "__main__":
    main()
