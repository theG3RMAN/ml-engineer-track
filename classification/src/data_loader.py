import pandas as pd
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(file_path)


def validate_schema(df: pd.DataFrame) -> None:
    if "Class" not in df.columns:
        raise ValueError("Missing target column: Class")


def inspect_data(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nClass distribution:\n", df["Class"].value_counts())
    print("\nTotal missing values:", df.isnull().sum().sum())


if __name__ == "__main__":
    df = load_data("classification/data/creditcard.csv")
    validate_schema(df)
    inspect_data(df)
