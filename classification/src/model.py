import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def train_baseline(df: pd.DataFrame):
    # Separate features and label
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratified split to preserve class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Baseline Logistic Regression ===")
    print(classification_report(y_test, y_pred))

    return model


if __name__ == "__main__":
    df = pd.read_csv("classification/data/creditcard.csv")
    train_baseline(df)
