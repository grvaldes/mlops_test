from pathlib import Path

from sklearn.model_selection import train_test_split

from material_mlops.data.ingestion import load_raw_data
from material_mlops.data.processing import process_data, split_features_target
from material_mlops.data.validation import validate_dataset
from material_mlops.models.baseline import evaluate_model, train_model


DATA_PATH = Path("data/raw/concrete.xls")


def main() -> None:
    data = load_raw_data(DATA_PATH)

    validate_dataset(data)

    processed = process_data(data)

    X, y = split_features_target(processed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = train_model(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("Model evaluation:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()