import pandas as pd

from material_mlops.models.baseline import evaluate_model, train_model


def make_training_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [5, 4, 3, 2, 1],
        }
    )

    y = pd.Series([2, 4, 6, 8, 10])

    return X, y


def test_train_model_returns_fitted_model():
    X, y = make_training_data()

    model = train_model(X, y)

    assert hasattr(model, "predict")


def test_model_can_make_predictions():
    X, y = make_training_data()

    model = train_model(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)


def test_evaluate_model_returns_expected_metrics():
    X, y = make_training_data()

    model = train_model(X, y)
    metrics = evaluate_model(model, X, y)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics


def test_metrics_are_finite():
    X, y = make_training_data()

    model = train_model(X, y)
    metrics = evaluate_model(model, X, y)

    assert all(pd.notna(value) for value in metrics.values())
