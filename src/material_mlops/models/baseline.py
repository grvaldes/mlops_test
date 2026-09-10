from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_model(
    X_train: Any,
    y_train: Any,
) -> RandomForestRegressor:
    """Train the baseline regression model."""
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: Any,
    y_test: Any,
) -> dict[str, float]:
    """Evaluate the model on the test set."""
    predictions = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
    }

    return metrics
