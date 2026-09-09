from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from material_mlops.training.train import DATA_PATH, run_training


EXPERIMENT_NAME = "concrete-strength-baseline"
MODEL_NAME = "concrete-strength-model"
MODEL_PATH = Path("models/concrete_strength_model.joblib")


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        model, metrics = run_training(DATA_PATH)

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME,
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print(f"Run ID: {run.info.run_id}")
        print("Model evaluation:")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()