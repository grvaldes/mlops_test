import mlflow
import mlflow.sklearn

from material_mlops.training.train import DATA_PATH, run_training

EXPERIMENT_NAME = "concrete-strength-baseline"
MODEL_NAME = "concrete-strength-model"


def main() -> None:
    mlflow.set_tracking_uri("http://mlflow:5000")
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

        print(f"Run ID: {run.info.run_id}")
        print("Model evaluation:")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
