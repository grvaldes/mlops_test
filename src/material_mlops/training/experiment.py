import mlflow

from material_mlops.training.train import DATA_PATH, run_training


def main() -> None:
    mlflow.set_experiment("concrete-strength-baseline")

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        metrics = run_training(DATA_PATH)

        mlflow.log_metrics(metrics)

        print("Model evaluation:")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()