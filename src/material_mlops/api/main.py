import os

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URI = "models:/concrete-strength-model/1"

model = mlflow.sklearn.load_model(MODEL_URI)


app = FastAPI(
    title="Concrete Strength Prediction API",
    version="0.1.0",
)

class ConcreteInput(BaseModel):
    cement: float
    blast_furnace_slag: float
    fly_ash: float
    water: float
    superplasticizer: float
    coarse_aggregate: float
    fine_aggregate: float
    age: float


class PredictionResponse(BaseModel):
    compressive_strength: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: ConcreteInput) -> PredictionResponse:
    features = pd.DataFrame(
        [
            {
                "cement": data.cement,
                "blast_furnace_slag": data.blast_furnace_slag,
                "fly_ash": data.fly_ash,
                "water": data.water,
                "superplasticizer": data.superplasticizer,
                "coarse_aggregate": data.coarse_aggregate,
                "fine_aggregate": data.fine_aggregate,
                "age": data.age,
            }
        ]
    )

    prediction = model.predict(features)[0]

    return PredictionResponse(
        compressive_strength=float(prediction)
    )