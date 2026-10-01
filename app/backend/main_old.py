from contextlib import asynccontextmanager
from pathlib import Path
import joblib

import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel,Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BEST_MODEL_DIR = PROJECT_ROOT / "app" / "backend" / "XGBClassifier_binary.pkl"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"project root: {PROJECT_ROOT}")
    print(f"project root: {BEST_MODEL_DIR}")
    global model

    with open(BEST_MODEL_DIR, "rb") as file:
        model = joblib.load(file)

    yield

    model = None


app = FastAPI(
    title="Machine-Failure Prediction API",
    description="API for serving predictions from a saved PKL model",
    version="1.0.0",
    lifespan=lifespan,
)

class PredictionInput(BaseModel):
    air_temperature_k : float = Field(gt = 0)
    process_temperature_k : float = Field(gt=0)
    rotational_speed_rpm : float = Field(gt = 0)
    torque_nm : float = Field(gt=0)
    tool_wear_min : float = Field(gt=0)
    load_type : str | None = None

class PredictionOutput(BaseModel):
    predicted_value : int


@app.get("/")
def root():
    return {
        "message": "Machine-Failure Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check(request : Request):
    if model is not None:
        return {"status":"healthy",
                "model_loaded":request.app.state.model is not None}
    return {"status":"unhealthy",
            "model_loaded":request.app.state.model is None}

@app.post("/predict")
def predict(input_data : PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )  
    try:

        input_features = pd.DataFrame([{
                "Air temperature [K]" : input_data.air_temperature_k,
                "Process temperature [K]" : input_data.process_temperature_k,
                "Rotational speed [rpm]" : input_data.rotational_speed_rpm,
                "Torque [Nm]" : input_data.torque_nm,
                "Tool wear [min]" : input_data.tool_wear_min,
                "Type" : input_data.load_type
            }])

        print(f"input features:{input_features}")
        predicted_value = model.predict(input_features)[0]

        return {
            'prediction' : int(predicted_value)
        }
    
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc



