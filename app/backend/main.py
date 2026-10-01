"""
Machine-Failure Prediction API — binary + multi-class serving.

Suggested location: app_v2/backend/main.py
Run:  uvicorn app_v2.backend.main:app --reload --port 8000   (from project root)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BINARY_MODEL_PATH = Path(
    os.getenv("BINARY_MODEL_PATH", PROJECT_ROOT / "models" / "binary" / "XGBoost_pipeline.pkl")
)
MULTI_MODEL_PATH = Path(
    os.getenv("MULTI_MODEL_PATH", PROJECT_ROOT / "models" / "multi" / "LightGBM_pipeline.pkl")
)
# Tune this from your PR curve instead of assuming 0.5 on an imbalanced dataset.
BINARY_THRESHOLD = float(os.getenv("BINARY_THRESHOLD", "0.5"))
# Multi-class: flag failure when P(any failure) = 1 - P(No Failure) >= this, then
# report the most likely *failure* class. Plain argmax lets "No Failure" win with
# e.g. 40% while failure modes split the remaining 60%.
MULTI_FAILURE_THRESHOLD = float(os.getenv("MULTI_FAILURE_THRESHOLD", "0.5"))

# Column order used at training time (matches data/split/*/X_*.csv)
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

BINARY_LABELS = {"0": "No Failure", "1": "Machine Failure"}

# Same encoding as src/models/train.py::encode_multiclass_target
MULTI_LABELS = {
    "0": {"code": "NONE", "name": "No Failure"},
    "1": {"code": "TWF", "name": "Tool Wear Failure"},
    "2": {"code": "HDF", "name": "Heat Dissipation Failure"},
    "3": {"code": "PWF", "name": "Power Failure"},
    "4": {"code": "OSF", "name": "Overstrain Failure"},
    "5": {"code": "RNF", "name": "Random Failure"},
}
NO_FAILURE_CLASS = "0"


class Mode(str, Enum):
    binary = "binary"
    multiclass = "multiclass"


# --------------------------------------------------------------------------- #
# Lifespan: load both models once
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {}
    app.state.load_errors = {}
    for mode, path in ((Mode.binary, BINARY_MODEL_PATH), (Mode.multiclass, MULTI_MODEL_PATH)):
        try:
            app.state.models[mode] = joblib.load(path)
            print(f"[startup] loaded {mode.value} model: {path}")
        except Exception as exc:  # keep the API up even if one model is missing
            app.state.load_errors[mode] = f"{type(exc).__name__}: {exc}"
            print(f"[startup] FAILED to load {mode.value} model from {path}: {exc}")
    yield
    app.state.models.clear()


app = FastAPI(
    title="Machine-Failure Prediction API",
    description="Binary (failure / no failure) and multi-class (failure mode) predictions.",
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class PredictionInput(BaseModel):
    load_type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(gt=0)
    process_temperature_k: float = Field(gt=0)
    rotational_speed_rpm: float = Field(gt=0)
    torque_nm: float = Field(ge=0)
    tool_wear_min: float = Field(ge=0)  # a brand-new tool has 0 min wear

    model_config = {
        "json_schema_extra": {
            "example": {
                "load_type": "L",
                "air_temperature_k": 298.1,
                "process_temperature_k": 308.6,
                "rotational_speed_rpm": 1551,
                "torque_nm": 42.8,
                "tool_wear_min": 0,
            }
        }
    }


class BatchInput(BaseModel):
    records: list[PredictionInput] = Field(min_length=1, max_length=5000)


class ClassProbability(BaseModel):
    class_id: str
    code: str
    name: str
    probability: float


class PredictionOutput(BaseModel):
    """One envelope for both modes so the frontend can render generically."""
    mode: Mode
    model_name: str
    class_id: str                   # predicted class as string ("0".."5")
    code: str                       # short label (e.g. "HDF")
    name: str                       # human label
    is_failure: bool                # common flag across both modes
    failure_probability: float      # P(any failure) — comparable across modes
    confidence: float               # probability of the predicted class
    probabilities: list[ClassProbability]  # sorted desc
    threshold: float | None = None  # decision threshold on failure_probability
    warnings: list[str] = []


class CombinedOutput(BaseModel):
    binary: PredictionOutput
    multiclass: PredictionOutput
    agreement: bool
    summary: str


class BatchOutput(BaseModel):
    mode: Mode
    count: int
    predictions: list[PredictionOutput]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_model(request: Request, mode: Mode):
    model = request.app.state.models.get(mode)
    if model is None:
        err = request.app.state.load_errors.get(mode, "not loaded")
        raise HTTPException(status_code=503, detail=f"{mode.value} model unavailable: {err}")
    return model


def to_frame(records: list[PredictionInput]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "Type": r.load_type,
                "Air temperature [K]": r.air_temperature_k,
                "Process temperature [K]": r.process_temperature_k,
                "Rotational speed [rpm]": r.rotational_speed_rpm,
                "Torque [Nm]": r.torque_nm,
                "Tool wear [min]": r.tool_wear_min,
            }
            for r in records
        ]
    )
    return df[FEATURE_COLUMNS]


def model_name(model) -> str:
    est = getattr(model, "named_steps", {}).get("classifier", model)
    return type(est).__name__


def input_warnings(r: PredictionInput) -> list[str]:
    """Cheap sanity checks against the AI4I 2020 training range."""
    w = []
    if r.process_temperature_k < r.air_temperature_k:
        w.append("Process temperature is below air temperature (unusual for this dataset).")
    if not 1150 <= r.rotational_speed_rpm <= 2900:
        w.append("Rotational speed is outside the training range (~1168–2886 rpm).")
    if r.torque_nm > 80:
        w.append("Torque is outside the training range (~3.8–76.6 Nm).")
    if r.tool_wear_min > 260:
        w.append("Tool wear is outside the training range (0–253 min).")
    return w


def build_binary(proba_row: np.ndarray, classes, name: str, rec: PredictionInput) -> PredictionOutput:
    cls = [str(c) for c in classes]
    p_fail = float(proba_row[cls.index("1")])
    pred = "1" if p_fail >= BINARY_THRESHOLD else "0"
    probs = sorted(
        (
            ClassProbability(
                class_id=c,
                code="FAIL" if c == "1" else "OK",
                name=BINARY_LABELS[c],
                probability=float(p),
            )
            for c, p in zip(cls, proba_row)
        ),
        key=lambda x: x.probability,
        reverse=True,
    )
    return PredictionOutput(
        mode=Mode.binary,
        model_name=name,
        class_id=pred,
        code="FAIL" if pred == "1" else "OK",
        name=BINARY_LABELS[pred],
        is_failure=pred == "1",
        failure_probability=p_fail,
        confidence=p_fail if pred == "1" else 1 - p_fail,
        probabilities=probs,
        threshold=BINARY_THRESHOLD,
        warnings=input_warnings(rec),
    )


def build_multi(proba_row: np.ndarray, classes, name: str, rec: PredictionInput) -> PredictionOutput:
    cls = [str(c) for c in classes]  # handles int or string class labels
    probs = sorted(
        (
            ClassProbability(
                class_id=c,
                code=MULTI_LABELS.get(c, {"code": c})["code"],
                name=MULTI_LABELS.get(c, {"name": f"Class {c}"})["name"],
                probability=float(p),
            )
            for c, p in zip(cls, proba_row)
        ),
        key=lambda x: x.probability,
        reverse=True,
    )
    p_no_fail = float(proba_row[cls.index(NO_FAILURE_CLASS)]) if NO_FAILURE_CLASS in cls else 0.0
    p_fail = 1.0 - p_no_fail
    failure_probs = [p for p in probs if p.class_id != NO_FAILURE_CLASS]

    if p_fail >= MULTI_FAILURE_THRESHOLD and failure_probs:
        top = failure_probs[0]  # most likely failure mode
    else:
        top = next(p for p in probs if p.class_id == NO_FAILURE_CLASS)

    warnings = input_warnings(rec)
    if top.class_id != probs[0].class_id:
        warnings.append(
            f"Argmax class is '{probs[0].code}', but P(any failure) = {p_fail:.1%} "
            f">= {MULTI_FAILURE_THRESHOLD:.0%}, so the top failure mode is reported."
        )
    elif abs(p_fail - MULTI_FAILURE_THRESHOLD) < 0.10:
        warnings.append(
            f"P(any failure) = {p_fail:.1%} is close to the {MULTI_FAILURE_THRESHOLD:.0%} "
            f"threshold (top failure mode: {failure_probs[0].code}) — treat as uncertain."
        )

    return PredictionOutput(
        mode=Mode.multiclass,
        model_name=name,
        class_id=top.class_id,
        code=top.code,
        name=top.name,
        is_failure=top.class_id != NO_FAILURE_CLASS,
        failure_probability=p_fail,
        confidence=top.probability,
        probabilities=probs,
        threshold=MULTI_FAILURE_THRESHOLD,
        warnings=warnings,
    )


BUILDERS = {Mode.binary: build_binary, Mode.multiclass: build_multi}


def run(request: Request, mode: Mode, records: list[PredictionInput]) -> list[PredictionOutput]:
    model = get_model(request, mode)
    try:
        proba = model.predict_proba(to_frame(records))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{mode.value} prediction failed: {exc}") from exc
    name = model_name(model)
    return [BUILDERS[mode](row, model.classes_, name, rec) for row, rec in zip(proba, records)]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    return {"message": "Machine-Failure Prediction API is running", "docs": "/docs"}


@app.get("/health")
def health(request: Request):
    loaded = {m.value: m in request.app.state.models for m in Mode}
    return {
        "status": "healthy" if all(loaded.values()) else "degraded",
        "models_loaded": loaded,
        "errors": {m.value: e for m, e in request.app.state.load_errors.items()},
    }


@app.get("/metadata")
def metadata(request: Request):
    """Frontend reads labels/threshold from here instead of hard-coding them."""
    out: dict[str, Any] = {"features": FEATURE_COLUMNS, "modes": {}}
    for mode, model in request.app.state.models.items():
        out["modes"][mode.value] = {
            "model_name": model_name(model),
            "classes": [str(c) for c in model.classes_],
            "labels": BINARY_LABELS if mode == Mode.binary else MULTI_LABELS,
            "threshold": BINARY_THRESHOLD if mode == Mode.binary else MULTI_FAILURE_THRESHOLD,
        }
    return out


# Declared before /predict/{mode} so "combined" isn't parsed as a Mode.
@app.post("/predict/combined", response_model=CombinedOutput)
def predict_combined(payload: PredictionInput, request: Request):
    b = run(request, Mode.binary, [payload])[0]
    m = run(request, Mode.multiclass, [payload])[0]
    agreement = b.is_failure == m.is_failure
    if b.is_failure and m.is_failure:
        summary = f"Failure risk detected — most likely mode: {m.code} ({m.name})."
    elif not b.is_failure and not m.is_failure:
        summary = "No failure expected by either model."
    elif b.is_failure:
        summary = (
            f"Binary model flags failure, but the multi-class model's top class is 'No Failure'. "
            f"Most likely failure mode if it happens: {next(p for p in m.probabilities if p.class_id != NO_FAILURE_CLASS).code}."
        )
    else:
        summary = f"Multi-class model predicts {m.code}, but binary model is below threshold — inspect manually."
    return CombinedOutput(binary=b, multiclass=m, agreement=agreement, summary=summary)


@app.post("/predict/{mode}", response_model=PredictionOutput)
def predict(mode: Mode, payload: PredictionInput, request: Request):
    return run(request, mode, [payload])[0]


@app.post("/predict/{mode}/batch", response_model=BatchOutput)
def predict_batch(mode: Mode, payload: BatchInput, request: Request):
    preds = run(request, mode, payload.records)
    return BatchOutput(mode=mode, count=len(preds), predictions=preds)