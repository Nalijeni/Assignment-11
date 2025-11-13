from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import numpy as np
from xgboost import XGBRegressor

app = FastAPI(title="Car Sales XGBoost API")

# CORS (allow browser requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # relax for assignment demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load feature order
with open("model/features.json") as f:
    FEATURES = json.load(f)["features"]

# Load model
model = XGBRegressor()
model.load_model("model/model.json")

class PredictRequest(BaseModel):
    features: List[float]  # validate count below

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")

@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != 6:
        raise HTTPException(status_code=400, detail="Provide exactly 6 values in lag_1..lag_6 order.")
    x = np.array(req.features, dtype=float).reshape(1, -1)
    pred = float(model.predict(x)[0])
    return {"prediction": pred}