from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
import xgboost as xgb
import numpy as np
from pathlib import Path
import pandas as pd

app = FastAPI()

# Load model (as before)
model = xgb.Booster()
model.load_model("model/model.json")

BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "frontend" / "index.html"

@app.get("/", response_class=HTMLResponse)
def read_root():
    try:
        content = HTML_FILE.read_text()
        return HTMLResponse(content=content)
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Error loading HTML</h1><p>{str(e)}</p>",
            status_code=500,
        )

@app.post("/predict")
def predict(
    lag1: float = Form(...),
    lag2: float = Form(...),
    lag3: float = Form(...),
    lag4: float = Form(...),
    lag5: float = Form(...),
    lag6: float = Form(...),
):
    try:
        # 1) Make sure everything is numeric
        features = np.array([[lag1, lag2, lag3, lag4, lag5, lag6]], dtype=float)

        # 2) Use numpy -> DMatrix explicitly (avoids some pandas/xgboost quirks)
        feature_names = [
            "Sales_Lag_1_Month",
            "Sales_Lag_2_Month",
            "Sales_Lag_3_Month",
            "Sales_Lag_4_Month",
            "Sales_Lag_5_Month",
            "Sales_Lag_6_Month",
        ]
        dmatrix = xgb.DMatrix(features, feature_names=feature_names)

        # 3) Predict
        pred = model.predict(dmatrix)[0]
        return {"prediction": float(pred)}
    except Exception as e:
        # This will still return 500, but with a helpful message
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
