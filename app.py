from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
import numpy as np
import xgboost as xgb

app = FastAPI()

# -----------------------------
# Load XGBoost model
# -----------------------------
# Assumes:
#   app.py
#   model/model.json
model = xgb.Booster()
model.load_model("model/model.json")

# -----------------------------
# Serve frontend
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "frontend" / "index.html"

@app.get("/", response_class=HTMLResponse)
def read_root():
    try:
        content = HTML_FILE.read_text(encoding="utf-8")
        return HTMLResponse(content=content)
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Error loading HTML</h1><p>{str(e)}</p>",
            status_code=500,
        )

# -----------------------------
# Prediction endpoint
# -----------------------------
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
        # These **must** match model.json:
        # "feature_names": ["price_lag_1", ..., "price_lag_6"]
        feature_names = [
            "price_lag_1",
            "price_lag_2",
            "price_lag_3",
            "price_lag_4",
            "price_lag_5",
            "price_lag_6",
        ]

        # 1 row, 6 columns
        features = np.array([[lag1, lag2, lag3, lag4, lag5, lag6]], dtype=float)

        dmatrix = xgb.DMatrix(features, feature_names=feature_names)

        pred = model.predict(dmatrix)[0]
        return {"prediction": float(pred)}

    except Exception as e:
        # Show the underlying error in the response while debugging
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
