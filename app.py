from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import xgboost as xgb
import numpy as np
from pathlib import Path
import pandas as pd

app = FastAPI()

# -----------------------------
# Load XGBoost model
# -----------------------------
model = xgb.Booster()
model.load_model("model/model.json")   # Make sure this path is correct!

# -----------------------------
# Serve frontend
# -----------------------------
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
            status_code=500
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
    lag6: float = Form(...)
):
    # Build DataFrame with training column names
    data = pd.DataFrame([{
        "Sales_Lag_1_Month": lag1,
        "Sales_Lag_2_Month": lag2,
        "Sales_Lag_3_Month": lag3,
        "Sales_Lag_4_Month": lag4,
        "Sales_Lag_5_Month": lag5,
        "Sales_Lag_6_Month": lag6
    }])

    # Convert to DMatrix
    dmatrix = xgb.DMatrix(data)

    # Predict
    prediction = model.predict(dmatrix)[0]

    return {"prediction": float(prediction)}
