from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from xgboost import XGBRegressor




# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), "model", "model.json")
model = XGBRegressor()
model.load_model(model_path)

# Initialize FastAPI app
app = FastAPI()

# Allow frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body schema
class SalesInput(BaseModel):
    sales: list[float]  # Expect exactly 6 values


# Mount the 'frontend' directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve index.html at root
@app.get("/")
def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

# Define the prediction endpoint
@app.post("/predict")
async def predict(input: SalesInput):
    if len(input.sales) != 6:
        return {"error": "Exactly 6 months of sales data required."}
    
    data = np.array(input.sales).reshape(1, -1)
    prediction = model.predict(data)[0]
    return {"prediction": float(prediction)}
