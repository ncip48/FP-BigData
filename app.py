from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <-- Import this
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json

app = FastAPI(title="House Price Prediction API")

# Add CORS configuration
app.add_middleware(
    CORSMiddleware,
    # In development, you can use ["*"] to allow all domains.
    # In production, replace with your frontend URL, e.g., ["https://my-frontend.com"]
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods like POST, GET, OPTIONS
    allow_headers=["*"],  # Allows all headers
)

model = joblib.load('models/house_price_model.joblib')

class HouseFeatures(BaseModel):
    land_size_m2: float
    building_size_m2: float

@app.post("/predict")
def predict_price(features: HouseFeatures):
    input_data = pd.DataFrame([{
        'land_size_m2': features.land_size_m2,
        'building_size_m2': features.building_size_m2
    }])
    
    pred_log_price = model.predict(input_data)[0]
    pred_price_million = np.expm1(pred_log_price)
    actual_price = pred_price_million * 1_000_000
    
    return {
        "land_size_m2": features.land_size_m2,
        "building_size_m2": features.building_size_m2,
        "predicted_log_price": float(pred_log_price),
        "predicted_price_million": float(pred_price_million),
        "formatted_price": f"Rp {actual_price:,.0f}"
    }
    
@app.get("/heatmap")
def get_heatmap_data():
    corr_matrix = pd.read_json('corr_matrix.json')
    labels = corr_matrix.columns.tolist()
    heatmap_data = []

    for i, y_label in enumerate(labels):
        for j, x_label in enumerate(labels):
            heatmap_data.append({
                "x": x_label,
                "y": y_label,
                "v": round(float(corr_matrix.iloc[i, j]), 2)
            })

    return {
        "labels": labels,
        "data": heatmap_data
    }
    
@app.get("/scatter")
def get_scatter_data():
    # Load the saved data
    with open('scatter_data.json', 'r') as f:
        raw_data = json.load(f)
    
    actual_coords = []
    predicted_coords = []
    
    # Format into Chart.js {x, y} structure
    for row in raw_data:
        actual_coords.append({
            "x": row['building_size_m2'], 
            "y": row['actual']
        })
        predicted_coords.append({
            "x": row['building_size_m2'], 
            "y": row['predicted']
        })
        
    return {
        "actual": actual_coords,
        "predicted": predicted_coords
    }