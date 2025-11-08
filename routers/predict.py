# backend/routes/predict.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from services.quick_predict import predict_power  # <- the function we just wrote

router = APIRouter()

# Define input model
class PowerInput(BaseModel):
    AT: float
    V: float
    AP: float
    RH: float

@router.post("/predict")
async def predict_endpoint(data: PowerInput):
    """
    Input: JSON { "AT": float, "V": float, "AP": float, "RH": float }
    Output: JSON { "PE": float, "efficiency": float, "mtoe": float, "twh": float }
    """
    try:
        prediction = predict_power(data.dict())
        return {
            "PE": prediction.get("MW", 0),
            "efficiency": prediction.get("efficiency", 0),
            "mtoe": prediction.get("mtoe", 0),
            "twh": prediction.get("twh", 0)
        }
    except Exception as e:
        return {"PE": 0, "error": str(e)}
