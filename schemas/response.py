from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    prediction: float
    metrics: Dict[str, float]

class MethodRecommendationResponse(BaseModel):
    method: str
    confidence: float

class AllocationResponse(BaseModel):
    allocations: Dict[str, int]
