# backend/routers/analyze.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from services.ai_service import AIService, PowerPredictor, MethodRecommender
from typing import Any

router = APIRouter(prefix="/analyze", tags=["Analyze"])

# Initialize AI Service
power_predictor = PowerPredictor()
method_recommender = MethodRecommender()
ai_service = AIService(power_predictor, method_recommender)

# Request model for frontend JSON
class DatasetRequest(BaseModel):
    dataset: list[list[Any]]  # Accept strings, numbers, etc.
    task: str = "regression"  # default task

@router.post("/")
async def analyze_dataset(req: DatasetRequest):
    try:
        if not req.dataset or len(req.dataset) < 2:
            raise HTTPException(status_code=400, detail="Dataset is empty or invalid")

        headers = req.dataset[0]
        rows = req.dataset[1:]

        df = pd.DataFrame(rows, columns=headers)

        if req.task == "regression":
            # Use AIService regression
            result = ai_service.run_regression(df)
        else:
            # Placeholder for classification (optional)
            result = ai_service.run_classification_placeholder(df)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
