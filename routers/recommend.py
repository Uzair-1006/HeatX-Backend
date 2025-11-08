from fastapi import APIRouter, HTTPException
from schemas.request import PowerInput, MethodRecommendationRequest
from schemas.response import PredictionResponse, MethodRecommendationResponse
from services.ai_service import AIService, PowerPredictor, MethodRecommender

router = APIRouter()

# Initialize dependencies
power_predictor = PowerPredictor()
method_recommender = MethodRecommender()

# Pass them to AIService
ai_service = AIService(power_predictor, method_recommender)

# --- Recommendation route ---
@router.post("/recommend-method", response_model=MethodRecommendationResponse)
async def recommend_method(data: MethodRecommendationRequest):
    try:
        result = ai_service.recommend_method(
            temp=data.temperature,
            pressure=data.pressure,
            scalability=data.scalability,
            budget=data.budget
        )
        return MethodRecommendationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
