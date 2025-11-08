from fastapi import APIRouter, HTTPException
from schemas.request import AllocationRequest
from schemas.response import AllocationResponse
from services.ai_service import AIService, PowerPredictor, MethodRecommender  # <- import needed classes

router = APIRouter(prefix="/allocate", tags=["Allocation"])

# Initialize dependencies
power_predictor = PowerPredictor()
method_recommender = MethodRecommender()

# Pass them to AIService
ai_service = AIService(power_predictor, method_recommender)

@router.post("/", response_model=AllocationResponse)
async def allocate_energy(data: AllocationRequest):
    print("Received request:", data.dict())
    try:
        result = ai_service.optimize_allocation([
            {"sector": "Livelihoods", "amount": data.livelihoods},
            {"sector": "Industries", "amount": data.industries},
            {"sector": "Govt Projects", "amount": data.govt},
        ])
        print("Allocation result:", result)
        return AllocationResponse(**result)
    except Exception as e:
        print("Error in allocation:", e)
        raise HTTPException(status_code=500, detail=f"Allocation failed: {str(e)}")
