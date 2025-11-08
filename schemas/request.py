from pydantic import BaseModel

class PowerInput(BaseModel):
    AT: float  # Ambient Temperature
    V: float   # Vacuum
    AP: float  # Atmospheric Pressure
    RH: float  # Relative Humidity

class MethodRecommendationRequest(BaseModel):
    temp: float
    pressure: float
    scalability: str
    budget: str

class AllocationRequest(BaseModel):
    livelihoods: int
    industries: int
    govt: int
    
class AIAnalysisRequest(BaseModel):
    field1: str
    field2: int
