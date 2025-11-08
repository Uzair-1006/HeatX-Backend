class MethodRecommender:
    def __init__(self):
        # Possible methods
        self.methods = ["Kalina Cycle", "Steam Cycle", "Organic Rankine Cycle (ORC)"]

    def recommend(self, temp: float, pressure: float, scalability: str, budget: str):
        """
        Basic rule-based method recommendation.
        Can be upgraded to ML later.
        """

        if temp > 500 and budget == "high":
            return {"method": "Kalina Cycle", "confidence": 0.9}
        elif temp > 400 and scalability == "large":
            return {"method": "Steam Cycle", "confidence": 0.85}
        else:
            return {"method": "Organic Rankine Cycle (ORC)", "confidence": 0.8}
