# backend/models/power_predictor.py
import joblib
import os

MODEL_SAVE_PATH = "backend/models/saved_regressor_model.pkl"

class PowerPredictor:
    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.model = None
        self.is_trained = False
        
        # Load existing model if available
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.is_trained = True
            print(f"Loaded trained power model from {model_path}")
        else:
            print("Model not found. Please train and save the model first.")

    def predict(self, input_list):
        """Input: [AT, V, AP, RH], Output: Predicted PE value"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Train it first.")
        return float(self.model.predict([input_list])[0])
