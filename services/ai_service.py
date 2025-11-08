# backend/services/ai_service.py
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from models.power_predictor import PowerPredictor
from models.method_recommender import MethodRecommender
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score
import math
import logging
import os
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIService:
    def __init__(self, power_predictor: PowerPredictor, method_recommender: MethodRecommender):
        self.power_predictor = power_predictor
        self.method_recommender = method_recommender

    # ---------------- Power Prediction ----------------
    def run_power_prediction(self, input_data: Dict[str, Any]):
        """
        Run power prediction using the trained regressor model.
        Expects keys: AT, V, AP, RH
        Returns dict with MW, efficiency, mtoe, twh and metrics
        """
        try:
            # ✅ Enforce input order [AT, V, AP, RH]
            input_list = [
                float(input_data.get("AT", 0)),
                float(input_data.get("V", 0)),
                float(input_data.get("AP", 0)),
                float(input_data.get("RH", 0)),
            ]
            logging.info(f"Running power prediction with ordered input: {input_list}")

            # ✅ Auto-train if model is not trained
            if not self.power_predictor.is_trained:
                dataset_path = "data/Folds5x2_pp.csv"
                if os.path.exists(dataset_path):
                    logging.info("Model not trained yet. Auto-training with dataset...")
                    df = pd.read_csv(dataset_path)
                    self.power_predictor.train(df)
                else:
                    logging.error("Dataset not found. Cannot train the model.")
                    return {"MW": 0, "efficiency": 0, "mtoe": 0, "twh": 0}, None

            # ✅ Perform prediction
            prediction = self.power_predictor.predict(input_list)

            # If predictor already returns dict
            if isinstance(prediction, dict):
                logging.info(f"Prediction result (dict): {prediction}")
                return prediction, getattr(self.power_predictor, "metrics", None)

            # If numeric value (MW) → convert to metrics
            if isinstance(prediction, (int, float, np.float64, np.float32)):
                power_mw = float(prediction)
                logging.info(f"Raw model prediction (MW): {power_mw}")
                converted_prediction = self.convert_power_to_energy_metrics(power_mw, input_list)
                logging.info(f"Converted prediction result: {converted_prediction}")
                return converted_prediction, getattr(self.power_predictor, "metrics", None)

            # Unexpected type fallback
            logging.error(f"Unexpected prediction type: {type(prediction)}")
            return {"MW": 0, "efficiency": 0, "mtoe": 0, "twh": 0}, None

        except Exception as e:
            logging.error(f"Prediction failed: {e}", exc_info=True)
            return {"MW": 0, "efficiency": 0, "mtoe": 0, "twh": 0}, None

    def convert_power_to_energy_metrics(self, power_mw: float, input_list: List[float]) -> Dict[str, float]:
        try:
            at, v, ap, rh = input_list

            base_efficiency = 85.0
            temp_factor = max(0.5, 1 - (abs(at - 25) * 0.01))
            pressure_factor = max(0.7, min(1.2, ap / 101.325))
            humidity_factor = max(0.8, 1 - (abs(rh - 50) * 0.002))
            voltage_factor = max(0.6, min(1.1, v / 50))

            efficiency = base_efficiency * temp_factor * pressure_factor * humidity_factor * voltage_factor
            efficiency = min(100, max(0, efficiency))

            mtoe_per_mw_year = 0.00075336
            twh_per_mw_year = 0.000008760

            mtoe = power_mw * mtoe_per_mw_year * (efficiency / 100)
            twh = power_mw * twh_per_mw_year * (efficiency / 100)

            return {
                "MW": round(power_mw, 2),         # ✅ keep MW
                "efficiency": round(efficiency, 2),
                "mtoe": round(mtoe, 6),
                "twh": round(twh, 4)
            }

        except Exception as e:
            logging.error(f"Energy conversion failed: {e}", exc_info=True)
            return {"MW": 0, "efficiency": 0, "mtoe": 0, "twh": 0}

    # ---------------- Regression ----------------
    def run_regression(self, df: pd.DataFrame) -> dict:
        try:
            logging.info(f"Original dataset shape: {df.shape}")
            df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            logging.info(f"Cleaned dataset shape (after NaN/inf removal): {df.shape}")

            if df.shape[0] < 5:
                logging.warning(f"Not enough valid rows after cleaning: {df.shape[0]} rows")
                return {"error": f"Not enough valid rows after cleaning ({df.shape[0]} rows)"}

            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            if y.nunique() <= 1:
                logging.warning("Target column has zero variance, cannot compute R².")
                return {"error": "Target column has zero variance, cannot compute R²."}

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}, Features: {X.shape[1]}")

            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
            model.fit(X_train, y_train)
            logging.info("Gradient Boosting Regressor fitted successfully.")

            # Save the trained regression model
            # Save the trained regression model
            reg_model_path = "backend/models/saved_regressor_model.pkl"
            os.makedirs(os.path.dirname(reg_model_path), exist_ok=True)  # ✅ Create folder if missing
            joblib.dump(model, reg_model_path)
            logging.info(f"Saved regression model to {reg_model_path}")


            def safe_float(x):
                if x is None or pd.isna(x) or math.isnan(x) or math.isinf(x):
                    return None
                return float(x)

            r2_train = safe_float(r2_score(y_train, model.predict(X_train))) if len(y_train) > 1 else None
            r2_test = safe_float(r2_score(y_test, model.predict(X_test))) if len(y_test) > 1 else None
            logging.info(f"R2 Train: {r2_train}, R2 Test: {r2_test}")

            cv_scores, cv_mean = [], None
            if len(df) >= 10:
                raw_cv = cross_val_score(model, X, y, cv=5, scoring="r2")
                cv_scores = [safe_float(s) for s in raw_cv]
                cv_mean = safe_float(np.mean(raw_cv))
                logging.info(f"CV Scores: {cv_scores}, CV Mean: {cv_mean}")

            feature_importances = [safe_float(f) for f in model.feature_importances_]
            logging.info(f"Feature Importances: {feature_importances}")

            return {
                "r2_train": r2_train,
                "r2_test": r2_test,
                "cv_mean": cv_mean,
                "cv_scores": cv_scores,
                "feature_importances": feature_importances,
                "n_rows": df.shape[0],
                "n_features": X.shape[1],
            }

        except Exception as e:
            logging.error(f"Regression analysis failed: {e}", exc_info=True)
            return {"error": f"Analysis failed: {str(e)}"}

    # ---------------- Classification Placeholder ----------------
    def run_classification_placeholder(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            X = df.iloc[:, :-1].astype(float)
            y = df.iloc[:, -1].astype(int)

            logging.info(f"Classification dataset shape: {df.shape}")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            scores = cross_val_score(model, X, y, cv=5)

            logging.info(f"Classification CV Scores: {scores}")

            def safe_float(x):
                if x is None or pd.isna(x) or math.isnan(x) or math.isinf(x):
                    return None
                return float(x)

            return {"accuracy": safe_float(scores.mean()), "cv_scores": [safe_float(s) for s in scores]}
        except Exception as e:
            logging.error(f"Classification analysis failed: {e}", exc_info=True)
            return {"error": f"Analysis failed: {str(e)}"}

    # ---------------- Method Recommendation ----------------
    def recommend_method(self, temp: float, pressure: float, scalability: str, budget: str) -> Dict[str, Any]:
        logging.info(f"Method recommendation request: Temp={temp}, Pressure={pressure}, Scalability={scalability}, Budget={budget}")
        result = self.method_recommender.recommend(temp, pressure, scalability, budget)
        logging.info(f"Recommendation result: {result}")
        return result

    # ---------------- Allocation ----------------
    def optimize_allocation(self, requests: list):
        total_energy = 100
        allocations = {}

        total_requested = sum(req["amount"] for req in requests)
        if total_requested == 0:
            # Avoid division by zero
            for req in requests:
                allocations[req["sector"]] = 0
            return allocations

        for req in requests:
            allocations[req["sector"]] = round(req["amount"] / total_requested * total_energy)

        # Adjust rounding errors
        allocated_total = sum(allocations.values())
        diff = total_energy - allocated_total
        if diff != 0:
            # Add difference to first sector
            allocations[requests[0]["sector"]] += diff

        return allocations
