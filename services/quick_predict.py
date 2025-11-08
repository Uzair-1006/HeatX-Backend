import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

DATASET_PATH = "data/Folds5x2_pp.csv"

def predict_power(input_data: dict) -> dict:
    """
    Trains a GradientBoostingRegressor on the dataset and predicts PE (MW),
    along with efficiency, mtoe, and twh.

    Input: dict with keys AT, V, AP, RH
    Output: dict with keys MW, efficiency, mtoe, twh
    """
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model.fit(X, y)

    # Prepare input
    at = float(input_data.get("AT", 0))
    v = float(input_data.get("V", 0))
    ap = float(input_data.get("AP", 0))
    rh = float(input_data.get("RH", 0))
    input_list = [at, v, ap, rh]

    # Predict MW
    power_mw = float(model.predict([input_list])[0])

    # Convert to energy metrics
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
        "MW": round(power_mw, 2),
        "efficiency": round(efficiency, 2),
        "mtoe": round(mtoe, 6),
        "twh": round(twh, 4)
    }
