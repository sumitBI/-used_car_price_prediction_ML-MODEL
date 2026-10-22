"""Load the trained model and predict historical used-car selling prices."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_car_price_model.joblib"
FEATURES = ["year", "km_driven", "fuel", "seller_type", "transmission", "owner"]

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run notebooks 06 and 08 to train the final model first.")
    return joblib.load(MODEL_PATH)  # Load only files created by you / trusted sources.

def predict_price(model, *, year, km_driven, fuel, seller_type, transmission, owner):
    if year < 1992 or year > 2020 or km_driven < 0:
        raise ValueError("Year must be 1992–2020 and kilometres cannot be negative.")
    row = pd.DataFrame([{
        "year": year, "km_driven": km_driven, "fuel": fuel,
        "seller_type": seller_type, "transmission": transmission, "owner": owner
    }], columns=FEATURES)
    return float(model.predict(row)[0])
