"""
FreightIQ - Inference Engine
Loads the trained LightGBM model pipeline, computes geospatial distance between states/coords,
and predicts freight cost with confidence intervals.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Brazilian state coordinate centroids for distance calculations
STATE_CENTROIDS = {
    "SP": (-23.5505, -46.6333), "RJ": (-22.9068, -43.1729), "MG": (-19.9167, -43.9345),
    "PR": (-25.4284, -49.2733), "RS": (-30.0346, -51.2177), "SC": (-27.5954, -48.5480),
    "BA": (-12.9714, -38.5014), "DF": (-15.7975, -47.8919), "GO": (-16.6869, -49.2648),
    "PE": (-8.0476, -34.8770),  "CE": (-3.7172, -38.5433),  "ES": (-20.3155, -40.3128),
    "PA": (-1.4558, -48.5039),  "AM": (-3.1190, -60.0217),  "MT": (-15.6014, -56.0979),
    "MS": (-20.4486, -54.6295), "MA": (-2.5307, -44.3068), "RN": (-5.7945, -35.2110),
    "PB": (-7.1195, -34.8450),  "AL": (-9.6658, -35.7353),  "SE": (-10.9095, -37.0748),
    "PI": (-5.0892, -42.8019),  "TO": (-10.2491, -48.3243), "RO": (-8.7612, -63.9039),
    "AC": (-9.9749, -67.8243),  "AP": (0.0349, -51.0694),   "RR": (2.8197, -60.6733)
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(R * c)

class FreightPredictor:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_dir, "models", "freight_lgbm_pipeline.joblib")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")
            
        print(f"[Predictor] Loading LightGBM model pipeline from {model_path}...")
        self.pipeline = joblib.load(model_path)

    def predict_freight(
        self,
        weight_g: float,
        origin_state: str = "SP",
        dest_state: str = "RJ",
        length_cm: float = 25.0,
        height_cm: float = 15.0,
        width_cm: float = 20.0,
        delivery_speed_days: int = 7,
        purchase_month: int = 10,
        distance_km: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Feeds shipment features into the trained LightGBM pipeline to predict freight cost.
        """
        origin_state = origin_state.strip().upper()
        dest_state = dest_state.strip().upper()
        
        # Calculate distance if not provided
        if distance_km is None:
            c1 = STATE_CENTROIDS.get(origin_state, (-23.5505, -46.6333))
            c2 = STATE_CENTROIDS.get(dest_state, (-22.9068, -43.1729))
            distance_km = round(haversine(c1[0], c1[1], c2[0], c2[1]), 2)
            
        # Volumetric & billable weight
        volume_cm3 = length_cm * height_cm * width_cm
        actual_weight_kg = round(weight_g / 1000.0, 3)
        volumetric_weight_kg = round(volume_cm3 / 6000.0, 3)
        billable_weight_kg = max(actual_weight_kg, volumetric_weight_kg)
        
        is_same_state = 1 if origin_state == dest_state else 0
        is_peak_season = 1 if purchase_month in [11, 12] else 0
        
        # Weight Band
        if billable_weight_kg < 1.0:
            weight_band = "<1kg"
        elif billable_weight_kg < 3.0:
            weight_band = "1-3kg"
        elif billable_weight_kg < 8.0:
            weight_band = "3-8kg"
        elif billable_weight_kg < 15.0:
            weight_band = "8-15kg"
        else:
            weight_band = "15kg+"

        # Distance Band
        if distance_km < 100.0:
            distance_band = "Local (<100km)"
        elif distance_km < 500.0:
            distance_band = "Regional (100-500km)"
        elif distance_km < 1500.0:
            distance_band = "Long Haul (500-1500km)"
        else:
            distance_band = "Cross Country (>1500km)"

        input_data = pd.DataFrame([{
            "distance_km": distance_km,
            "billable_weight_kg": billable_weight_kg,
            "actual_weight_kg": actual_weight_kg,
            "volumetric_weight_kg": volumetric_weight_kg,
            "product_volume_cm3": volume_cm3,
            "estimated_delivery_days": delivery_speed_days,
            "is_same_state": is_same_state,
            "is_peak_season": is_peak_season,
            "purchase_month": purchase_month,
            "seller_state": origin_state,
            "customer_state": dest_state,
            "weight_band": weight_band,
            "distance_band": distance_band
        }])
        
        pred_cost = float(self.pipeline.predict(input_data)[0])
        pred_cost = max(6.50, round(pred_cost, 2))
        
        # Error bound estimation (MAE ~ 1.57 BRL)
        mae_uncertainty = 1.57
        return {
            "predicted_cost_brl": pred_cost,
            "cost_lower_bound_brl": round(max(5.0, pred_cost - mae_uncertainty), 2),
            "cost_upper_bound_brl": round(pred_cost + mae_uncertainty, 2),
            "distance_km": distance_km,
            "billable_weight_kg": billable_weight_kg,
            "actual_weight_kg": actual_weight_kg,
            "volumetric_weight_kg": volumetric_weight_kg,
            "weight_band": weight_band,
            "distance_band": distance_band,
            "route_lane": f"{origin_state}->{dest_state}",
            "is_same_state": bool(is_same_state),
            "is_peak_season": bool(is_peak_season)
        }

if __name__ == "__main__":
    predictor = FreightPredictor()
    result = predictor.predict_freight(
        weight_g=2500,
        origin_state="SP",
        dest_state="BA",
        delivery_speed_days=4
    )
    print("Inference Test Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
