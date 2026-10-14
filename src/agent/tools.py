"""
FreightIQ - Stage 5: Agent Tools
Defines tool functions for Gemini function calling.
"""

import json
import os
from typing import Dict, Any, Optional
from src.modeling.predict import FreightPredictor
from src.pricing.pricing_engine import calculate_recommended_price

_predictor = None

def get_predictor() -> FreightPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FreightPredictor()
    return _predictor

def tool_predict_shipping_cost(
    weight_g: float,
    origin_state: str = "SP",
    dest_state: str = "RJ",
    delivery_speed_days: int = 7,
    length_cm: float = 25.0,
    height_cm: float = 15.0,
    width_cm: float = 20.0,
    purchase_month: int = 10
) -> Dict[str, Any]:
    """
    Predicts the baseline carrier shipping cost for a cargo shipment using the trained LightGBM model.
    
    Args:
        weight_g: Cargo deadweight in grams (e.g. 2500 for 2.5kg)
        origin_state: 2-letter origin Brazilian state code (e.g. SP, RJ, MG)
        dest_state: 2-letter destination Brazilian state code (e.g. BA, PE, RS)
        delivery_speed_days: Target SLA delivery timeframe in days (1 to 30)
        length_cm: Package length in centimeters
        height_cm: Package height in centimeters
        width_cm: Package width in centimeters
        purchase_month: Shipment calendar month (1 to 12)
    """
    predictor = get_predictor()
    return predictor.predict_freight(
        weight_g=weight_g,
        origin_state=origin_state,
        dest_state=dest_state,
        length_cm=length_cm,
        height_cm=height_cm,
        width_cm=width_cm,
        delivery_speed_days=delivery_speed_days,
        purchase_month=purchase_month
    )

def tool_calculate_target_price(
    predicted_cost: float,
    target_margin_pct: float = 20.0,
    delivery_speed_days: int = 7,
    is_same_state: bool = False,
    destination_state: str = "RJ",
    billable_weight_kg: float = 2.0,
    is_oversized: bool = False,
    is_peak_season: bool = False
) -> Dict[str, Any]:
    """
    Calculates the recommended commercial quote to charge the customer to achieve target profit margin.
    
    Args:
        predicted_cost: The carrier shipping cost in BRL
        target_margin_pct: Desired gross profit margin percentage (e.g. 25 for 25%)
        delivery_speed_days: Delivery commitment speed in days
        is_same_state: True if intrastate shipment, False if interstate
        destination_state: 2-letter destination state
        billable_weight_kg: Total billable weight (higher of actual or volumetric)
        is_oversized: True if cargo exceeds standard pallet/box specifications
        is_peak_season: True if shipping during peak Q4 e-commerce holiday surge
    """
    return calculate_recommended_price(
        predicted_cost=predicted_cost,
        target_margin_pct=target_margin_pct,
        delivery_speed_days=delivery_speed_days,
        is_same_state=is_same_state,
        destination_state=destination_state,
        billable_weight_kg=billable_weight_kg,
        is_oversized=is_oversized,
        is_peak_season=is_peak_season
    )

def tool_get_lane_analytics(origin_state: str, dest_state: str) -> Dict[str, Any]:
    """
    Fetches historical BigQuery freight trends and lane benchmarks for a given origin and destination corridor.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    summary_path = os.path.join(base_dir, "data", "processed", "analytics_summary.json")
    
    lane_key = f"{origin_state.strip().upper()}->{dest_state.strip().upper()}"
    
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data = json.load(f)
            
        top_lanes = data.get("top_lanes", [])
        for lane in top_lanes:
            if lane.get("route_lane") == lane_key:
                return {
                    "found_direct_lane": True,
                    "lane": lane_key,
                    "historical_shipments": lane.get("shipment_count"),
                    "avg_historical_freight": lane.get("avg_freight"),
                    "min_historical_freight": lane.get("min_freight"),
                    "max_historical_freight": lane.get("max_freight"),
                    "avg_transit_days": lane.get("avg_est_days"),
                    "distance_km": lane.get("distance_km")
                }
                
        # Return regional state summary if specific lane not found
        interstate = data.get("interstate_vs_intrastate", [])
        is_same = (origin_state.strip().upper() == dest_state.strip().upper())
        state_benchmark = next((item for item in interstate if ("Intrastate" in item["shipment_type"]) == is_same), {})
        
        return {
            "found_direct_lane": False,
            "lane": lane_key,
            "regional_benchmark": state_benchmark,
            "note": "Corridor benchmark estimated from regional state data."
        }
        
    return {"error": "Analytics summary data not found. Run run_analytics.py first."}
