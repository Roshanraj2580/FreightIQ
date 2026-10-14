"""
FreightIQ - Stage 4: Dynamic Pricing Logic Engine
Transforms ML predicted cost into a commercial freight quote based on target margin,
risk factors, service speed, and route constraints.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class PricingRequest(BaseModel):
    predicted_cost: float = Field(..., gt=0, description="Base freight cost predicted by ML model in BRL")
    target_margin_pct: float = Field(20.0, ge=0.0, lt=90.0, description="Target gross margin percentage (e.g. 20 for 20%)")
    delivery_speed_days: int = Field(7, ge=1, le=45, description="Expected delivery lead time in days")
    is_same_state: bool = Field(False, description="Whether shipment is intrastate or interstate")
    destination_state: str = Field("RJ", description="Destination state code (e.g. SP, RJ, BA, AM)")
    billable_weight_kg: float = Field(2.0, gt=0, description="Billable weight in kg")
    is_oversized: bool = Field(False, description="Whether shipment is oversized or low-density bulky")
    is_peak_season: bool = Field(False, description="Whether current date is in Q4 peak holiday season")

class PricingResponse(BaseModel):
    predicted_cost: float
    target_margin_pct: float
    recommended_price: float
    effective_margin_pct: float
    gross_profit: float
    breakeven_price: float
    min_floor_price: float
    risk_tier: str
    adjustments: Dict[str, float]
    summary_text: str

# High-risk / remote logistics states in Brazil
REMOTE_STATES = {"AM", "PA", "RR", "AP", "AC", "RO", "MT", "MS"}

def calculate_recommended_price(
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
    Computes commercial freight price quote from predicted cost and business logic.
    """
    if predicted_cost <= 0:
        raise ValueError("Predicted cost must be greater than zero.")
        
    margin_ratio = max(0.0, min(target_margin_pct / 100.0, 0.85))
    
    # 1. Base Cost-Plus Target Margin Calculation
    # Price = Cost / (1 - Margin) ensures exact target gross margin
    raw_base_price = predicted_cost / (1.0 - margin_ratio)
    
    # 2. Multi-factor Dynamic Surcharges and Adjustments
    adjustments = {}
    multiplier = 1.0
    
    # A. Delivery Speed Factor (SLA Urgency)
    if delivery_speed_days <= 3:
        speed_surcharge = 0.18  # +18% for express SLA
        multiplier += speed_surcharge
        adjustments["express_delivery_premium"] = round(raw_base_price * speed_surcharge, 2)
    elif delivery_speed_days <= 5:
        speed_surcharge = 0.08  # +8% for expedited SLA
        multiplier += speed_surcharge
        adjustments["expedited_delivery_premium"] = round(raw_base_price * speed_surcharge, 2)
    elif delivery_speed_days >= 15:
        economy_discount = -0.05  # -5% economy consolidation discount
        multiplier += economy_discount
        adjustments["economy_discount"] = round(raw_base_price * economy_discount, 2)

    # B. Remote Corridor / State Risk Buffer
    dest_clean = destination_state.strip().upper()
    if dest_clean in REMOTE_STATES:
        remote_risk = 0.10  # +10% risk contingency for riverine or long-haul access
        multiplier += remote_risk
        adjustments["remote_state_contingency"] = round(raw_base_price * remote_risk, 2)
    elif not is_same_state:
        interstate_tax_buffer = 0.04  # +4% interstate logistics fee
        multiplier += interstate_tax_buffer
        adjustments["interstate_tax_buffer"] = round(raw_base_price * interstate_tax_buffer, 2)

    # C. Bulky / Oversized Surcharge
    if is_oversized or billable_weight_kg > 20.0:
        bulky_fee = 0.12  # +12% pallet handling and space penalty
        multiplier += bulky_fee
        adjustments["oversized_pallet_fee"] = round(raw_base_price * bulky_fee, 2)

    # D. Peak Season Capacity Premium
    if is_peak_season:
        peak_fee = 0.08  # +8% holiday peak demand buffer
        multiplier += peak_fee
        adjustments["peak_season_buffer"] = round(raw_base_price * peak_fee, 2)

    # Final adjusted price
    adjusted_price = raw_base_price * multiplier
    
    # 3. Floor and Boundary Constraints
    # Minimum floor price ensures carrier cost + minimum 5% safety margin
    min_floor_price = round(predicted_cost * 1.05, 2)
    breakeven_price = round(predicted_cost, 2)
    
    final_price = max(adjusted_price, min_floor_price)
    final_price = round(final_price, 2)
    
    # Financial breakdown
    gross_profit = round(final_price - predicted_cost, 2)
    effective_margin_pct = round((gross_profit / final_price) * 100.0, 2)
    
    # Determine risk tier
    if dest_clean in REMOTE_STATES or delivery_speed_days <= 3:
        risk_tier = "High"
    elif not is_same_state or is_oversized:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"
        
    summary_text = (
        f"For a predicted baseline cost of R$ {predicted_cost:.2f} with a {target_margin_pct:.1f}% target margin, "
        f"the recommended commercial quote is R$ {final_price:.2f} (yielding R$ {gross_profit:.2f} gross profit, "
        f"{effective_margin_pct:.1f}% effective margin). Breakeven price is R$ {breakeven_price:.2f}."
    )
    
    return {
        "predicted_cost": round(predicted_cost, 2),
        "target_margin_pct": round(target_margin_pct, 2),
        "recommended_price": final_price,
        "effective_margin_pct": effective_margin_pct,
        "gross_profit": gross_profit,
        "breakeven_price": breakeven_price,
        "min_floor_price": min_floor_price,
        "risk_tier": risk_tier,
        "adjustments": adjustments,
        "summary_text": summary_text
    }

if __name__ == "__main__":
    # Test example quote
    sample = calculate_recommended_price(
        predicted_cost=24.50,
        target_margin_pct=22.0,
        delivery_speed_days=3,
        is_same_state=False,
        destination_state="BA",
        billable_weight_kg=4.5,
        is_oversized=False,
        is_peak_season=False
    )
    print("Pricing Calculation Test:")
    for k, v in sample.items():
        print(f"  {k}: {v}")
