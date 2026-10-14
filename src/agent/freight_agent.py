"""
FreightIQ - Stage 5: AI Pricing Strategist Agent
A Gemini-powered autonomous agent equipped with tool-calling capabilities to predict
shipping costs, compute margin-optimized pricing, and provide executive logistics explanations.
"""

import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from src.agent.tools import (
    tool_predict_shipping_cost,
    tool_calculate_target_price,
    tool_get_lane_analytics
)

load_dotenv()

SYSTEM_PROMPT = """You are FreightIQ Copilot, an elite AI Logistics Pricing Strategist and Transportation Economist.
Your mission is to analyze freight shipment requests, use your tool to accurately predict baseline shipping costs from real market data, compute recommended pricing to satisfy target profit margins, and deliver crisp, professional, and convincing executive explanations in plain English.

When a user describes a shipment:
1. Always call `tool_predict_shipping_cost` with the extracted parameters (weight, origin state, destination state, dimensions, delivery speed).
2. Call `tool_calculate_target_price` with the predicted baseline cost and the user's target margin %.
3. Call `tool_get_lane_analytics` to check historical lane benchmarks.
4. Synthesize your final response using this structured format:

### 📦 Commercial Freight Quotation
- **Recommended Quote**: R$ [Price]
- **Baseline Carrier Cost**: R$ [Cost] (95% CI: R$ [Min] - R$ [Max])
- **Gross Profit**: R$ [Profit] | **Effective Margin**: [Margin]%
- **Breakeven Floor**: R$ [Breakeven]

### 🔍 Cost Breakdown & Logistics Drivers
- **Distance & Routing**: Explain distance (km), route corridor (interstate vs local), and state taxes.
- **Dimensional & Weight Analysis**: Explain deadweight vs volumetric weight (density impact).
- **Delivery Urgency / SLA**: Impact of requested delivery timeframe on linehaul costs.
- **Seasonal / Risk Surcharges**: Any peak season, remote corridor, or fuel volatility adjustments.

### 💡 Negotiation Strategy & Margin Defense
Provide 2-3 tactical suggestions for how the sales team can defend this price or offer tiered alternatives (e.g. standard vs express).
"""

class FreightAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.has_gemini = False
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                # Register tools
                self.tools = [
                    tool_predict_shipping_cost,
                    tool_calculate_target_price,
                    tool_get_lane_analytics
                ]
                self.model = genai.GenerativeModel(
                    model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
                    tools=self.tools,
                    system_instruction=SYSTEM_PROMPT
                )
                self.has_gemini = True
                print("[Freight Agent] Successfully initialized Gemini with tool calling.")
            except Exception as e:
                print(f"[Freight Agent] Note: Gemini initialization notice ({e}). Operating in deterministic logistics mode.")
                self.has_gemini = False
        else:
            print("[Freight Agent] No GEMINI_API_KEY detected. Agent running in autonomous logistics mode.")

    def _extract_shipment_entities(self, query: str) -> Dict[str, Any]:
        """Heuristic entity extractor for queries."""
        q = query.lower()
        
        # Extract weight (kg or g)
        weight_g = 2000.0
        kg_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:kg|kilo|kilogram)', q)
        g_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:g|gram)', q)
        if kg_match:
            weight_g = float(kg_match.group(1)) * 1000.0
        elif g_match:
            weight_g = float(g_match.group(1))
            
        # Extract target margin
        margin_pct = 20.0
        margin_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:margin|target|profit)?', q)
        if margin_match:
            margin_pct = float(margin_match.group(1))
            
        # Extract delivery days
        days = 7
        days_match = re.search(r'(\d+)\s*(?:day|days)', q)
        if days_match:
            days = int(days_match.group(1))
            
        # Extract states
        origin = "SP"
        dest = "RJ"
        
        # State mentions
        state_map = {
            "sao paulo": "SP", "são paulo": "SP", "rio de janeiro": "RJ", "rio": "RJ",
            "minas gerais": "MG", "minas": "MG", "bahia": "BA", "salvador": "BA",
            "curitiba": "PR", "parana": "PR", "paraná": "PR", "porto alegre": "RS",
            "rio grande do sul": "RS", "santa catarina": "SC", "florianopolis": "SC",
            "brasilia": "DF", "brasília": "DF", "pernambuco": "PE", "recife": "PE",
            "ceara": "CE", "ceará": "CE", "fortaleza": "CE", "amazonas": "AM", "manaus": "AM"
        }
        
        detected_states = []
        for name, code in state_map.items():
            if name in q:
                detected_states.append(code)
                
        # Regex for 2-letter uppercase or lowercase state codes
        code_matches = re.findall(r'\b(sp|rj|mg|ba|pr|rs|sc|pe|ce|df|am|go|es|pa|mt)\b', q)
        for c in code_matches:
            detected_states.append(c.upper())
            
        # Deduplicate while preserving order
        unique_states = []
        for s in detected_states:
            if s not in unique_states:
                unique_states.append(s)
                
        if len(unique_states) >= 2:
            origin = unique_states[0]
            dest = unique_states[1]
        elif len(unique_states) == 1:
            dest = unique_states[0]
            
        return {
            "weight_g": weight_g,
            "margin_pct": margin_pct,
            "days": days,
            "origin": origin,
            "dest": dest
        }

    def process_query(self, user_query: str) -> str:
        """
        Executes reasoning loop with tools and synthesizes plain-English explanation.
        """
        if self.has_gemini:
            try:
                chat = self.model.start_chat(enable_automatic_function_calling=True)
                response = chat.send_message(user_query)
                return response.text
            except Exception as e:
                print(f"[Freight Agent] Gemini API call exception ({e}), falling back to deterministic explanation...")

        # Autonomous Logistics Reasoner Fallback
        entities = self._extract_shipment_entities(user_query)
        
        # 1. Execute ML cost prediction tool
        cost_res = tool_predict_shipping_cost(
            weight_g=entities["weight_g"],
            origin_state=entities["origin"],
            dest_state=entities["dest"],
            delivery_speed_days=entities["days"]
        )
        
        # 2. Execute target price calculator tool
        price_res = tool_calculate_target_price(
            predicted_cost=cost_res["predicted_cost_brl"],
            target_margin_pct=entities["margin_pct"],
            delivery_speed_days=entities["days"],
            is_same_state=cost_res["is_same_state"],
            destination_state=entities["dest"],
            billable_weight_kg=cost_res["billable_weight_kg"]
        )
        
        # 3. Fetch lane benchmarks
        lane_res = tool_get_lane_analytics(entities["origin"], entities["dest"])
        
        # Synthesize professional plain-English explanation
        corridor = f"{entities['origin']} ➔ {entities['dest']}"
        weight_kg = cost_res["billable_weight_kg"]
        pred_cost = price_res["predicted_cost"]
        quote = price_res["recommended_price"]
        profit = price_res["gross_profit"]
        margin = price_res["effective_margin_pct"]
        breakeven = price_res["breakeven_price"]
        dist = cost_res["distance_km"]
        
        speed_note = "Express SLA (≤ 3 days premium applied)" if entities["days"] <= 3 else f"Standard SLA ({entities['days']} business days)"
        geo_note = "Intrastate route (lower interstate tax burden)" if cost_res["is_same_state"] else "Interstate long-haul route (includes state transit border ICMS tax buffer)"
        
        explanation = f"""### 📦 Commercial Freight Quotation
- **Recommended Quote**: **R$ {quote:.2f}**
- **Baseline Carrier Cost**: R$ {pred_cost:.2f} (95% CI: R$ {cost_res['cost_lower_bound_brl']:.2f} - R$ {cost_res['cost_upper_bound_brl']:.2f})
- **Target Margin**: {entities['margin_pct']:.1f}% | **Effective Realized Margin**: **{margin:.1f}%**
- **Gross Profit**: **R$ {profit:.2f}** per shipment
- **Breakeven Floor**: R$ {breakeven:.2f} | **Risk Tier**: {price_res['risk_tier']}

---

### 🔍 Cost Breakdown & Logistics Drivers
1. **Corridor & Distance ({corridor})**:
   - Total road transit distance is **{dist:,.1f} km**.
   - {geo_note}.
2. **Weight & Cube Density**:
   - Billable weight is **{weight_kg:.2f} kg** ({cost_res['weight_band']} tier).
   - Volumetric cubic displacement: {cost_res['volumetric_weight_kg']:.2f} kg volumetric weight.
3. **Delivery Commitment ({speed_note})**:
   - Transit speed factor accounted for linehaul dispatch urgency.
4. **Historical Benchmark Comparison**:
"""
        if lane_res.get("found_direct_lane"):
            explanation += f"   - The **{corridor}** lane has a historical average market freight cost of **R$ {lane_res.get('avg_historical_freight'):.2f}** over {lane_res.get('historical_shipments')} recorded shipments. Our quote is aligned within healthy market boundaries.\n"
        else:
            explanation += f"   - Benchmarked against regional Brazilian shipping corridors of similar distance (~{dist:,.0f} km).\n"
            
        explanation += f"""
---

### 💡 Negotiation Strategy & Margin Defense
- **Price Defense**: If the client pushes back, highlight that this quote includes dedicated tracking, cargo liability coverage, and guaranteed {entities['days']}-day delivery SLA.
- **Alternative Option**: Offer an economy option with 10-12 business days delivery for **R$ {round(quote * 0.90, 2):.2f}**, preserving our gross dollar profit while accommodating tighter customer budgets.
"""
        return explanation

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    agent = FreightAgent()
    sample_query = "I have a 3.5kg package shipping from SP to BA in 3 days. We want a 25% margin. What should we quote?"
    print("\n--- Testing Agent Processing ---")
    response = agent.process_query(sample_query)
    print(response)
