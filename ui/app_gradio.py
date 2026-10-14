"""
FreightIQ - Stage 5: Interactive Web UI (Gradio Dashboard & AI Copilot)
Features real-time quote generation, interactive margin controls, conversational AI copilot,
BigQuery analytical trends, and model diagnostic benchmarks.
"""

import os
import json
import gradio as gr
import pandas as pd

from src.modeling.predict import FreightPredictor
from src.pricing.pricing_engine import calculate_recommended_price
from src.agent.freight_agent import FreightAgent

# Initialize services
predictor = FreightPredictor()
agent = FreightAgent()

# Load analytics and model metrics
base_dir = os.path.dirname(os.path.dirname(__file__))
analytics_path = os.path.join(base_dir, "data", "processed", "analytics_summary.json")
metrics_path = os.path.join(base_dir, "models", "model_metrics.json")

analytics_data = {}
if os.path.exists(analytics_path):
    with open(analytics_path, "r") as f:
        analytics_data = json.load(f)

model_metrics = {}
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        model_metrics = json.load(f)

STATES = [
    "SP", "RJ", "MG", "PR", "RS", "SC", "BA", "DF",
    "GO", "PE", "CE", "ES", "PA", "AM", "MT", "MS", "MA"
]

def quote_calculator(
    origin, dest, weight_kg, length, height, width,
    delivery_days, target_margin, is_peak
):
    """Calculates ML predicted cost and commercial recommended price."""
    weight_g = weight_kg * 1000.0
    
    # 1. Predict cost via LightGBM
    cost_res = predictor.predict_freight(
        weight_g=weight_g,
        origin_state=origin,
        dest_state=dest,
        length_cm=length,
        height_cm=height,
        width_cm=width,
        delivery_speed_days=delivery_days,
        purchase_month=11 if is_peak else 6
    )
    
    pred_cost = cost_res["predicted_cost_brl"]
    
    # 2. Dynamic pricing engine
    pricing_res = calculate_recommended_price(
        predicted_cost=pred_cost,
        target_margin_pct=target_margin,
        delivery_speed_days=delivery_days,
        is_same_state=cost_res["is_same_state"],
        destination_state=dest,
        billable_weight_kg=cost_res["billable_weight_kg"],
        is_oversized=(cost_res["billable_weight_kg"] > 15.0 or length > 80),
        is_peak_season=is_peak
    )
    
    quote_brl = f"R$ {pricing_res['recommended_price']:.2f}"
    cost_brl = f"R$ {pricing_res['predicted_cost']:.2f} (± R$ 1.57)"
    profit_brl = f"R$ {pricing_res['gross_profit']:.2f}"
    margin_pct = f"{pricing_res['effective_margin_pct']:.1f}%"
    breakeven_brl = f"R$ {pricing_res['breakeven_price']:.2f}"
    
    # Build adjustments dataframe
    adj_records = [
        {"Factor": "Baseline Carrier Cost (LightGBM)", "Value": f"R$ {pred_cost:.2f}", "Impact": "Core Baseline"},
        {"Factor": f"Target Margin ({target_margin:.0f}%)", "Value": f"+ R$ {(pred_cost / (1 - target_margin/100) - pred_cost):.2f}", "Impact": "Base Spread"}
    ]
    for k, v in pricing_res["adjustments"].items():
        adj_records.append({
            "Factor": k.replace("_", " ").title(),
            "Value": f"+ R$ {v:.2f}" if v > 0 else f"- R$ {abs(v):.2f}",
            "Impact": "Dynamic Surcharge" if v > 0 else "Discount"
        })
    df_adjustments = pd.DataFrame(adj_records)
    
    # Generate executive narrative
    query_str = f"Quote for {weight_kg:.1f}kg from {origin} to {dest} in {delivery_days} days with {target_margin:.0f}% target margin"
    narrative = agent.process_query(query_str)
    
    return quote_brl, cost_brl, profit_brl, margin_pct, breakeven_brl, df_adjustments, narrative

def chat_interface(user_message, history):
    """Processes conversational questions via the FreightIQ AI agent."""
    if not user_message.strip():
        return ""
    reply = agent.process_query(user_message)
    return reply

# Build Gradio UI
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate"
)

with gr.Blocks(title="FreightIQ - AI Freight Pricing Engine") as demo:
    gr.Markdown(
        """
        # 🚚 FreightIQ: AI-Powered Shipping Price Recommendation Engine
        ### *Predict baseline carrier cost • Optimize target margin • Explain logistics decisions in plain English*
        """
    )
    
    with gr.Tabs():
        # TAB 1: Real-Time Pricing Engine
        with gr.TabItem("⚡ Smart Price Quotation"):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("#### 📍 Shipment Route & Cargo Parameters")
                    with gr.Row():
                        origin_input = gr.Dropdown(choices=STATES, value="SP", label="Origin State")
                        dest_input = gr.Dropdown(choices=STATES, value="BA", label="Destination State")
                    
                    with gr.Row():
                        weight_input = gr.Slider(0.2, 30.0, value=3.5, step=0.1, label="Actual Weight (kg)")
                        lead_time_input = gr.Slider(1, 25, value=3, step=1, label="Target Delivery Time (Days)")
                        
                    with gr.Accordion("📐 Package Dimensions (cm)", open=False):
                        with gr.Row():
                            len_input = gr.Slider(10, 120, value=30, step=5, label="Length (cm)")
                            height_input = gr.Slider(5, 80, value=20, step=5, label="Height (cm)")
                            width_input = gr.Slider(10, 80, value=25, step=5, label="Width (cm)")
                            
                    gr.Markdown("#### 🎯 Commercial Margin & Market Conditions")
                    with gr.Row():
                        margin_input = gr.Slider(5.0, 50.0, value=25.0, step=1.0, label="Target Gross Margin (%)")
                        peak_input = gr.Checkbox(value=False, label="Q4 Peak Season Surcharge (Nov/Dec)")
                        
                    calc_btn = gr.Button("🚀 Generate Smart Quote & AI Strategy", variant="primary", size="lg")

                with gr.Column(scale=6):
                    gr.Markdown("#### 📊 Recommendation Summary")
                    with gr.Row():
                        quote_output = gr.Textbox(label="Commercial Recommended Quote", interactive=False)
                        cost_output = gr.Textbox(label="Predicted Baseline Cost (LightGBM)", interactive=False)
                    with gr.Row():
                        profit_output = gr.Textbox(label="Projected Gross Profit", interactive=False)
                        margin_output = gr.Textbox(label="Effective Realized Margin", interactive=False)
                        floor_output = gr.Textbox(label="Breakeven Cost Floor", interactive=False)
                        
                    gr.Markdown("#### 📑 Cost & Surcharge Breakdown")
                    table_output = gr.DataFrame(headers=["Factor", "Value", "Impact"], label="Dynamic Surcharge Matrix")
                    
                    gr.Markdown("#### 🧠 AI Strategic Logistics Explanation")
                    narrative_output = gr.Markdown("Click **Generate Smart Quote** to review the AI pricing strategy.")

            calc_btn.click(
                fn=quote_calculator,
                inputs=[
                    origin_input, dest_input, weight_input, len_input, height_input, width_input,
                    lead_time_input, margin_input, peak_input
                ],
                outputs=[quote_output, cost_output, profit_output, margin_output, floor_output, table_output, narrative_output]
            )

        # TAB 2: AI Copilot Chat
        with gr.TabItem("💬 FreightIQ AI Copilot"):
            gr.Markdown(
                """
                ### Conversational Logistics Pricing Assistant
                Ask any complex shipment quotation or pricing strategy question. The copilot uses **tool calling** to query the LightGBM model, apply dynamic business logic, and explain quotes.
                """
            )
            chatbot = gr.Chatbot(height=420)
            msg_input = gr.Textbox(
                placeholder="e.g. 'I want to send 5kg of coffee beans from MG to RJ in 2 days. What should I quote for a 28% margin?'",
                label="Your Logistics Request"
            )
            with gr.Row():
                send_btn = gr.Button("Send Request", variant="primary")
                clear_btn = gr.Button("Clear Chat")
                
            gr.Examples(
                examples=[
                    ["I have a 3.5kg package shipping from SP to BA in 3 days. We want a 25% margin. What should we quote?"],
                    ["What is the freight cost difference between sending 8kg from SP to RJ vs SP to AM (Amazonas)?"],
                    ["If we have a client asking for a discount on a 15kg shipment from PR to RS, what is our breakeven floor at 15% margin?"]
                ],
                inputs=msg_input
            )
            
            def respond(message, chat_history):
                bot_reply = agent.process_query(message)
                chat_history.append((message, bot_reply))
                return "", chat_history
                
            send_btn.click(respond, [msg_input, chatbot], [msg_input, chatbot])
            msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])
            clear_btn.click(lambda: None, None, chatbot, queue=False)

        # TAB 3: BigQuery Regional & Seasonal Analytics
        with gr.TabItem("📈 BigQuery Market Trends"):
            gr.Markdown("### Historical Freight Patterns & Regional Corridors (Olist Analytics)")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🛣️ Interstate vs Intrastate Efficiency")
                    state_summary_df = pd.DataFrame(analytics_data.get("interstate_vs_intrastate", []))
                    gr.DataFrame(state_summary_df)
                with gr.Column():
                    gr.Markdown("#### 📅 Monthly & Holiday Season Trends")
                    seasonal_df = pd.DataFrame(analytics_data.get("seasonal_trends", []))
                    gr.DataFrame(seasonal_df)

            gr.Markdown("#### 🚛 Top Shipping Corridors (Volume, Cost, and Lead Time)")
            lanes_df = pd.DataFrame(analytics_data.get("top_lanes", []))
            gr.DataFrame(lanes_df)

        # TAB 4: ML Modeling Benchmarks
        with gr.TabItem("🔬 Model Benchmarks & Explainability"):
            gr.Markdown("### Machine Learning Validation: Linear Regression Baseline vs. LightGBM")
            
            models_info = model_metrics.get("models", {})
            lr_info = models_info.get("linear_regression_baseline", {})
            lgb_info = models_info.get("lightgbm_regressor", {})
            
            comparison_data = [
                {"Metric": "Root Mean Squared Error (RMSE)", "Linear Baseline": f"R$ {lr_info.get('rmse')}", "LightGBM": f"R$ {lgb_info.get('rmse')}", "Advantage": f"{lgb_info.get('rmse_improvement_pct')}% Lower Error"},
                {"Metric": "Mean Absolute Error (MAE)", "Linear Baseline": f"R$ {lr_info.get('mae')}", "LightGBM": f"R$ {lgb_info.get('mae')}", "Advantage": f"{lgb_info.get('mae_improvement_pct')}% Lower Error"},
                {"Metric": "R² (Coefficient of Determination)", "Linear Baseline": f"{lr_info.get('r2')}", "LightGBM": f"{lgb_info.get('r2')}", "Advantage": "Higher Explainability"},
                {"Metric": "Mean Absolute % Error (MAPE)", "Linear Baseline": f"{lr_info.get('mape')}%", "LightGBM": f"{lgb_info.get('mape')}%", "Advantage": "Sub-7% Error Rate"}
            ]
            gr.DataFrame(pd.DataFrame(comparison_data), label="Performance Comparison Matrix")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🎯 Top Feature Importance Ranking (LightGBM)")
                    fi_list = model_metrics.get("statistical_checks", {}).get("feature_importance_ranking", [])
                    df_fi = pd.DataFrame(fi_list, columns=["Feature", "Splits / Information Gain"])
                    gr.DataFrame(df_fi)
                    
                with gr.Column():
                    gr.Markdown("#### 🔍 Correlation with Freight Cost")
                    corr_dict = model_metrics.get("statistical_checks", {}).get("correlations_with_target", {})
                    df_corr = pd.DataFrame(list(corr_dict.items()), columns=["Feature", "Pearson Correlation (r)"])
                    gr.DataFrame(df_corr)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
