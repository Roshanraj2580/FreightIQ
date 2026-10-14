"""
FreightIQ - Unified Orchestration Entry Point
Run stages, launch API, or start the interactive Gradio UI.
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="FreightIQ: AI-Powered Freight Pricing Recommendation Engine")
    parser.add_argument("--stage", choices=["1", "2", "3", "4", "5", "all"], default=None,
                        help="Execute a specific stage (1=ETL, 2=Cloud Analytics, 3=ML Modeling, 4=Pricing Logic, 5=App, all=All)")
    parser.add_argument("--mode", choices=["ui", "api", "both"], default="ui",
                        help="Service mode to start: 'ui' (Gradio dashboard), 'api' (FastAPI REST server), or 'both'")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the service on")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)

    if args.stage == "1" or args.stage == "all":
        print("\n=== Executing Stage 1: Data Pipeline (ETL & Feature Engineering) ===")
        from src.pipeline.data_generator import generate_olist_dataset
        from src.pipeline.duckdb_pipeline import run_duckdb_pipeline
        raw_dir = os.path.join(base_dir, "data", "raw")
        proc_dir = os.path.join(base_dir, "data", "processed")
        if not os.path.exists(os.path.join(raw_dir, "olist_orders_dataset.csv")):
            generate_olist_dataset(raw_dir, num_orders=12000)
        run_duckdb_pipeline(raw_dir, proc_dir)

    if args.stage == "2" or args.stage == "all":
        print("\n=== Executing Stage 2: Cloud Storage & BigQuery SQL Analytics ===")
        from src.cloud.run_analytics import run_regional_and_seasonal_analytics
        parquet_file = os.path.join(base_dir, "data", "processed", "clean_freight_features.parquet")
        summary_file = os.path.join(base_dir, "data", "processed", "analytics_summary.json")
        run_regional_and_seasonal_analytics(parquet_file, summary_file)

    if args.stage == "3" or args.stage == "all":
        print("\n=== Executing Stage 3: ML Modeling (Linear Regression vs LightGBM) ===")
        from src.modeling.train import train_freight_models
        data_file = os.path.join(base_dir, "data", "processed", "clean_freight_features.csv")
        models_dir = os.path.join(base_dir, "models")
        train_freight_models(data_file, models_dir)

    if args.stage == "4" or args.stage == "all":
        print("\n=== Executing Stage 4: Dynamic Pricing Logic Validation ===")
        from src.pricing.pricing_engine import calculate_recommended_price
        sample = calculate_recommended_price(
            predicted_cost=25.0,
            target_margin_pct=25.0,
            delivery_speed_days=3,
            is_same_state=False,
            destination_state="BA"
        )
        print("Pricing Engine Sample Quote:")
        for k, v in sample.items():
            print(f"  {k}: {v}")

    if args.stage is None or args.stage == "5" or args.stage == "all":
        print("\n=== Stage 5: Launching FreightIQ Application ===")
        if args.mode == "ui":
            print(f"[FreightIQ] Starting Gradio Web UI on http://localhost:{args.port}...")
            from ui.app_gradio import demo
            demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)
        elif args.mode == "api":
            print(f"[FreightIQ] Starting FastAPI REST server on http://localhost:{args.port}...")
            import uvicorn
            uvicorn.run("src.api.main:app", host="0.0.0.0", port=args.port, reload=False)
        elif args.mode == "both":
            import subprocess
            print("[FreightIQ] Starting both FastAPI (port 8000) and Gradio UI (port 7860)...")
            api_proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"])
            from ui.app_gradio import demo
            demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()
