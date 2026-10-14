"""
FreightIQ - Stage 2: SQL Analytics Engine (Dual BigQuery / DuckDB Engine)
Runs the regional, seasonal, and lane profitability SQL queries and exports analytical summaries.
"""

import os
import json
import duckdb
import pandas as pd

def run_regional_and_seasonal_analytics(processed_data_path: str, output_json_path: str):
    """
    Executes analytical SQL queries across the curated dataset.
    """
    con = duckdb.connect(database=":memory:")
    path_sql = processed_data_path.replace("\\", "/")
    
    # Register view
    con.execute(f"CREATE VIEW clean_freight_features AS SELECT * FROM '{path_sql}'")
    
    # 1. Interstate vs Intrastate Summary
    q_state_type = """
    SELECT 
        CASE WHEN is_same_state = 1 THEN 'Intrastate (Same State)' ELSE 'Interstate (Cross State)' END AS shipment_type,
        COUNT(*) AS shipment_count,
        ROUND(AVG(freight_value), 2) AS avg_freight,
        ROUND(AVG(distance_km), 1) AS avg_distance_km,
        ROUND(AVG(freight_value / NULLIF(distance_km, 0)), 4) AS cost_per_km,
        ROUND(AVG(freight_ratio) * 100, 2) AS avg_freight_pct_of_price
    FROM clean_freight_features
    GROUP BY 1
    """
    df_state_type = con.execute(q_state_type).df()
    
    # 2. Monthly / Seasonal Freight Trends
    q_seasonal = """
    SELECT 
        purchase_month,
        CASE purchase_month
            WHEN 1 THEN 'Jan' WHEN 2 THEN 'Feb' WHEN 3 THEN 'Mar'
            WHEN 4 THEN 'Apr' WHEN 5 THEN 'May' WHEN 6 THEN 'Jun'
            WHEN 7 THEN 'Jul' WHEN 8 THEN 'Aug' WHEN 9 THEN 'Sep'
            WHEN 10 THEN 'Oct' WHEN 11 THEN 'Nov (Black Friday)' WHEN 12 THEN 'Dec (Holidays)'
        END AS month_name,
        is_peak_season,
        COUNT(*) AS order_count,
        ROUND(AVG(freight_value), 2) AS avg_freight_value,
        ROUND(AVG(estimated_delivery_days), 1) AS avg_delivery_days,
        ROUND(AVG(price), 2) AS avg_product_price
    FROM clean_freight_features
    GROUP BY 1, 2, 3
    ORDER BY purchase_month
    """
    df_seasonal = con.execute(q_seasonal).df()
    
    # 3. Top Shipping Lanes (Corridors)
    q_lanes = """
    SELECT 
        seller_state,
        customer_state,
        route_lane,
        is_same_state,
        COUNT(*) AS shipment_count,
        ROUND(AVG(distance_km), 1) AS distance_km,
        ROUND(AVG(freight_value), 2) AS avg_freight,
        ROUND(AVG(estimated_delivery_days), 1) AS avg_est_days,
        ROUND(MIN(freight_value), 2) AS min_freight,
        ROUND(MAX(freight_value), 2) AS max_freight
    FROM clean_freight_features
    GROUP BY 1, 2, 3, 4
    ORDER BY shipment_count DESC
    LIMIT 20
    """
    df_lanes = con.execute(q_lanes).df()
    
    # 4. Weight Band & Distance Band Matrix
    q_matrix = """
    SELECT 
        weight_band,
        distance_band,
        COUNT(*) AS count,
        ROUND(AVG(freight_value), 2) AS avg_freight,
        ROUND(MIN(freight_value), 2) AS min_freight,
        ROUND(MAX(freight_value), 2) AS max_freight
    FROM clean_freight_features
    GROUP BY 1, 2
    ORDER BY weight_band, distance_band
    """
    df_matrix = con.execute(q_matrix).df()
    
    summary = {
        "interstate_vs_intrastate": df_state_type.to_dict(orient="records"),
        "seasonal_trends": df_seasonal.to_dict(orient="records"),
        "top_lanes": df_lanes.to_dict(orient="records"),
        "weight_distance_matrix": df_matrix.to_dict(orient="records")
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"[Analytics Engine] Successfully executed SQL analytics!")
    print(f"[Analytics Engine] Saved analytical summary to {output_json_path}")
    con.close()
    return summary

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    parquet = os.path.join(base, "data", "processed", "clean_freight_features.parquet")
    out_json = os.path.join(base, "data", "processed", "analytics_summary.json")
    run_regional_and_seasonal_analytics(parquet, out_json)
