"""
FreightIQ - Stage 1 (High-Performance Engine): DuckDB SQL Pipeline
Executes vectorized columnar joins and feature engineering mirroring the PySpark pipeline.
Exports to parquet and CSV for BigQuery loading and Machine Learning modeling.
"""

import os
import duckdb
import pandas as pd

def run_duckdb_pipeline(raw_data_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Executes vectorized ETL pipeline using DuckDB.
    """
    os.makedirs(output_dir, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    
    print("[DuckDB Pipeline] Connecting to in-memory vectorized analytical engine...")
    
    # Path mappings
    orders_csv = os.path.join(raw_data_dir, "olist_orders_dataset.csv").replace("\\", "/")
    items_csv = os.path.join(raw_data_dir, "olist_order_items_dataset.csv").replace("\\", "/")
    products_csv = os.path.join(raw_data_dir, "olist_products_dataset.csv").replace("\\", "/")
    customers_csv = os.path.join(raw_data_dir, "olist_customers_dataset.csv").replace("\\", "/")
    sellers_csv = os.path.join(raw_data_dir, "olist_sellers_dataset.csv").replace("\\", "/")
    geo_csv = os.path.join(raw_data_dir, "olist_geolocation_dataset.csv").replace("\\", "/")

    query = f"""
    WITH geo_centroids AS (
        -- Aggregate zip code prefix centroids to eliminate duplication
        SELECT 
            geolocation_zip_code_prefix AS zip_prefix,
            AVG(geolocation_lat) AS geo_lat,
            AVG(geolocation_lng) AS geo_lng
        FROM read_csv_auto('{geo_csv}')
        GROUP BY geolocation_zip_code_prefix
    ),
    clean_orders AS (
        SELECT 
            order_id,
            customer_id,
            order_status,
            CAST(order_purchase_timestamp AS TIMESTAMP) AS purchase_time,
            CAST(order_estimated_delivery_date AS TIMESTAMP) AS est_delivery_time,
            ROUND(date_diff('hour', CAST(order_purchase_timestamp AS TIMESTAMP), CAST(order_estimated_delivery_date AS TIMESTAMP)) / 24.0, 1) AS estimated_delivery_days
        FROM read_csv_auto('{orders_csv}')
        WHERE order_status = 'delivered'
          AND order_purchase_timestamp IS NOT NULL
          AND order_estimated_delivery_date IS NOT NULL
    ),
    clean_products AS (
        SELECT 
            product_id,
            COALESCE(product_category_name, 'outros') AS product_category_name,
            COALESCE(product_weight_g, 500.0) AS product_weight_g,
            COALESCE(product_length_cm, 20.0) AS product_length_cm,
            COALESCE(product_height_cm, 15.0) AS product_height_cm,
            COALESCE(product_width_cm, 15.0) AS product_width_cm,
            (COALESCE(product_length_cm, 20.0) * COALESCE(product_height_cm, 15.0) * COALESCE(product_width_cm, 15.0)) AS product_volume_cm3,
            ROUND((COALESCE(product_length_cm, 20.0) * COALESCE(product_height_cm, 15.0) * COALESCE(product_width_cm, 15.0)) / 6000.0, 3) AS volumetric_weight_kg,
            ROUND(COALESCE(product_weight_g, 500.0) / 1000.0, 3) AS actual_weight_kg,
            GREATEST(ROUND(COALESCE(product_weight_g, 500.0) / 1000.0, 3), ROUND((COALESCE(product_length_cm, 20.0) * COALESCE(product_height_cm, 15.0) * COALESCE(product_width_cm, 15.0)) / 6000.0, 3)) AS billable_weight_kg
        FROM read_csv_auto('{products_csv}')
    ),
    clean_items AS (
        SELECT 
            order_id,
            order_item_id,
            product_id,
            seller_id,
            price,
            freight_value
        FROM read_csv_auto('{items_csv}')
        WHERE freight_value > 0 AND price > 0
    ),
    joined AS (
        SELECT 
            i.order_id,
            i.order_item_id,
            i.product_id,
            i.seller_id,
            o.customer_id,
            i.price,
            i.freight_value,
            ROUND(i.freight_value / i.price, 4) AS freight_ratio,
            p.product_category_name,
            p.product_weight_g,
            p.actual_weight_kg,
            p.volumetric_weight_kg,
            p.billable_weight_kg,
            p.product_length_cm,
            p.product_height_cm,
            p.product_width_cm,
            p.product_volume_cm3,
            s.seller_state,
            c.customer_state,
            CASE WHEN c.customer_state = s.seller_state THEN 1 ELSE 0 END AS is_same_state,
            s.seller_state || '->' || c.customer_state AS route_lane,
            o.estimated_delivery_days,
            EXTRACT(YEAR FROM o.purchase_time) AS purchase_year,
            EXTRACT(MONTH FROM o.purchase_time) AS purchase_month,
            EXTRACT(QUARTER FROM o.purchase_time) AS purchase_quarter,
            EXTRACT(DOW FROM o.purchase_time) AS purchase_dayofweek,
            CASE WHEN EXTRACT(MONTH FROM o.purchase_time) IN (11, 12) THEN 1 ELSE 0 END AS is_peak_season,
            COALESCE(cg.geo_lat, -23.5505) AS cust_lat,
            COALESCE(cg.geo_lng, -46.6333) AS cust_lng,
            COALESCE(sg.geo_lat, -23.5505) AS seller_lat,
            COALESCE(sg.geo_lng, -46.6333) AS seller_lng
        FROM clean_items i
        JOIN clean_orders o ON i.order_id = o.order_id
        JOIN clean_products p ON i.product_id = p.product_id
        JOIN read_csv_auto('{customers_csv}') c ON o.customer_id = c.customer_id
        JOIN read_csv_auto('{sellers_csv}') s ON i.seller_id = s.seller_id
        LEFT JOIN geo_centroids cg ON c.customer_zip_code_prefix = cg.zip_prefix
        LEFT JOIN geo_centroids sg ON s.seller_zip_code_prefix = sg.zip_prefix
    )
    SELECT 
        order_id,
        order_item_id,
        product_id,
        seller_id,
        customer_id,
        price,
        freight_value,
        freight_ratio,
        product_category_name,
        product_weight_g,
        actual_weight_kg,
        volumetric_weight_kg,
        billable_weight_kg,
        CASE 
            WHEN billable_weight_kg < 1.0 THEN '<1kg'
            WHEN billable_weight_kg < 3.0 THEN '1-3kg'
            WHEN billable_weight_kg < 8.0 THEN '3-8kg'
            WHEN billable_weight_kg < 15.0 THEN '8-15kg'
            ELSE '15kg+' 
        END AS weight_band,
        product_length_cm,
        product_height_cm,
        product_width_cm,
        product_volume_cm3,
        seller_state,
        customer_state,
        is_same_state,
        route_lane,
        -- Haversine formula calculation in SQL: R = 6371 km
        ROUND(
            6371.0 * 2.0 * ASIN(
                SQRT(
                    SIN(RADIANS(cust_lat - seller_lat) / 2.0)^2 +
                    COS(RADIANS(seller_lat)) * COS(RADIANS(cust_lat)) *
                    SIN(RADIANS(cust_lng - seller_lng) / 2.0)^2
                )
            ), 2
        ) AS distance_km,
        CASE 
            WHEN (6371.0 * 2.0 * ASIN(SQRT(SIN(RADIANS(cust_lat - seller_lat) / 2.0)^2 + COS(RADIANS(seller_lat)) * COS(RADIANS(cust_lat)) * SIN(RADIANS(cust_lng - seller_lng) / 2.0)^2))) < 100.0 THEN 'Local (<100km)'
            WHEN (6371.0 * 2.0 * ASIN(SQRT(SIN(RADIANS(cust_lat - seller_lat) / 2.0)^2 + COS(RADIANS(seller_lat)) * COS(RADIANS(cust_lat)) * SIN(RADIANS(cust_lng - seller_lng) / 2.0)^2))) < 500.0 THEN 'Regional (100-500km)'
            WHEN (6371.0 * 2.0 * ASIN(SQRT(SIN(RADIANS(cust_lat - seller_lat) / 2.0)^2 + COS(RADIANS(seller_lat)) * COS(RADIANS(cust_lat)) * SIN(RADIANS(cust_lng - seller_lng) / 2.0)^2))) < 1500.0 THEN 'Long Haul (500-1500km)'
            ELSE 'Cross Country (>1500km)'
        END AS distance_band,
        estimated_delivery_days,
        purchase_year,
        purchase_month,
        purchase_quarter,
        purchase_dayofweek,
        is_peak_season
    FROM joined
    """
    
    # Save outputs using DuckDB native high-speed export (zero external dependencies)
    parquet_path = os.path.join(output_dir, "clean_freight_features.parquet").replace("\\", "/")
    csv_path = os.path.join(output_dir, "clean_freight_features.csv").replace("\\", "/")
    
    con.execute(f"COPY ({query}) TO '{parquet_path}' (FORMAT PARQUET);")
    con.execute(f"COPY ({query}) TO '{csv_path}' (FORMAT CSV, HEADER);")
    
    df_clean = con.execute(f"SELECT * FROM '{parquet_path}'").df()
    
    print(f"[DuckDB Pipeline] Successfully processed {len(df_clean)} freight records!")
    print(f"[DuckDB Pipeline] Saved to:")
    print(f"  - Parquet: {parquet_path}")
    print(f"  - CSV: {csv_path}")
    
    con.close()
    return df_clean

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    raw = os.path.join(base, "data", "raw")
    processed = os.path.join(base, "data", "processed")
    run_duckdb_pipeline(raw, processed)
