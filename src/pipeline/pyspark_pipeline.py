"""
FreightIQ - Stage 1: Distributed PySpark Data Pipeline
Loads raw Olist Brazilian E-Commerce tables, cleans data, performs distributed joins,
and engineers geospatial, dimensional, and temporal freight features.
"""

import os
import sys

def build_spark_pipeline(raw_data_dir: str, output_path: str):
    """
    Executes PySpark pipeline for Olist freight feature engineering.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import DoubleType, IntegerType, StringType
    except ImportError:
        print("[PySpark Pipeline] PySpark is not installed in the active environment.")
        print("[PySpark Pipeline] Use duckdb_pipeline.py for local execution or install pyspark.")
        return False

    print("[PySpark Pipeline] Initializing SparkSession...")
    spark = SparkSession.builder \
        .appName("FreightIQ-Olist-ETL") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    # 1. Read Raw CSV Tables
    print(f"[PySpark Pipeline] Loading tables from {raw_data_dir}...")
    orders_df = spark.read.csv(os.path.join(raw_data_dir, "olist_orders_dataset.csv"), header=True, inferSchema=True)
    order_items_df = spark.read.csv(os.path.join(raw_data_dir, "olist_order_items_dataset.csv"), header=True, inferSchema=True)
    products_df = spark.read.csv(os.path.join(raw_data_dir, "olist_products_dataset.csv"), header=True, inferSchema=True)
    customers_df = spark.read.csv(os.path.join(raw_data_dir, "olist_customers_dataset.csv"), header=True, inferSchema=True)
    sellers_df = spark.read.csv(os.path.join(raw_data_dir, "olist_sellers_dataset.csv"), header=True, inferSchema=True)
    geo_df = spark.read.csv(os.path.join(raw_data_dir, "olist_geolocation_dataset.csv"), header=True, inferSchema=True)

    # 2. Data Cleaning & Aggregations
    # Filter delivered orders with valid timestamps
    clean_orders = orders_df.filter(F.col("order_status") == "delivered") \
        .withColumn("purchase_time", F.to_timestamp("order_purchase_timestamp")) \
        .withColumn("est_delivery_time", F.to_timestamp("order_estimated_delivery_date")) \
        .filter(F.col("purchase_time").isNotNull() & F.col("est_delivery_time").isNotNull()) \
        .withColumn("estimated_delivery_days", 
                    F.round((F.unix_timestamp("est_delivery_time") - F.unix_timestamp("purchase_time")) / 86400.0, 1)) \
        .filter(F.col("estimated_delivery_days") > 0)

    # Aggregate geolocation to prefix level (mean lat & lng) to prevent duplication
    geo_centroids = geo_df.groupBy("geolocation_zip_code_prefix").agg(
        F.avg("geolocation_lat").alias("geo_lat"),
        F.avg("geolocation_lng").alias("geo_lng"),
        F.first("geolocation_state").alias("geo_state")
    )

    # Clean products: handle missing dimensions and weights
    clean_products = products_df.na.fill({
        "product_weight_g": 500.0,
        "product_length_cm": 20.0,
        "product_height_cm": 15.0,
        "product_width_cm": 15.0,
        "product_category_name": "outros"
    }).withColumn("product_volume_cm3", 
                  F.col("product_length_cm") * F.col("product_height_cm") * F.col("product_width_cm")) \
      .withColumn("volumetric_weight_kg", F.round(F.col("product_volume_cm3") / 6000.0, 3)) \
      .withColumn("actual_weight_kg", F.round(F.col("product_weight_g") / 1000.0, 3)) \
      .withColumn("billable_weight_kg", F.greatest(F.col("actual_weight_kg"), F.col("volumetric_weight_kg")))

    # Clean order items: positive freight and price
    clean_items = order_items_df.filter((F.col("freight_value") > 0) & (F.col("price") > 0))

    # 3. Distributed Joins
    # Join items with orders, products, customers, sellers, and geolocation centroids
    joined_df = clean_items.join(clean_orders, on="order_id", how="inner") \
        .join(clean_products, on="product_id", how="inner") \
        .join(customers_df.select(
            F.col("customer_id"),
            F.col("customer_zip_code_prefix"),
            F.col("customer_state"),
            F.col("customer_city")
        ), on="customer_id", how="inner") \
        .join(sellers_df.select(
            F.col("seller_id"),
            F.col("seller_zip_code_prefix"),
            F.col("seller_state"),
            F.col("seller_city")
        ), on="seller_id", how="inner")

    # Join Geolocation for Customer and Seller
    customer_geo = geo_centroids.select(
        F.col("geolocation_zip_code_prefix").alias("customer_zip_code_prefix"),
        F.col("geo_lat").alias("cust_lat"),
        F.col("geo_lng").alias("cust_lng")
    )
    seller_geo = geo_centroids.select(
        F.col("geolocation_zip_code_prefix").alias("seller_zip_code_prefix"),
        F.col("geo_lat").alias("seller_lat"),
        F.col("geo_lng").alias("seller_lng")
    )

    enriched_df = joined_df.join(customer_geo, on="customer_zip_code_prefix", how="left") \
                           .join(seller_geo, on="seller_zip_code_prefix", how="left") \
                           .na.fill({"cust_lat": -23.5505, "cust_lng": -46.6333, "seller_lat": -23.5505, "seller_lng": -46.6333})

    # 4. Feature Engineering
    # A. Native Vectorized Haversine distance calculation in Spark SQL
    # earth radius = 6371 km
    lat1_rad = F.radians(F.col("cust_lat"))
    lat2_rad = F.radians(F.col("seller_lat"))
    dlat_rad = F.radians(F.col("seller_lat") - F.col("cust_lat"))
    dlng_rad = F.radians(F.col("seller_lng") - F.col("cust_lng"))

    haversine_a = F.sin(dlat_rad / 2.0)**2 + F.cos(lat1_rad) * F.cos(lat2_rad) * F.sin(dlng_rad / 2.0)**2
    haversine_c = 2.0 * F.atan2(F.sqrt(haversine_a), F.sqrt(1.0 - haversine_a))
    distance_km = F.round(6371.0 * haversine_c, 2)

    features_df = enriched_df.withColumn("distance_km", distance_km) \
        .withColumn("is_same_state", F.when(F.col("customer_state") == F.col("seller_state"), 1).otherwise(0)) \
        .withColumn("route_lane", F.concat(F.col("seller_state"), F.lit("->"), F.col("customer_state"))) \
        .withColumn("weight_band", F.when(F.col("billable_weight_kg") < 1.0, "<1kg")
                    .when(F.col("billable_weight_kg") < 3.0, "1-3kg")
                    .when(F.col("billable_weight_kg") < 8.0, "3-8kg")
                    .when(F.col("billable_weight_kg") < 15.0, "8-15kg")
                    .otherwise("15kg+")) \
        .withColumn("distance_band", F.when(F.col("distance_km") < 100.0, "Local (<100km)")
                    .when(F.col("distance_km") < 500.0, "Regional (100-500km)")
                    .when(F.col("distance_km") < 1500.0, "Long Haul (500-1500km)")
                    .otherwise("Cross Country (>1500km)")) \
        .withColumn("purchase_year", F.year("purchase_time")) \
        .withColumn("purchase_month", F.month("purchase_time")) \
        .withColumn("purchase_quarter", F.quarter("purchase_time")) \
        .withColumn("purchase_dayofweek", F.dayofweek("purchase_time")) \
        .withColumn("is_peak_season", F.when(F.col("purchase_month").isin([11, 12]), 1).otherwise(0)) \
        .withColumn("freight_ratio", F.round(F.col("freight_value") / F.col("price"), 4))

    # Select clean analytical schema
    final_df = features_df.select(
        "order_id", "order_item_id", "product_id", "seller_id", "customer_id",
        "price", "freight_value", "freight_ratio",
        "product_category_name", "product_weight_g", "actual_weight_kg",
        "volumetric_weight_kg", "billable_weight_kg", "weight_band",
        "product_length_cm", "product_height_cm", "product_width_cm", "product_volume_cm3",
        "seller_state", "customer_state", "is_same_state", "route_lane",
        "distance_km", "distance_band", "estimated_delivery_days",
        "purchase_year", "purchase_month", "purchase_quarter", "purchase_dayofweek", "is_peak_season"
    )

    print(f"[PySpark Pipeline] Writing cleaned parquet dataset to {output_path}...")
    final_df.write.mode("overwrite").partitionBy("purchase_year", "purchase_month").parquet(output_path)
    print(f"[PySpark Pipeline] PySpark ETL completed successfully! Processed {final_df.count()} records.")
    spark.stop()
    return True

if __name__ == "__main__":
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "spark_warehouse")
    build_spark_pipeline(raw_dir, out_dir)
