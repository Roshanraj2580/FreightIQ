-- FreightIQ Analytics: Freight Trends by Region and Season (BigQuery SQL)
-- Table: freightiq_analytics.clean_freight_features

-- 1. Regional Freight Breakdown: Interstate vs Intrastate & Key Lanes
SELECT 
    seller_state,
    customer_state,
    route_lane,
    is_same_state,
    COUNT(order_id) AS total_shipments,
    ROUND(AVG(freight_value), 2) AS avg_freight_cost_brl,
    ROUND(MIN(freight_value), 2) AS min_freight_cost_brl,
    ROUND(MAX(freight_value), 2) AS max_freight_cost_brl,
    ROUND(AVG(distance_km), 1) AS avg_distance_km,
    ROUND(AVG(billable_weight_kg), 2) AS avg_billable_weight_kg,
    -- Normalized Unit Metric: Freight Cost per Ton-Kilometer
    ROUND(AVG(freight_value / NULLIF((billable_weight_kg * distance_km) / 1000.0, 0)), 4) AS cost_per_ton_km,
    -- Freight to Product Value Ratio (Logistics Burden)
    ROUND(AVG(freight_ratio) * 100.0, 2) AS avg_freight_burden_pct
FROM `{project_id}.{dataset_id}.clean_freight_features`
GROUP BY 1, 2, 3, 4
HAVING COUNT(order_id) >= 10
ORDER BY total_shipments DESC;

-- 2. Seasonality & Holiday Surge Analysis (Monthly & Peak Season Comparison)
SELECT 
    purchase_year,
    purchase_month,
    CASE purchase_month
        WHEN 1 THEN 'Jan' WHEN 2 THEN 'Feb' WHEN 3 THEN 'Mar'
        WHEN 4 THEN 'Apr' WHEN 5 THEN 'May' WHEN 6 THEN 'Jun'
        WHEN 7 THEN 'Jul' WHEN 8 THEN 'Aug' WHEN 9 THEN 'Sep'
        WHEN 10 THEN 'Oct' WHEN 11 THEN 'Nov (Black Friday)' WHEN 12 THEN 'Dec (Holidays)'
    END AS month_name,
    is_peak_season,
    COUNT(order_id) AS total_orders,
    ROUND(AVG(freight_value), 2) AS avg_freight_value,
    ROUND(AVG(estimated_delivery_days), 1) AS avg_delivery_promise_days,
    ROUND(AVG(price), 2) AS avg_order_price,
    ROUND(AVG(freight_ratio) * 100.0, 2) AS freight_to_price_ratio_pct
FROM `{project_id}.{dataset_id}.clean_freight_features`
GROUP BY 1, 2, 3, 4
ORDER BY purchase_year, purchase_month;

-- 3. Matrix: Weight Band vs Distance Tier Freight Index
SELECT 
    weight_band,
    distance_band,
    COUNT(*) AS volume,
    ROUND(AVG(freight_value), 2) AS baseline_cost,
    ROUND(STDDEV(freight_value), 2) AS cost_volatility,
    ROUND(PERCENTILE_CONT(freight_value, 0.50) OVER(PARTITION BY weight_band, distance_band), 2) AS median_cost,
    ROUND(PERCENTILE_CONT(freight_value, 0.90) OVER(PARTITION BY weight_band, distance_band), 2) AS p90_cost_ceiling
FROM `{project_id}.{dataset_id}.clean_freight_features`
ORDER BY weight_band, distance_band;
