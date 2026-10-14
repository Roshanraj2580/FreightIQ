-- FreightIQ Analytics: Shipping Lane Profitability, SLA Speed, and Surcharges
-- Table: freightiq_analytics.clean_freight_features

-- Top Shipping Corridors Performance Breakdown
SELECT 
    route_lane,
    seller_state AS origin,
    customer_state AS destination,
    is_same_state,
    COUNT(*) AS total_shipments,
    ROUND(AVG(distance_km), 1) AS distance_km,
    ROUND(AVG(estimated_delivery_days), 1) AS avg_lead_time_days,
    ROUND(AVG(freight_value), 2) AS mean_freight_cost,
    -- Express Delivery Surcharge Estimation (Lead time <= 7 days vs standard)
    ROUND(AVG(CASE WHEN estimated_delivery_days <= 7 THEN freight_value ELSE NULL END), 2) AS express_freight_cost,
    ROUND(AVG(CASE WHEN estimated_delivery_days > 7 THEN freight_value ELSE NULL END), 2) AS standard_freight_cost,
    -- Express Premium Percentage
    ROUND(
        (AVG(CASE WHEN estimated_delivery_days <= 7 THEN freight_value ELSE NULL END) -
         AVG(CASE WHEN estimated_delivery_days > 7 THEN freight_value ELSE NULL END)) /
         NULLIF(AVG(CASE WHEN estimated_delivery_days > 7 THEN freight_value ELSE NULL END), 0) * 100.0, 
        2
    ) AS express_premium_pct
FROM `{project_id}.{dataset_id}.clean_freight_features`
GROUP BY 1, 2, 3, 4
HAVING COUNT(*) >= 25
ORDER BY total_shipments DESC;
