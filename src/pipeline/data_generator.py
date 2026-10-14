"""
FreightIQ Data Generator & Loader
Simulates or loads the Olist Brazilian E-Commerce dataset tables matching the exact
Kaggle schema and statistical distributions with full 100,000+ scale vectorized generation.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BRAZILIAN_STATES = {
    "SP": {"name": "São Paulo", "lat": -23.5505, "lng": -46.6333, "weight": 0.42},
    "RJ": {"name": "Rio de Janeiro", "lat": -22.9068, "lng": -43.1729, "weight": 0.14},
    "MG": {"name": "Minas Gerais", "lat": -19.9167, "lng": -43.9345, "weight": 0.12},
    "PR": {"name": "Paraná", "lat": -25.4284, "lng": -49.2733, "weight": 0.06},
    "RS": {"name": "Rio Grande do Sul", "lat": -30.0346, "lng": -51.2177, "weight": 0.06},
    "SC": {"name": "Santa Catarina", "lat": -27.5954, "lng": -48.5480, "weight": 0.04},
    "BA": {"name": "Bahia", "lat": -12.9714, "lng": -38.5014, "weight": 0.04},
    "DF": {"name": "Distrito Federal", "lat": -15.7975, "lng": -47.8919, "weight": 0.03},
    "GO": {"name": "Goiás", "lat": -16.6869, "lng": -49.2648, "weight": 0.02},
    "PE": {"name": "Pernambuco", "lat": -8.0476, "lng": -34.8770, "weight": 0.02},
    "CE": {"name": "Ceará", "lat": -3.7172, "lng": -38.5433, "weight": 0.02},
    "ES": {"name": "Espírito Santo", "lat": -20.3155, "lng": -40.3128, "weight": 0.02},
    "PA": {"name": "Pará", "lat": -1.4558, "lng": -48.5039, "weight": 0.01},
    "AM": {"name": "Amazonas", "lat": -3.1190, "lng": -60.0217, "weight": 0.01},
    "MT": {"name": "Mato Grosso", "lat": -15.6014, "lng": -56.0979, "weight": 0.01},
}

CATEGORIES = [
    "beleza_saude", "informatica_acessorios", "cama_mesa_banho",
    "moveis_decoracao", "esporte_lazer", "relogios_presentes",
    "telefonia", "automotivo", "brinquedos", "ferramentas_jardim",
    "eletronicos", "utilidades_domesticas", "bebes", "perfumaria"
]

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def generate_olist_dataset(output_dir: str, num_orders: int = 100000, seed: int = 42):
    """
    High-speed vectorized generator for 100,000+ Olist e-commerce orders matching Kaggle distributions.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating realistic Olist dataset with {num_orders:,} orders...")
    
    states = list(BRAZILIAN_STATES.keys())
    raw_probs = [BRAZILIAN_STATES[s]["weight"] for s in states]
    state_probs = np.array(raw_probs) / np.sum(raw_probs)
    
    # 1. Geolocation Table
    geo_rows = []
    valid_zips_by_state = {}
    zip_coords = {}
    for state, info in BRAZILIAN_STATES.items():
        base_zip = random.randint(1000, 99000)
        state_zips = []
        for i in range(40):
            zip_prefix = base_zip + i * 10
            lat_jitter = float(info["lat"] + np.random.normal(0, 0.15))
            lng_jitter = float(info["lng"] + np.random.normal(0, 0.15))
            geo_rows.append({
                "geolocation_zip_code_prefix": zip_prefix,
                "geolocation_lat": round(lat_jitter, 6),
                "geolocation_lng": round(lng_jitter, 6),
                "geolocation_city": info["name"],
                "geolocation_state": state
            })
            state_zips.append(zip_prefix)
            zip_coords[zip_prefix] = (lat_jitter, lng_jitter, state)
        valid_zips_by_state[state] = state_zips
        
    df_geo = pd.DataFrame(geo_rows)
    df_geo.to_csv(os.path.join(output_dir, "olist_geolocation_dataset.csv"), index=False)
    
    # 2. Customers Table
    num_customers = int(num_orders * 0.85)
    cust_states = np.random.choice(states, size=num_customers, p=state_probs)
    cust_zips = [random.choice(valid_zips_by_state[st]) for st in cust_states]
    
    df_customers = pd.DataFrame({
        "customer_id": [f"cust_{i:06d}" for i in range(num_customers)],
        "customer_unique_id": [f"uniq_{i:06d}" for i in range(num_customers)],
        "customer_zip_code_prefix": cust_zips,
        "customer_city": [BRAZILIAN_STATES[st]["name"] for st in cust_states],
        "customer_state": cust_states
    })
    df_customers.to_csv(os.path.join(output_dir, "olist_customers_dataset.csv"), index=False)
    
    # 3. Sellers Table
    num_sellers = 1500
    seller_state_weights = np.array([0.55 if s == "SP" else 0.12 if s in ["RJ", "MG", "PR"] else 0.01 for s in states])
    seller_state_weights /= seller_state_weights.sum()
    seller_states = np.random.choice(states, size=num_sellers, p=seller_state_weights)
    seller_zips = [random.choice(valid_zips_by_state[st]) for st in seller_states]
    
    df_sellers = pd.DataFrame({
        "seller_id": [f"sell_{i:04d}" for i in range(num_sellers)],
        "seller_zip_code_prefix": seller_zips,
        "seller_city": [BRAZILIAN_STATES[st]["name"] for st in seller_states],
        "seller_state": seller_states
    })
    df_sellers.to_csv(os.path.join(output_dir, "olist_sellers_dataset.csv"), index=False)
    
    # 4. Products Table
    num_products = 5000
    prod_weights = np.clip(np.random.lognormal(mean=7.0, sigma=1.1, size=num_products), 50, 30000).astype(int)
    prod_lengths = np.clip(np.random.normal(30, 15, size=num_products), 10, 105).astype(int)
    prod_heights = np.clip(np.random.normal(18, 12, size=num_products), 5, 80).astype(int)
    prod_widths = np.clip(np.random.normal(20, 10, size=num_products), 10, 80).astype(int)
    prod_cats = np.random.choice(CATEGORIES, size=num_products)
    
    df_products = pd.DataFrame({
        "product_id": [f"prod_{i:05d}" for i in range(num_products)],
        "product_category_name": prod_cats,
        "product_name_lenght": np.random.randint(20, 60, size=num_products),
        "product_description_lenght": np.random.randint(100, 2000, size=num_products),
        "product_photos_qty": np.random.randint(1, 6, size=num_products),
        "product_weight_g": prod_weights,
        "product_length_cm": prod_lengths,
        "product_height_cm": prod_heights,
        "product_width_cm": prod_widths
    })
    df_products.to_csv(os.path.join(output_dir, "olist_products_dataset.csv"), index=False)
    
    # 5. Vectorized Orders Generation
    base_time = pd.Timestamp("2017-01-01")
    random_days = np.random.randint(0, 600, size=num_orders)
    random_seconds = np.random.randint(0, 86400, size=num_orders)
    purchase_dates = base_time + pd.to_timedelta(random_days, unit="D") + pd.to_timedelta(random_seconds, unit="s")
    
    est_lead_days = np.random.randint(5, 30, size=num_orders)
    est_delivery_dates = purchase_dates + pd.to_timedelta(est_lead_days, unit="D")
    
    actual_lead_days = np.maximum(2, np.random.normal(est_lead_days - 2, 4)).astype(int)
    actual_delivery_dates = purchase_dates + pd.to_timedelta(actual_lead_days, unit="D")
    carrier_dates = purchase_dates + pd.to_timedelta(np.random.randint(1, 4, size=num_orders), unit="D")
    
    chosen_cust_indices = np.random.randint(0, num_customers, size=num_orders)
    order_ids = [f"ord_{i:06d}" for i in range(num_orders)]
    cust_id_col = [f"cust_{idx:06d}" for idx in chosen_cust_indices]
    
    df_orders = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": cust_id_col,
        "order_status": "delivered",
        "order_purchase_timestamp": purchase_dates.strftime("%Y-%m-%d %H:%M:%S"),
        "order_approved_at": (purchase_dates + pd.to_timedelta(np.random.randint(1, 12, size=num_orders), unit="h")).strftime("%Y-%m-%d %H:%M:%S"),
        "order_delivered_carrier_date": carrier_dates.strftime("%Y-%m-%d %H:%M:%S"),
        "order_delivered_customer_date": actual_delivery_dates.strftime("%Y-%m-%d %H:%M:%S"),
        "order_estimated_delivery_date": est_delivery_dates.strftime("%Y-%m-%d %H:%M:%S")
    })
    df_orders.to_csv(os.path.join(output_dir, "olist_orders_dataset.csv"), index=False)
    
    # 6. Vectorized Order Items Generation (~1.15 items per order -> ~115,000+ items)
    item_counts = np.random.choice([1, 2, 3], size=num_orders, p=[0.85, 0.12, 0.03])
    total_items = int(np.sum(item_counts))
    
    rep_order_ids = np.repeat(order_ids, item_counts)
    rep_cust_indices = np.repeat(chosen_cust_indices, item_counts)
    rep_est_days = np.repeat(est_lead_days, item_counts)
    rep_purchase_dates = np.repeat(purchase_dates, item_counts)
    
    # Sequence item id per order
    item_seq = []
    for c in item_counts:
        item_seq.extend(range(1, c + 1))
        
    chosen_prods = np.random.randint(0, num_products, size=total_items)
    chosen_sellers = np.random.randint(0, num_sellers, size=total_items)
    
    # Pre-calculated arrays for vectorization
    cust_zips_arr = np.array(cust_zips)[rep_cust_indices]
    seller_zips_arr = np.array(seller_zips)[chosen_sellers]
    
    # Look up coords
    c_lats = np.array([zip_coords[z][0] for z in cust_zips_arr])
    c_lngs = np.array([zip_coords[z][1] for z in cust_zips_arr])
    c_states = np.array([zip_coords[z][2] for z in cust_zips_arr])
    
    s_lats = np.array([zip_coords[z][0] for z in seller_zips_arr])
    s_lngs = np.array([zip_coords[z][1] for z in seller_zips_arr])
    s_states = np.array([zip_coords[z][2] for z in seller_zips_arr])
    
    # Vectorized Haversine
    dist_kms = haversine_np(c_lats, c_lngs, s_lats, s_lngs)
    
    p_weights = prod_weights[chosen_prods]
    p_lengths = prod_lengths[chosen_prods]
    p_heights = prod_heights[chosen_prods]
    p_widths = prod_widths[chosen_prods]
    
    weight_kg = p_weights / 1000.0
    vol_weight_kg = (p_lengths * p_heights * p_widths) / 6000.0
    billable_weights = np.maximum(weight_kg, vol_weight_kg)
    
    is_same = (c_states == s_states)
    tax_mult = np.where(is_same, 1.0, 1.35)
    
    months = rep_purchase_dates.month
    season_mult = np.where(np.isin(months, [11, 12]), 1.18, np.where(np.isin(months, [1, 2]), 0.95, 1.0))
    speed_mult = np.where(rep_est_days <= 7, 1.25, np.where(rep_est_days <= 15, 1.0, 0.90))
    
    base_cost = 8.50
    dist_cost = (dist_kms / 100.0) * 1.85
    weight_cost = billable_weights * 2.10
    
    noise = np.random.normal(0, 1.8, size=total_items)
    raw_freight = (base_cost + dist_cost + weight_cost) * tax_mult * season_mult * speed_mult + noise
    freight_values = np.maximum(6.50, np.round(raw_freight, 2))
    
    prices = np.round(np.random.lognormal(mean=4.2, sigma=0.8, size=total_items), 2)
    
    prod_ids = [f"prod_{p:05d}" for p in chosen_prods]
    sell_ids = [f"sell_{s:04d}" for s in chosen_sellers]
    ship_limits = (rep_purchase_dates + pd.to_timedelta(5, unit="D")).strftime("%Y-%m-%d %H:%M:%S")
    
    df_order_items = pd.DataFrame({
        "order_id": rep_order_ids,
        "order_item_id": item_seq,
        "product_id": prod_ids,
        "seller_id": sell_ids,
        "shipping_limit_date": ship_limits,
        "price": prices,
        "freight_value": freight_values
    })
    df_order_items.to_csv(os.path.join(output_dir, "olist_order_items_dataset.csv"), index=False)
    
    print(f"Data generation complete! Saved 6 tables to: {output_dir}")
    print(f"- Orders: {len(df_orders):,}")
    print(f"- Order items: {len(df_order_items):,}")
    print(f"- Products: {len(df_products):,}")
    print(f"- Customers: {len(df_customers):,}")
    print(f"- Sellers: {len(df_sellers):,}")
    print(f"- Geolocation centroids: {len(df_geo):,}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic Olist dataset for FreightIQ")
    parser.add_argument("--output-dir", default="data/raw", help="Target output directory")
    parser.add_argument("--num-orders", type=int, default=100000, help="Number of orders to generate (default 100,000)")
    args = parser.parse_args()
    
    generate_olist_dataset(args.output_dir, args.num_orders)
