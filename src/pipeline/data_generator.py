"""
FreightIQ Data Generator & Loader
Simulates or loads the Olist Brazilian E-Commerce dataset tables matching the exact
Kaggle schema and statistical distributions.
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
    """Calculate the great circle distance in kilometers between two points on earth."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def generate_olist_dataset(output_dir: str, num_orders: int = 15000, seed: int = 42):
    """
    Generates realistic synthetic Olist dataset tables matching Kaggle schema.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating synthetic Olist dataset with {num_orders} orders...")
    
    states = list(BRAZILIAN_STATES.keys())
    state_probs = [BRAZILIAN_STATES[s]["weight"] for s in states]
    
    # 1. Geolocation Table
    geo_rows = []
    for state, info in BRAZILIAN_STATES.items():
        base_zip = random.randint(1000, 99000)
        for i in range(25):
            zip_prefix = base_zip + i * 10
            lat_jitter = info["lat"] + np.random.normal(0, 0.15)
            lng_jitter = info["lng"] + np.random.normal(0, 0.15)
            geo_rows.append({
                "geolocation_zip_code_prefix": zip_prefix,
                "geolocation_lat": round(lat_jitter, 6),
                "geolocation_lng": round(lng_jitter, 6),
                "geolocation_city": info["name"],
                "geolocation_state": state
            })
    df_geo = pd.DataFrame(geo_rows)
    df_geo.to_csv(os.path.join(output_dir, "olist_geolocation_dataset.csv"), index=False)
    
    # Map zip code prefix to mean lat/lng
    zip_coords = df_geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean().to_dict(orient="index")
    valid_zips_by_state = df_geo.groupby("geolocation_state")["geolocation_zip_code_prefix"].unique().to_dict()
    
    # 2. Customers Table
    num_customers = int(num_orders * 0.85)
    cust_states = np.random.choice(states, size=num_customers, p=state_probs)
    customers = []
    for i in range(num_customers):
        st = cust_states[i]
        zip_prefix = int(np.random.choice(valid_zips_by_state[st]))
        customers.append({
            "customer_id": f"cust_{i:06d}",
            "customer_unique_id": f"uniq_{i:06d}",
            "customer_zip_code_prefix": zip_prefix,
            "customer_city": BRAZILIAN_STATES[st]["name"],
            "customer_state": st
        })
    df_customers = pd.DataFrame(customers)
    df_customers.to_csv(os.path.join(output_dir, "olist_customers_dataset.csv"), index=False)
    
    # 3. Sellers Table
    num_sellers = 600
    # Sellers skewed heavily towards SP, RJ, MG, PR
    seller_state_weights = [0.55 if s == "SP" else 0.12 if s in ["RJ", "MG", "PR"] else 0.01 for s in states]
    seller_state_weights = np.array(seller_state_weights) / sum(seller_state_weights)
    seller_states = np.random.choice(states, size=num_sellers, p=seller_state_weights)
    sellers = []
    for i in range(num_sellers):
        st = seller_states[i]
        zip_prefix = int(np.random.choice(valid_zips_by_state[st]))
        sellers.append({
            "seller_id": f"sell_{i:04d}",
            "seller_zip_code_prefix": zip_prefix,
            "seller_city": BRAZILIAN_STATES[st]["name"],
            "seller_state": st
        })
    df_sellers = pd.DataFrame(sellers)
    df_sellers.to_csv(os.path.join(output_dir, "olist_sellers_dataset.csv"), index=False)
    
    # 4. Products Table
    num_products = 3000
    products = []
    for i in range(num_products):
        cat = random.choice(CATEGORIES)
        # Weight distribution: lognormal with peak around 800g
        weight_g = int(np.clip(np.random.lognormal(mean=7.0, sigma=1.1), 50, 30000))
        length_cm = int(np.clip(np.random.normal(30, 15), 10, 105))
        height_cm = int(np.clip(np.random.normal(18, 12), 5, 80))
        width_cm = int(np.clip(np.random.normal(20, 10), 10, 80))
        products.append({
            "product_id": f"prod_{i:05d}",
            "product_category_name": cat,
            "product_name_lenght": random.randint(20, 60),
            "product_description_lenght": random.randint(100, 2000),
            "product_photos_qty": random.randint(1, 6),
            "product_weight_g": weight_g,
            "product_length_cm": length_cm,
            "product_height_cm": height_cm,
            "product_width_cm": width_cm
        })
    df_products = pd.DataFrame(products)
    df_products.to_csv(os.path.join(output_dir, "olist_products_dataset.csv"), index=False)
    
    # 5. Orders & Order Items Tables
    orders = []
    order_items = []
    
    start_date = datetime(2017, 1, 1)
    date_range_days = 600
    
    # Pre-index for quick lookup
    cust_dict = df_customers.set_index("customer_id").to_dict(orient="index")
    seller_dict = df_sellers.set_index("seller_id").to_dict(orient="index")
    prod_dict = df_products.set_index("product_id").to_dict(orient="index")
    
    cust_keys = list(cust_dict.keys())
    seller_keys = list(seller_dict.keys())
    prod_keys = list(prod_dict.keys())
    
    for i in range(num_orders):
        order_id = f"ord_{i:06d}"
        cust_id = random.choice(cust_keys)
        cust_info = cust_dict[cust_id]
        
        purchase_time = start_date + timedelta(
            days=random.randint(0, date_range_days),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Delivery speed requirement: estimated delivery date
        estimated_days = random.randint(5, 30)
        estimated_delivery = purchase_time + timedelta(days=estimated_days)
        actual_days = max(2, int(np.random.normal(estimated_days - 2, 4)))
        actual_delivery = purchase_time + timedelta(days=actual_days)
        
        orders.append({
            "order_id": order_id,
            "customer_id": cust_id,
            "order_status": "delivered",
            "order_purchase_timestamp": purchase_time.strftime("%Y-%m-%d %H:%M:%S"),
            "order_approved_at": (purchase_time + timedelta(hours=random.randint(1, 12))).strftime("%Y-%m-%d %H:%M:%S"),
            "order_delivered_carrier_date": (purchase_time + timedelta(days=random.randint(1, 3))).strftime("%Y-%m-%d %H:%M:%S"),
            "order_delivered_customer_date": actual_delivery.strftime("%Y-%m-%d %H:%M:%S"),
            "order_estimated_delivery_date": estimated_delivery.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 1-3 items per order
        num_items = np.random.choice([1, 2, 3], p=[0.85, 0.12, 0.03])
        for item_idx in range(1, num_items + 1):
            prod_id = random.choice(prod_keys)
            sell_id = random.choice(seller_keys)
            prod_info = prod_dict[prod_id]
            sell_info = seller_dict[sell_id]
            
            # Geographic distance
            c_zip = cust_info["customer_zip_code_prefix"]
            s_zip = sell_info["seller_zip_code_prefix"]
            c_coord = zip_coords.get(c_zip, {"geolocation_lat": -23.55, "geolocation_lng": -46.63})
            s_coord = zip_coords.get(s_zip, {"geolocation_lat": -23.55, "geolocation_lng": -46.63})
            
            dist_km = haversine_np(
                c_coord["geolocation_lat"], c_coord["geolocation_lng"],
                s_coord["geolocation_lat"], s_coord["geolocation_lng"]
            )
            
            # Brazilian Freight pricing model:
            # Base fee (~R$ 8.50) + weight rate + distance rate + volumetric surcharge + interstate tax
            weight_kg = prod_info["product_weight_g"] / 1000.0
            volume_cm3 = prod_info["product_length_cm"] * prod_info["product_height_cm"] * prod_info["product_width_cm"]
            volumetric_weight_kg = volume_cm3 / 6000.0
            billable_weight = max(weight_kg, volumetric_weight_kg)
            
            is_same_state = (cust_info["customer_state"] == sell_info["seller_state"])
            state_tax_multiplier = 1.0 if is_same_state else 1.35
            
            # Seasonal factor: November (Black Friday) / December surge
            month = purchase_time.month
            seasonal_factor = 1.18 if month in [11, 12] else (0.95 if month in [1, 2] else 1.0)
            
            # Delivery speed factor: tighter estimated window costs more
            speed_factor = 1.25 if estimated_days <= 7 else (1.0 if estimated_days <= 15 else 0.90)
            
            # Realistic Freight formula
            base_cost = 8.50
            dist_cost = (dist_km / 100.0) * 1.85
            weight_cost = billable_weight * 2.10
            
            raw_freight = (base_cost + dist_cost + weight_cost) * state_tax_multiplier * seasonal_factor * speed_factor
            noise = np.random.normal(0, 1.8)
            freight_value = max(6.50, round(raw_freight + noise, 2))
            
            item_price = round(float(np.random.lognormal(mean=4.2, sigma=0.8)), 2)
            
            order_items.append({
                "order_id": order_id,
                "order_item_id": item_idx,
                "product_id": prod_id,
                "seller_id": sell_id,
                "shipping_limit_date": (purchase_time + timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "price": item_price,
                "freight_value": freight_value
            })
            
    df_orders = pd.DataFrame(orders)
    df_orders.to_csv(os.path.join(output_dir, "olist_orders_dataset.csv"), index=False)
    
    df_order_items = pd.DataFrame(order_items)
    df_order_items.to_csv(os.path.join(output_dir, "olist_order_items_dataset.csv"), index=False)
    
    print(f"Data generation complete! Saved 6 tables to: {output_dir}")
    print(f"- Orders: {len(df_orders)}")
    print(f"- Order items: {len(df_order_items)}")
    print(f"- Products: {len(df_products)}")
    print(f"- Customers: {len(df_customers)}")
    print(f"- Sellers: {len(df_sellers)}")
    print(f"- Geolocation centroids: {len(df_geo)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic Olist dataset for FreightIQ")
    parser.add_argument("--output-dir", default="data/raw", help="Target output directory")
    parser.add_argument("--num-orders", type=int, default=15000, help="Number of orders to generate")
    args = parser.parse_args()
    
    generate_olist_dataset(args.output_dir, args.num_orders)
