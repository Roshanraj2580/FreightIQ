"""
FreightIQ - Stage 3: Machine Learning Training Pipeline
Trains Linear Regression baseline vs LightGBM Regressor.
Computes RMSE, MAE, R², MAPE, correlations, and feature importances.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

def compute_mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100.0)

def train_freight_models(data_path: str, output_model_dir: str):
    """
    Loads curated freight features, trains baseline vs LightGBM,
    evaluates statistical metrics and exports artifacts.
    """
    os.makedirs(output_model_dir, exist_ok=True)
    print(f"[Modeling] Loading training dataset from {data_path}...")
    
    if data_path.endswith(".parquet"):
        import duckdb
        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM '{data_path.replace(os.sep, '/')}'").df()
        con.close()
    else:
        df = pd.read_csv(data_path)
        
    print(f"[Modeling] Total dataset size: {len(df)} records.")
    
    # Define features
    numeric_features = [
        "distance_km",
        "billable_weight_kg",
        "actual_weight_kg",
        "volumetric_weight_kg",
        "product_volume_cm3",
        "estimated_delivery_days",
        "is_same_state",
        "is_peak_season",
        "purchase_month"
    ]
    
    categorical_features = [
        "seller_state",
        "customer_state",
        "weight_band",
        "distance_band"
    ]
    
    target = "freight_value"
    
    # Drop records with missing values in key columns
    feature_cols = numeric_features + categorical_features
    df_clean = df[feature_cols + [target]].dropna()
    
    # 1. Statistical Checks: Feature Correlations with Target
    numeric_corr = df_clean[numeric_features + [target]].corr()[target].sort_values(ascending=False).to_dict()
    print("\n[Modeling] Statistical Check - Feature Correlations with Target (freight_value):")
    for feat, corr in numeric_corr.items():
        if feat != target:
            print(f"  - {feat:25s}: {corr:+.4f}")
            
    # Train / Test split
    X = df_clean[feature_cols]
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[Modeling] Train set: {len(X_train)} | Test set: {len(X_test)}")
    
    # 2. Baseline Model: Linear Regression (Ridge)
    preprocessor_lr = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor_lr),
        ("regressor", Ridge(alpha=1.0))
    ])
    
    print("\n[Modeling] Training Linear Regression baseline...")
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    
    rmse_lr = float(np.sqrt(mean_squared_error(y_test, y_pred_lr)))
    mae_lr = float(mean_absolute_error(y_test, y_pred_lr))
    r2_lr = float(r2_score(y_test, y_pred_lr))
    mape_lr = compute_mape(y_test, y_pred_lr)
    
    print(f"  [Linear Baseline] RMSE: R$ {rmse_lr:.2f} | MAE: R$ {mae_lr:.2f} | R²: {r2_lr:.4f} | MAPE: {mape_lr:.2f}%")
    
    # 3. Production Model: LightGBM Regressor
    preprocessor_lgb = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    
    lgb_pipeline = Pipeline([
        ("preprocessor", preprocessor_lgb),
        ("regressor", lgb.LGBMRegressor(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("\n[Modeling] Training LightGBM Regressor...")
    lgb_pipeline.fit(X_train, y_train)
    y_pred_lgb = lgb_pipeline.predict(X_test)
    
    rmse_lgb = float(np.sqrt(mean_squared_error(y_test, y_pred_lgb)))
    mae_lgb = float(mean_absolute_error(y_test, y_pred_lgb))
    r2_lgb = float(r2_score(y_test, y_pred_lgb))
    mape_lgb = compute_mape(y_test, y_pred_lgb)
    
    print(f"  [LightGBM]        RMSE: R$ {rmse_lgb:.2f} | MAE: R$ {mae_lgb:.2f} | R²: {r2_lgb:.4f} | MAPE: {mape_lgb:.2f}%")
    
    # Calculate percentage improvement
    mae_improvement = ((mae_lr - mae_lgb) / mae_lr) * 100.0
    rmse_improvement = ((rmse_lr - rmse_lgb) / rmse_lr) * 100.0
    print(f"\n[Modeling] LightGBM Performance Advantage:")
    print(f"  - MAE Reduction:  {mae_improvement:.2f}% better than baseline")
    print(f"  - RMSE Reduction: {rmse_improvement:.2f}% better than baseline")
    
    # 4. Feature Importance Extraction (LightGBM)
    ohe_cat_names = lgb_pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(ohe_cat_names)
    importances = lgb_pipeline.named_steps["regressor"].feature_importances_
    
    # Aggregate importance by parent feature
    importance_dict = {}
    for name, imp in zip(all_feature_names, importances):
        parent_col = name.split("_")[0] if "_" in name else name
        # match against categorical or numeric names
        matched = False
        for c in categorical_features:
            if name.startswith(c):
                importance_dict[c] = importance_dict.get(c, 0) + int(imp)
                matched = True
                break
        if not matched:
            importance_dict[name] = importance_dict.get(name, 0) + int(imp)
            
    sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    print("\n[Modeling] Top Feature Importances (LightGBM Splits):")
    for feat, imp in sorted_importances[:10]:
        print(f"  - {feat:25s}: {imp}")
        
    # 5. Residual Diagnostics
    residuals = y_test - y_pred_lgb
    residual_stats = {
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "p90_abs_error": float(np.percentile(np.abs(residuals), 90)),
        "p95_abs_error": float(np.percentile(np.abs(residuals), 95)),
        "max_abs_error": float(np.max(np.abs(residuals)))
    }
    
    # 6. Save Model Artifacts & Metrics
    metrics_report = {
        "dataset": {
            "total_records": len(df_clean),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features_used": feature_cols
        },
        "models": {
            "linear_regression_baseline": {
                "rmse": round(rmse_lr, 3),
                "mae": round(mae_lr, 3),
                "r2": round(r2_lr, 4),
                "mape": round(mape_lr, 2)
            },
            "lightgbm_regressor": {
                "rmse": round(rmse_lgb, 3),
                "mae": round(mae_lgb, 3),
                "r2": round(r2_lgb, 4),
                "mape": round(mape_lgb, 2),
                "mae_improvement_pct": round(mae_improvement, 2),
                "rmse_improvement_pct": round(rmse_improvement, 2)
            }
        },
        "statistical_checks": {
            "correlations_with_target": {k: round(v, 4) for k, v in numeric_corr.items() if k != target},
            "feature_importance_ranking": sorted_importances,
            "residuals": residual_stats
        }
    }
    
    metrics_path = os.path.join(output_model_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
        
    lgb_model_path = os.path.join(output_model_dir, "freight_lgbm_pipeline.joblib")
    lr_model_path = os.path.join(output_model_dir, "freight_lr_pipeline.joblib")
    joblib.dump(lgb_pipeline, lgb_model_path)
    joblib.dump(lr_pipeline, lr_model_path)
    
    metadata = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "valid_states": sorted(list(df_clean["seller_state"].unique())),
        "weight_bands": ["<1kg", "1-3kg", "3-8kg", "8-15kg", "15kg+"],
        "distance_bands": ["Local (<100km)", "Regional (100-500km)", "Long Haul (500-1500km)", "Cross Country (>1500km)"]
    }
    with open(os.path.join(output_model_dir, "feature_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\n[Modeling] Training complete! Models and metrics saved to: {output_model_dir}")
    return metrics_report

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    curated_data = os.path.join(base, "data", "processed", "clean_freight_features.csv")
    models_dir = os.path.join(base, "models")
    train_freight_models(curated_data, models_dir)
