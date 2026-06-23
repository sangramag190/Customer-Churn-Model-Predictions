import os
import joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.utils.logger import get_production_logger

logger = get_production_logger(__name__)

def run_product_segmentation(invoice_sales: pd.DataFrame, models_dir: str) -> pd.DataFrame:
    """Aggregates product behavior, executes log-transform grid search, and saves model states."""
    logger.info("Starting product profile clustering aggregations...")
    
    product_profiles = invoice_sales.groupby('StockCode').agg(
        TotalQuantity=('Quantity', 'sum'),
        AvgPrice=('Price', 'mean'),
        TotalMonetary=('monetary_value', 'sum')
    ).reset_index()
    
    # Boundary validation using standard exceptions
    if product_profiles.empty:
        raise ValueError("Aggregated product data matrix contains zero rows.")
        
    product_profiles = product_profiles[(product_profiles['TotalQuantity'] > 0) & (product_profiles['AvgPrice'] > 0)].copy()
    if product_profiles.empty:
        raise ValueError("No items contain strictly positive quantity/price attributes. Log scaling failed.")
        
    # Log Transformations
    product_profiles['log_qty'] = np.log1p(product_profiles['TotalQuantity'])
    product_profiles['log_price'] = np.log1p(product_profiles['AvgPrice'])
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(product_profiles[['log_qty', 'log_price']])
    
    # Hyperparameter Evaluation Loop
    best_k = 4
    best_score = -1
    best_km_model = None
    
    for k_candidate in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k_candidate, init='k-means++', max_iter=300, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels)
        
        if sil_score > best_score:
            best_score = sil_score
            best_k = k_candidate
            best_km_model = km
            
    final_labels = best_km_model.labels_
    db_index = davies_bouldin_score(X_scaled, final_labels)
    
    logger.info(f"Optimal Cluster Match Decided: K={best_k} | Silhouette: {best_score:.4f}")
    
    # Track metrics inside MLflow
    mlflow.log_param("optimal_k", best_k)
    mlflow.log_metric("silhouette_score", best_score)
    mlflow.log_metric("davies_bouldin_index", db_index)
    
    # Model Serialization Artifacts
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "product_scaler.joblib"))
    joblib.dump(best_km_model, os.path.join(models_dir, "product_kmeans.joblib"))
    logger.info("StandardScaler and KMeans artifacts successfully written to storage.")
    
    product_profiles['Cluster'] = final_labels
    return product_profiles[['StockCode', 'Cluster']]