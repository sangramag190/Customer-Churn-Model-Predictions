#!/usr/bin/env python3
import os
import sys
import pandas as pd
import mlflow
import numpy as np
from src.utils.logger import get_production_logger
from src.product_clustering import run_product_segmentation

logger = get_production_logger(__name__)

def run_data_preparation(input_path: str, output_path: str, models_dir: str):
    mlflow.set_experiment("Ecommerce_Retail_Data_Prep")
    
    with mlflow.start_run(run_name="Data_Cleaning_and_Segmentation"):
        try:
            logger.info(f"Importing multi-year workbook from: {input_path}")

            # 1. Read all worksheets dynamically into a dictionary of dataframes
            excel_dict = pd.read_excel(input_path, sheet_name=None)
            logger.info(f"Found sheets in workbook: {list(excel_dict.keys())}")

            # 2. Concat all sheets together into a single global transaction log
            df = pd.concat(excel_dict.values(), ignore_index=True)
            
            initial_row_count = len(df)
            mlflow.log_metric("raw_rows_count", initial_row_count)
            
            # --- CLEANING LAYER ---
            df = df.dropna(subset=["Customer ID"]).reset_index(drop=True)
            df = df[df["Customer ID"].astype(str) != 'NaN']
            df["Description"] = df["Description"].fillna("").replace(['NaN', 'nan'], "")
            df['StockCode'] = df['StockCode'].astype(str).str.strip()
            
            admin_noise = ['POST', 'C2', 'M', 'BANK CHARGES', 'TEST001', 'TEST002', 'PADS', 'ADJUST', 'D', 'ADJUST2', 'SP1002']
            df = df[~df['StockCode'].isin(admin_noise)]
            df = df.drop_duplicates()
            logger.info(f"Succefully Removed Admin Noises")
            
            # --- VECTORIZED PRICE IMPUTATION LAYER ---
            mean_prices = df[df['Price'] > 0].groupby('StockCode')['Price'].mean()
            zero_price_mask = (df['Price'] == 0) | (df['Price'].isna())
            df.loc[zero_price_mask, 'Price'] = df.loc[zero_price_mask, 'StockCode'].map(mean_prices)
            
            df = df[df['Price'] > 0].copy()
            df['monetary_value'] = df['Quantity'] * df['Price']
            
            ## Removing top and bottom two outliers:
            top_2_price_idx = df[df['Price']>25000].index
            top_2_qty_idx = df[df['Quantity']>50000].index
            #return invoices:
            ret_top_2_price_idx = df[df['Price']<-25000].index
            ret_top_2_qty_idx = df[df['Quantity']<-50000].index

            df = df.drop(index=top_2_price_idx.union(top_2_qty_idx))
            df = df.drop(index=ret_top_2_price_idx.union(ret_top_2_qty_idx)).reset_index(drop=True)
            logger.info(f"Dataset formated for removing extreme outliers")
            
            
            # --- MODEL EXTRACTION STEP (Isolated Customer-Product Pivot Filter) ---
            invoice_sales = df
            # 1. Calculate net totals at the unique Customer ID + StockCode pivot level
            net_pivot_totals = invoice_sales.groupby(['Customer ID', 'StockCode'])['Quantity'].sum().reset_index()
            # 2. Identifing the specific invalid combinations (Net quantity <= 0)
            invalid_pivots = net_pivot_totals[net_pivot_totals['Quantity'] <= 0].copy()
            # 3. Creating a unique composite tracking string for the invalid combinations
            invalid_pivots['pivot_key'] = invalid_pivots['Customer ID'].astype(str) + "_" + invalid_pivots['StockCode'].astype(str)
            # 4. Creating the same composite tracking key on primary invoice_sales DataFrame
            invoice_sales['pivot_key'] = invoice_sales['Customer ID'].astype(str) + "_" + invoice_sales['StockCode'].astype(str)
            # 5. Dropping only the rows matching those specific invalid keys
            invoice_sales = invoice_sales[~invoice_sales['pivot_key'].isin(invalid_pivots['pivot_key'])].copy()
            # 6. Dropping the temporary composite key column to keep final matrix clean
            invoice_sales = invoice_sales.drop(columns=['pivot_key']).reset_index()

            # Run product segmentation on the safely isolated matrix
            cluster_lookup = run_product_segmentation(invoice_sales, models_dir)
            
            # Append generated classifications back to original tracking matrix
            df = pd.merge(df, cluster_lookup, on='StockCode', how='left')
            df['Cluster'] = df['Cluster'].fillna(-1).astype(int)
            
            # Export Final Training File
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            
            final_row_count = len(df)
            mlflow.log_metric("cleaned_rows_count", final_row_count)
            mlflow.log_metric("rows_removed", initial_row_count - final_row_count)
            logger.info(f"Execution complete. Labeled output ready at: {output_path}")
            
        except ValueError as ve:
            logger.error(f"Data value validation error: {ve}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Uncaught pipeline crash error: {e}")
            sys.exit(2)
  

def create_customer_analytical_dataset(transactions_path: str, output_path: str):
    logger.info("Generating customer-level analytical profile with leak-free snapshot windowing...")
    df = pd.read_csv(transactions_path)
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # 1. LEAK-FREE TIMELINE TIMING (Maximum 90th Percentile window is 270 days for Cluster 2)
    global_max_date = df['InvoiceDate'].max()
    snapshot_date = global_max_date - pd.Timedelta(days=269)
    
    logger.info(f"Global Timeline Boundary End: {global_max_date.strftime('%Y-%m-%d')}")
    logger.info(f"Train/Feature Generation Snapshot Anchor: {snapshot_date.strftime('%Y-%m-%d')}")
    
    # Segment data cleanly across the timeline anchor
    feature_history = df[df['InvoiceDate'] <= snapshot_date].copy()
    label_history = df[df['InvoiceDate'] > snapshot_date].copy()
    
    # 2. SLICE A: Positive sales metrics (Calculated strictly BEFORE snapshot)
    positive_invoices = feature_history[feature_history['Quantity'] > 0]
    rf_positive = positive_invoices.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        FirstInvoiceDate=('InvoiceDate', 'min'),
        LastInvoiceDate=('InvoiceDate', 'max')
    )

    # 3. SLICE B: Return-Only metrics (Calculated strictly BEFORE snapshot)
    return_invoices = feature_history[feature_history['Quantity'] < 0].copy()
    return_invoices['Return_Value_Absolute'] = return_invoices['monetary_value'] * -1

    rf_returns = return_invoices.groupby('Customer ID').agg(
        Return_Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Return_Frequency=('Invoice', 'nunique'),
        TotalReturnValue=('Return_Value_Absolute', 'sum'),
        AvgReturnValue=('Return_Value_Absolute', 'mean'),
        TotalReturnLineItems=('StockCode', 'count'),
        AvgReturnQuantityPerLine=('Quantity', lambda x: x.mean() * -1)
    )

    # 4. SLICE C: Overall metrics (Calculated strictly BEFORE snapshot)
    overall_metrics = feature_history.groupby('Customer ID').agg(
        LifetimeSpend=('monetary_value', 'sum'),
        AvgQuantityPerLine=('Quantity', 'mean'),
        TotalLineItems=('StockCode', 'count'),
        Country=('Country', lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    )

    # 5. Merging historical metrics together
    customer_matrix = overall_metrics.join(rf_positive, how='inner').join(rf_returns, how='left')

    # 6. FEATURE ENGINEERING LAYER (Calculated strictly BEFORE snapshot)
    customer_matrix['CustomerTenureDays'] = (snapshot_date - customer_matrix['FirstInvoiceDate']).dt.days
    customer_matrix['AOV'] = customer_matrix['LifetimeSpend'] / customer_matrix['Frequency']
    customer_matrix['PurchaseVelocity'] = customer_matrix['Frequency'] / customer_matrix['CustomerTenureDays'].clip(lower=1)
    customer_matrix['Return_Value_Ratio'] = customer_matrix['TotalReturnValue'].fillna(0) / customer_matrix['LifetimeSpend'].clip(lower=1)

    # Standardizing structural null values for clean model parsing
    customer_matrix['Frequency'] = customer_matrix['Frequency'].fillna(0).astype(int)
    customer_matrix['Recency'] = customer_matrix['Recency'].fillna(270).astype(int)
    customer_matrix['CustomerTenureDays'] = customer_matrix['CustomerTenureDays'].fillna(0).astype(int)
    customer_matrix['Return_Frequency'] = customer_matrix['Return_Frequency'].fillna(0).astype(int)
    customer_matrix['TotalReturnValue'] = customer_matrix['TotalReturnValue'].fillna(0.0)
    customer_matrix['AvgReturnValue'] = customer_matrix['AvgReturnValue'].fillna(0.0)
    customer_matrix['TotalReturnLineItems'] = customer_matrix['TotalReturnLineItems'].fillna(0).astype(int)
    customer_matrix['AvgReturnQuantityPerLine'] = customer_matrix['AvgReturnQuantityPerLine'].fillna(0.0)
    customer_matrix['Return_Value_Ratio'] = customer_matrix['Return_Value_Ratio'].fillna(0.0)
    customer_matrix['Return_Recency'] = customer_matrix['Return_Recency'].fillna(270).astype(int)

    customer_analytical_dataset = customer_matrix.reset_index()
    
    # 7. Pivot K-Means cluster spend shares
    cluster_spend = feature_history.groupby(['Customer ID', 'Cluster'])['monetary_value'].sum().unstack(fill_value=0)
    cluster_spend.columns = [
        f'Spend_Cluster_{int(col)}' if col != -1 else 'Spend_Cluster_Unclassified' for col in cluster_spend.columns
    ]

    # Explicit multi-index join map
    customer_analytical_dataset = customer_analytical_dataset.set_index('Customer ID').join(cluster_spend, how='left').reset_index()

    for col in cluster_spend.columns:
        customer_analytical_dataset[col] = customer_analytical_dataset[col].fillna(0.0)

    # 8. ADDING THE REQUESTED ADVANCED BEHAVIORAL INFLECTION FEATURES
    logger.info("Computing advanced cluster inflection multipliers...")
    
    # Extract structural column names for fallback protection
    active_cluster_cols = [c for c in cluster_spend.columns if 'Spend_Cluster_' in c and 'Unclassified' not in c]
    
    # Compute relative cluster median delta proxies
    # Medians derived via interval research steps: C0=26.9, C1=32.0, C2=48.9
    if len(active_cluster_cols) > 0:
        dominant_cluster_series = customer_analytical_dataset[active_cluster_cols].idxmax(axis=1)
        medians_map = dominant_cluster_series.map({
            'Spend_Cluster_0': 26.9, 'Spend_Cluster_1': 32.0, 'Spend_Cluster_2': 48.9
        }).fillna(32.0)
    else:
        medians_map = 32.0

    customer_analytical_dataset['Cluster_Median_Delta'] = customer_analytical_dataset['Recency'] / medians_map
    customer_analytical_dataset['Financial_Momentum'] = customer_analytical_dataset['LifetimeSpend'] / customer_analytical_dataset['Recency'].clip(lower=1)
    customer_analytical_dataset['Lifespan_Consistency'] = customer_analytical_dataset['Frequency'] * customer_analytical_dataset['CustomerTenureDays']

    # 9. STRATIFIED TARGET LABELING (Looking forward into post-snapshot label_history)
    logger.info("Executing 90th percentile cluster-stratified target window evaluations...")
    customer_analytical_dataset['Is_Churned'] = 0
    
    for idx, row in customer_analytical_dataset.iterrows():
        cid = row['Customer ID']
        
        # Pull dynamic cluster classification
        if len(active_cluster_cols) > 0:
            dom_cluster = row[active_cluster_cols].idxmax()
        else:
            dom_cluster = 'Spend_Cluster_0'
            
        # Target assignment rules governed explicitly by 90th percentile values:
        if dom_cluster == 'Spend_Cluster_0':
            # Cluster 0 Window: 140 Days
            limit_date = snapshot_date + pd.Timedelta(days=139)
            future_buys = label_history[(label_history['Customer ID'] == cid) & 
                                        (label_history['InvoiceDate'] <= limit_date) & 
                                        (label_history['Quantity'] > 0)]
            if future_buys.empty:
                customer_analytical_dataset.at[idx, 'Is_Churned'] = 1
                
        elif dom_cluster == 'Spend_Cluster_1':
            # Cluster 1 Window: 160 Days
            limit_date = snapshot_date + pd.Timedelta(days=158)
            future_buys = label_history[(label_history['Customer ID'] == cid) & 
                                        (label_history['InvoiceDate'] <= limit_date) & 
                                        (label_history['Quantity'] > 0)]
            if future_buys.empty:
                customer_analytical_dataset.at[idx, 'Is_Churned'] = 1
                
        elif dom_cluster == 'Spend_Cluster_2':
            # Cluster 2 Window: 270 Days
            limit_date = snapshot_date + pd.Timedelta(days=269)
            future_buys = label_history[(label_history['Customer ID'] == cid) & 
                                        (label_history['InvoiceDate'] <= limit_date) & 
                                        (label_history['Quantity'] > 0)]
            if future_buys.empty:
                customer_analytical_dataset.at[idx, 'Is_Churned'] = 1
        else:
            # Baseline fallback window
            limit_date = snapshot_date + pd.Timedelta(days=160)
            future_buys = label_history[(label_history['Customer ID'] == cid) & 
                                        (label_history['InvoiceDate'] <= limit_date) & 
                                        (label_history['Quantity'] > 0)]
            if future_buys.empty:
                customer_analytical_dataset.at[idx, 'Is_Churned'] = 1

    # Log metrics summary
    final_churn_rate = customer_analytical_dataset['Is_Churned'].mean() * 100
    logger.info(f"Target Labeling Complete. Active Matrix Shape: {customer_analytical_dataset.shape}")
    logger.info(f"Resulting Dataset Imbalance Churn Rate: {final_churn_rate:.2f}%")
    
    # Strip time markers to prevent tree algorithm target shortcuts
    drop_pillars = ['FirstInvoiceDate', 'LastInvoiceDate']
    customer_analytical_dataset = customer_analytical_dataset.drop(columns=[c for c in drop_pillars if c in customer_analytical_dataset.columns])
    
    # Exporting finalized analytical matrices
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    customer_analytical_dataset.to_csv(output_path, index=False)
    logger.info(f"Customer Analytical Dataset generated successfully at: {output_path}")

