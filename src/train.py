#!/usr/bin/env python3
import os
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("Churn_Model_Training_Pipeline")

def build_preprocessor(X_train: pd.DataFrame):
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    return preprocessor, numerical_cols, categorical_cols

def run_model_training_pipeline(data_path: str, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    mlflow.set_experiment("Ecommerce_Churn_Prediction_Suite")
    
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Customer ID', 'Is_Churned'])
    y = df['Is_Churned']
    
    # Stratified split to enforce balanced evaluation distributions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    preprocessor, numerical_cols, categorical_cols = build_preprocessor(X_train)
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)
    
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_features = cat_encoder.get_feature_names_out(categorical_cols).tolist() if categorical_cols else []
    all_features = numerical_cols + encoded_cat_features
    
    X_train_df = pd.DataFrame(X_train_arr, columns=all_features)
    X_test_df = pd.DataFrame(X_test_arr, columns=all_features)
    
    # Explicit feature drop step for Logistic Regression to cure severe multicollinearity
    linear_drop_cols = ['Lifespan_Consistency', 'Recency', 'LifetimeSpend']
    X_train_linear = X_train_df.drop(columns=linear_drop_cols, errors='ignore')
    X_test_linear = X_test_df.drop(columns=linear_drop_cols, errors='ignore')
    
    scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    model_configs = {
        "Logistic_Regression": {
            "model": LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42),
            "data": (X_train_linear, X_test_linear),
            "params": {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"]}
        },
        "Random_Forest": {
            "model": RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
            "data": (X_train_df, X_test_df),
            "params": {"n_estimators": [100, 200, 300], "max_depth": [8, 12, 16]}
        },
        "Gradient_Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "data": (X_train_df, X_test_df),
            "params": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [4, 6]}
        },
        "XGBoost": {
            "model": XGBClassifier(scale_pos_weight=scale_weight, eval_metric='logloss', random_state=42, n_jobs=-1),
            "data": (X_train_df, X_test_df),
            "params": {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [4, 6]}
        }
    }
    
    for model_name, config in model_configs.items():
        X_tr, X_te = config["data"]
        
        with mlflow.start_run(run_name=model_name):
            logger.info(f"Optimizing hyperparameter grids for: {model_name}")
            
            grid_search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_tr, y_train)
            best_model = grid_search.best_estimator_
            
            preds = best_model.predict(X_te)
            probs = best_model.predict_proba(X_te)[:, 1]
            
            auc_score = roc_auc_score(y_test, probs)
            report = classification_report(y_test, preds, output_dict=True)
            
            # Log metrics out to MLflow server tracking dashboards
            for p_name, p_val in grid_search.best_params_.items():
                mlflow.log_param(f"best_{p_name}", p_val)
                
            mlflow.log_metric("Test_ROC_AUC", auc_score)
            mlflow.log_metric("Churn_Recall", report['1']['recall'])
            mlflow.log_metric("Churn_F1_Score", report['1']['f1-score'])
            
            if "XGBoost" in model_name:
                mlflow.xgboost.log_model(best_model, name="model")
            else:
                mlflow.sklearn.log_model(best_model, name="model")
                
            # Log Evaluation ROC Curve Image plots as binary run artifacts
            plt.figure(figsize=(6, 5))
            fpr, tpr, _ = roc_curve(y_test, probs)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', color='darkorange', lw=2)
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            
            plot_file = os.path.join(plots_dir, f"roc_curve_{model_name}.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(plot_file)
            
    logger.info(" Core cross-validated training suite complete.")