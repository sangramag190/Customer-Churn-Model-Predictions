
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Enforce headless rendering to insulate parallel worker loops
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Configure clean structured logging matrix
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (TRAIN_PIPELINE) : %(message)s')
logger = logging.getLogger(__name__)

def interpret_logistic_coefficients(log_reg_model, feature_names, output_dir):
    """Extracts, sorts, and translates linear regression weights into Odds Ratios."""
    coefficients = log_reg_model.coef_[0]
    if len(coefficients) != len(feature_names):
        feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
        
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient (Beta)': coefficients,
        'Odds Ratio (e^Beta)': np.exp(coefficients)
    })
    coef_df['Abs_Impact'] = coef_df['Coefficient (Beta)'].abs()
    coef_df = coef_df.sort_values(by='Abs_Impact', ascending=False).drop(columns=['Abs_Impact'])
    
    csv_path = os.path.join(output_dir, "logistic_cluster_feature_impacts.csv")
    coef_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)
    
    print("\n=== LOGISTIC REGRESSION VARIABLE LEVEL IMPACT (ODDS RATIOS) ===")
    print(coef_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return coef_df

def interpret_xgboost_importance(xgb_model, feature_names, output_dir):
    """Extracts and ranks features based on their Gain importance within the tree structures."""
    importances = xgb_model.feature_importances_
    if len(importances) != len(feature_names):
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
    xgb_df = pd.DataFrame({
        'Feature': feature_names,
        'Gain Importance': importances
    }).sort_values(by='Gain Importance', ascending=False)
    
    csv_path = os.path.join(output_dir, "xgboost_cluster_feature_gains.csv")
    xgb_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)
    
    print("\n=== XGBOOST ENSEMBLE VARIABLE LEVEL IMPACT (GAIN) ===")
    print(xgb_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return xgb_df

def run_cluster_based_training_pipeline(data_path:str,plots_dir:str):
    """Executes ingestion, split processing, grid evaluation tuning, and local serialization blocks."""
    logger.info(f"💾 Ingesting processed cluster matrix from workspace: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing analytical target frame at path location: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Define primary model targets and remove keys/leakage blocks
    target_col = 'Is_Churned'
    drop_cols = ['Customer ID', target_col]
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]
    
    # Split using stratified baseline constraints to preserve target balance ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Isolate feature types for scikit-learn preprocessing ColumnTransformer mapping
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
    
    # Compute native custom balancing metrics to counter residual noise scale shifts
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    calculated_scale_weight = float(neg_count) / float(pos_count)
    
    # Setup the multi-model dictionary grids
    model_configs = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            "grid": {'classifier__C': [0.01, 0.1, 1.0, 10.0]}
        },
        "Random_Forest": {
            "model": RandomForestClassifier(class_weight='balanced', random_state=42),
            "grid": {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [6, 12]}
        },
        "XGBoost": {
            "model": XGBClassifier(scale_pos_weight=calculated_scale_weight, eval_metric='logloss', random_state=42),
            "grid": {'classifier__n_estimators': [100, 150], 'classifier__learning_rate': [0.05, 0.1], 'classifier__max_depth': [4, 6]}
        }
    }
    
    # Build core infrastructure filesystem folders
    models_dir = "models"
    plots_dir = "plots"
    outputs_dir = "outputs"
    for d in [models_dir, plots_dir, outputs_dir]:
        os.makedirs(d, exist_ok=True)
        
    logger.info("⚡ Executing parallel multi-model tuning architecture runs across MLflow profiles...")
    
    for model_name, config in model_configs.items():
        with mlflow.start_run(run_name=f"Cluster_{model_name}"):
            # Bind composite scaling preprocessor and raw estimator into unified pipeline 
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', config['model'])
            ])
            
            # Orchestrate Stratified 5-Fold Grid Search
            grid_search = GridSearchCV(
                estimator=full_pipeline,
                param_grid=config['grid'],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Predict validation metrics
            preds = best_model.predict(X_test)
            probs = best_model.predict_proba(X_test)[:, 1]
            
            # Generate metrics and print evaluation matrices
            auc_score = roc_auc_score(y_test, probs)
            report = classification_report(y_test, preds, output_dict=True)
            
            # Log structural metrics to remote dashboard interfaces
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("ROC_AUC", auc_score)
            mlflow.log_metric("Churn_Recall", report['1']['recall'])
            mlflow.log_metric("F1_Score", report['1']['f1-score'])
            
            logger.info(f" [{model_name}] Parameter Optimization Array Sweep Finished.")
            logger.info(f" [{model_name}] Eval AUC: {auc_score:.4f} | Target Churn Recall: {report['1']['recall']:.4f}")
            
            # Log core models back to active experiments context panels
            mlflow.sklearn.log_model(
                sk_model=best_model, 
                artifact_path="model",
                skops_trusted_types=[
                    "numpy.dtype", 
                    "numpy.core.multiarray._reconstruct",
                    "numpy.ndarray",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier"
                ]
            )
                
            # Render evaluation chart plots
            plt.figure(figsize=(6, 5))
            fpr, tpr, _ = roc_curve(y_test, probs)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', color='darkorange', lw=2)
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Cluster-Segmented ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            
            plot_file = os.path.join(plots_dir, f"cluster_roc_{model_name}.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(plot_file)
            
            #  Serializing optimized instances safely to local disk workspace configurations
            model_filename = os.path.join(models_dir, f"cluster_{model_name.lower()}_best.pkl")
            joblib.dump(best_model, model_filename)
            mlflow.log_artifact(model_filename)
            logger.info(f"💾 Saved binary serialization layer path directly to: {model_filename}")
            
            #  Dynamically resolve and output localized feature interpretations
            try:
                fitted_preprocessor = best_model.named_steps['preprocessor']
                try:
                    all_features = fitted_preprocessor.get_feature_names_out().tolist()
                except Exception:
                    all_features = list(X_train.columns)
                    
                raw_estimator = best_model.named_steps['classifier']
                
                if "Logistic_Regression" in model_name:
                    interpret_logistic_coefficients(raw_estimator, all_features, outputs_dir)
                elif "XGBoost" in model_name:
                    interpret_xgboost_importance(raw_estimator, all_features, outputs_dir)
            except Exception as e:
                logger.warning(f"⚠️ Could not complete matrix extraction weights for {model_name}: {str(e)}")

    logger.info(" Cluster-stratified model prediction pipeline execution complete.")

