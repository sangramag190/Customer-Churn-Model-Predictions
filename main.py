#!/usr/bin/env python3
import os
import sys
import logging

# Append project path routing structures smoothly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_prep import run_data_preparation, create_customer_analytical_dataset
from src.train import run_model_training_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ORCHESTRATOR) : %(message)s")
logger = logging.getLogger("Master_Execution_Pipeline")

def main():
    logger.info("LAUNCHING END-TO-END STRATIFIED CHURN INFRASTRUCTURE PIPELINE    ")
    
    # Path initializations
    RAW_DATA_PATH = "data/raw/Online retail dataset.xlsx"
    CLEANED_TX_PATH = "data/processed/cleaned_transactions.csv"
    CUSTOMER_MATRIX_PATH = "data/processed/customer_analytical_dataset.csv"
    MODELS_DIR = "models"
    PLOTS_DIR = "notebooks/plots"
    
    try:
        logger.info(" STAGE 1/3: Running Transaction Processing and K-Means Clustering...")
        run_data_preparation(
            input_path=RAW_DATA_PATH,
            output_path=CLEANED_TX_PATH,
            models_dir=MODELS_DIR
        )
        
        logger.info(" STAGE 2/3: Executing Leak-Free Customer Profiling & Target Assignment...")
        create_customer_analytical_dataset(
            transactions_path=CLEANED_TX_PATH,
            output_path=CUSTOMER_MATRIX_PATH
        )
        
        logger.info(" STAGE 3/3: Running GridSearchCV Tuning loops via MLflow Contexts...")
        run_model_training_pipeline(
            data_path=CUSTOMER_MATRIX_PATH,
            plots_dir=PLOTS_DIR
        )
        
        logger.info(" SUCCESS: Entire production pipeline executed without failure metrics.")
        
    except Exception as e:
        logger.critical(f" PIPELINE RUNTIME FAILURE: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()