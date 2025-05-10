import argparse
import logging
import sys
import os
from datetime import datetime

from utils import load_config, setup_logging, set_global_random_seed

from data_ingestion import ingest_data
from preprocessing import preprocess_data
from feat_engin import engineer_features_and_split_data
from model_trainer import train_and_tune_models
from model_evaluator import evaluate_model_on_test_set


# --- Main Pipeline Execution ---
def run_pipeline(config_path: str):
    """
    Executes the end-to-end machine learning pipeline.

    Args:
        config_path (str): Path to the pipeline configuration YAML file.
    """
    try:
        # Load configuration (already done in __main__, but good practice if this function is called elsewhere)
        # For this script, config is loaded in __main__ and passed if run_pipeline was a class method or took config dict
        config = load_config(config_path) # If not already loaded by main, load it.

        # Initial setup steps (logging and random seed are set in __main__ before calling this)
        logging.info(f"Starting the ML pipeline with configuration from: {config_path}")
        logging.info(f"Project Name: {config.get('project_name', 'N/A')}, Random Seed: {config.get('random_seed')}")

        # --- Step 1: Data Ingestion (FR-DI-001, FR-DI-002) ---
        logging.info("===== Initiating Data Ingestion =====")
        raw_df = ingest_data(config['data_source']) 
        logging.info(f"Data ingestion complete. Initial DataFrame shape: {raw_df.shape}")

        # --- Step 2: Data Cleaning & Preprocessing (FR-DP series) ---
        logging.info("===== Initiating Data Cleaning & Preprocessing =====")
        cleaned_df = preprocess_data(raw_df, config) # Replace with actual call
        logging.info(f"Data cleaning and preprocessing complete. DataFrame shape: {cleaned_df.shape}")

        # --- Step 3: Feature Engineering (FR-FE series) & Data Splitting (MD-SPLIT-001) ---
        logging.info("===== Initiating Feature Engineering & Data Splitting =====")
        X_train, X_test, y_train, y_test, preprocessor_object = engineer_features_and_split_data(cleaned_df.copy(), config)
        # Note: You'll need to decide if/how to use/save the 'preprocessor_object'. It's essential for consistent transformation of new data.
        logging.info("Feature engineering and data splitting complete.")

        # --- Step 4: Model Training & Tuning (FR-MT series, MD series) ---
        logging.info("===== Initiating Model Training & Tuning =====")
        trained_models, best_model_name, best_model_object = train_and_tune_models(X_train, y_train, config) 
        logging.info(f"Model training and tuning complete. Best model identified: {best_model_name}")

        # --- Step 5: Model Evaluation (FR-ME series, MD-EVAL series) ---
        logging.info(f"===== Initiating Final Evaluation for Best Model: {best_model_name} =====")
        if best_model_object:
            # final_evaluation_results = evaluate_final_model(best_model_object, X_test, y_test, config, best_model_name) # Replace
            final_evaluation_results = evaluate_model_on_test_set(best_model_object, X_test, y_test, best_model_name, config)
            logging.info(f"Final evaluation for {best_model_name} on test set: {final_evaluation_results}")

        logging.info("===== ML Pipeline execution finished successfully. =====")

    except FileNotFoundError:
        logging.critical(f"Configuration file not found at {config_path}. Pipeline aborted.")
        sys.exit(1) # Exit with a non-zero status code
    except Exception as e:
        logging.critical(f"An critical error occurred in the pipeline: {e}", exc_info=True)
        sys.exit(1) # Exit with a non-zero status code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline for Bank Term Deposit Subscription Prediction")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to the pipeline configuration YAML file."
    )
    args = parser.parse_args()

    # 1. Load configuration
    try:
        pipeline_config = load_config(args.config)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.critical(f"Failed to load configuration from {args.config}. Error: {e}")
        print(f"CRITICAL: Failed to load configuration from {args.config}. Error: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Setup logging using the loaded configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_base = pipeline_config.get('logging', {}).get('file', 'logs/pipeline/run.log')
    log_dir = os.path.dirname(log_file_base)
    log_filename = os.path.basename(log_file_base)
    log_name, log_ext = os.path.splitext(log_filename)
    pipeline_config['logging']['file'] = f"{log_dir}/{log_name}_{timestamp}{log_ext}"
    setup_logging(pipeline_config, 'pipeline') # This sets up the global logger

    # 3. Set global random seed for reproducibility (FR-MT-004)
    set_global_random_seed(pipeline_config.get("random_seed", 42)) # Default to 42 if not in config

    # 4. Run the main pipeline
    run_pipeline(args.config) # Pass the config path