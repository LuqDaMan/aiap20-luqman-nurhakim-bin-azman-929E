import argparse
import logging
import joblib
import sys
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import datetime

from utils.pipe_utils import load_config, setup_logging, set_global_random_seed

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
        # MLflow run name (optional, but good for organization)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.get('project_name', 'MLPipelineRun')}_{timestamp}"
        artifacts_dir = config['output_paths']['artifacts_dir']

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logging.info(f"MLflow Run Started: Name='{run_name}', ID='{run_id}'")
            mlflow.log_param("config_path", config_path) # Log config path
            mlflow.log_dict(config, "pipeline_config.yaml")

            # --- Step 1: Data Ingestion ---
            logging.info("===== Initiating Data Ingestion =====")
            raw_df = ingest_data(config['data_source']) 
            logging.info(f"Data ingestion complete. Initial DataFrame shape: {raw_df.shape}")

            # --- Step 2: Data Cleaning & Preprocessing ---
            logging.info("===== Initiating Data Cleaning & Preprocessing =====")
            cleaned_df = preprocess_data(raw_df, config) # Replace with actual call
            logging.info(f"Data cleaning and preprocessing complete. DataFrame shape: {cleaned_df.shape}")

            # --- Step 3: Feature Engineering & Data Splitting ---
            logging.info("===== Initiating Feature Engineering & Data Splitting =====")
            X_train, X_test, y_train, y_test, preprocessor_object = engineer_features_and_split_data(cleaned_df.copy(), config)
            os.makedirs(artifacts_dir, exist_ok=True)
            preprocessor_filename = "preprocessor.joblib"
            preprocessor_save_path = os.path.join(artifacts_dir, preprocessor_filename)
            try:
                joblib.dump(preprocessor_object, preprocessor_save_path)
                logging.info(f"Preprocessor object saved to: {preprocessor_save_path}")
                # Log as an MLflow artifact as well, so it's versioned with the run
                mlflow.log_artifact(preprocessor_save_path, artifact_path="preprocessor_artefact") # artifact_path is a subfolder in MLflow
            except Exception as e:
                logging.error(f"Error saving preprocessor object: {e}", exc_info=True)
                # Decide if this is a critical error that should stop the pipeline
                # raise # Uncomment to make it critical
            logging.info("Feature engineering and data splitting complete.")

            # --- Step 4: Model Training & Tuning ---
            logging.info("===== Initiating Model Training & Tuning =====")
            trained_models, best_model_name, best_model_object = train_and_tune_models(X_train, y_train, config, run_id) 
            logging.info(f"Model training and tuning complete. Best model identified: {best_model_name}")

            # --- Step 5: Model Evaluation ---
            logging.info(f"===== Initiating Final Evaluation for Best Model: {best_model_name} =====")
            all_evaluation_results = [] # To store results for CSV

            if trained_models:
                for model_name, model_pipeline in trained_models.items():
                    if model_pipeline:
                        logging.info(f"--- Evaluating model: {model_name} on the test set ---")
                        evaluation_results = evaluate_model_on_test_set(
                            model_pipeline=model_pipeline,
                            X_test=X_test,
                            y_test=y_test,
                            model_name=model_name,
                            config=config,
                            run_id=run_id # Add run_id here if you use it for plot naming within evaluate_model_on_test_set
                        )
                        all_evaluation_results.append(evaluation_results)
                        logging.info(f"Evaluation results for {model_name}: {evaluation_results}")
                    else:
                        logging.warning(f"Model pipeline for '{model_name}' is None. Skipping evaluation.")
            else:
                logging.error("No models were trained successfully. Skipping evaluation.")
            
            if all_evaluation_results:
                results_df = pd.DataFrame(all_evaluation_results)
                
                # Ensure results directory exists
                results_dir = config.get("output_paths", {}).get("results_dir", "results/")
                os.makedirs(results_dir, exist_ok=True)

                # Create a run-specific CSV filename
                results_filename = f"evaluation_metrics_run_{run_id}.csv" 
                results_csv_path = os.path.join(results_dir, results_filename)
                
                try:
                    results_df.to_csv(results_csv_path, index=False)
                    logging.info(f"All model evaluation results saved to: {results_csv_path}")
                    mlflow.log_artifact(results_csv_path, artifact_path="results") 
                except Exception as e:
                    logging.error(f"Failed to save evaluation results CSV: {e}")
            else:
                logging.info("No evaluation results to save to CSV.")

            logging.info(f"MLflow Run Ended: ID='{run_id}'")
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
    log_file_with_timestamp = f"{log_dir}/{log_name}_{timestamp}{log_ext}"

    # Create the two dictionaries that setup_logging expects
    logger_specific_config = {
        'logger_name': 'pipeline',  # The name you want for your logger
        'level': pipeline_config.get('logging', {}).get('level', 'INFO'),
        'log_dir': log_dir,
        'log_file_base_name': f"{log_name}_{timestamp}{log_ext}"  # Just the filename part
    }

    common_logging_config = {
        'format': pipeline_config.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        'date_format': pipeline_config.get('logging', {}).get('date_format', '%Y-%m-%d %H:%M:%S'),
        'timestamp_log_files': False,  # We've already added timestamp to the filename
        'max_size_bytes': pipeline_config.get('logging', {}).get('max_size_bytes', 10 * 1024 * 1024),
        'backup_count': pipeline_config.get('logging', {}).get('backup_count', 3)
    }

    # Call setup_logging with the properly structured arguments
    logger = setup_logging(logger_specific_config, common_logging_config)

    # 3. Set global random seed for reproducibility
    set_global_random_seed(pipeline_config.get("random_seed", 42)) # Default to 42 if not in config

    # 4. Run the main pipeline
    run_pipeline(args.config) # Pass the config path