# src/config_loader.py

import yaml
from pydantic import BaseModel, field_validator, Field
from typing import List, Dict, Any, Optional, Union
import os
import logging

logger = logging.getLogger(__name__)

class DataPaths(BaseModel):
    """Pydantic model for data paths configuration."""
    database_file: str = "your_database_name.db" # Default, to be located in data/
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    output_dir: str = "outputs"
    model_output_dir: str = Field(default="outputs/models", alias="model_dir") # Example of alias
    metrics_output_file: str = "outputs/metrics.json"
    plot_output_dir: str = "outputs/plots"

    @field_validator('database_file')
    def validate_db_path(cls, v, values, **kwargs):
        # Assuming run.sh is at the root, and data_loader.py will construct relative path
        # This validator just checks the name format, not existence here.
        if not v.endswith(".db"):
            raise ValueError("database_file must be a .db file")
        return v

class DataLoadingParams(BaseModel):
    """Pydantic model for data loading parameters."""
    table_name: str = "main_table"
    target_column: str = "target"

class FeatureParams(BaseModel):
    """Pydantic model for feature parameters."""
    numerical_features: List[str]
    categorical_features: List[str]
    features_to_drop: Optional[List[str]] = None
    datetime_features: Optional[List[str]] = None


class PreprocessingParams(BaseModel):
    """Pydantic model for preprocessing parameters."""
    numerical_imputer_strategy: str = "mean"
    categorical_imputer_strategy: str = "most_frequent"
    scaler_type: str = "standard" # e.g., 'standard', 'minmax'
    # Add more specific encoder params if needed, e.g., for specific columns
    one_hot_encode_categories: Optional[bool] = True # Simple flag for categorical encoding approach


class FeatureEngineeringParams(BaseModel):
    """Pydantic model for feature engineering parameters."""
    create_polynomial_features: bool = False
    polynomial_degree: int = 2
    interaction_term_pairs: Optional[List[List[str]]] = None
    # Add other feature engineering flags/parameters

class ModelParams(BaseModel):
    """Pydantic model for individual model parameters."""
    name: str
    hyperparameters: Dict[str, Any] # For fixed hyperparameters
    hyperparameter_tuning_grid: Optional[Dict[str, Any]] = None # For hyperparameter search

class TrainingParams(BaseModel):
    """Pydantic model for training parameters."""
    test_size: float = 0.2
    cv_folds: int = 5
    # Stratification will be handled by StratifiedKFold, target is known
    hyperparameter_tuning_strategy: str = "RandomizedSearchCV" # or "GridSearchCV", "Optuna"
    tuning_iterations: Optional[int] = 10 # For RandomizedSearchCV or Optuna
    models_to_run: List[ModelParams]
    ensemble_models: Optional[List[Dict[str, Any]]] = None # For stacking/blending

class EvaluationParams(BaseModel):
    """Pydantic model for evaluation parameters."""
    metrics: List[str] = ["accuracy", "f1_score", "roc_auc_score", "precision_score", "recall_score"]
    positive_label: Union[str, int] = 1 # For binary classification metrics

class MLflowParams(BaseModel):
    """Pydantic model for MLflow tracking parameters."""
    experiment_name: str = "Default_ML_Experiment"
    tracking_uri: Optional[str] = None # Defaults to local ./mlruns
    log_models: bool = True
    log_shap_summary: bool = True

class GlobalConfig(BaseModel):
    """Main Pydantic model for the entire configuration."""
    project_name: str = "ML_Pipeline_Project"
    pipeline_version: str = "1.0.0"
    random_seed: int = 42
    data_paths: DataPaths
    data_loading_params: DataLoadingParams
    feature_params: FeatureParams
    preprocessing_params: PreprocessingParams
    feature_engineering_params: Optional[FeatureEngineeringParams] = Field(default_factory=FeatureEngineeringParams)
    training_params: TrainingParams
    evaluation_params: EvaluationParams
    mlflow_params: MLflowParams
    fastapi_port: int = 8000

    @field_validator('training_params')
    def check_models_to_run(cls, v):
        if not v.models_to_run:
            raise ValueError("At least one model must be specified in training_params.models_to_run")
        return v

def load_config(config_path: str) -> GlobalConfig:
    """
    Loads configuration from a YAML file and validates it using Pydantic.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        GlobalConfig: A Pydantic model instance containing the validated configuration.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
        pydantic.ValidationError: If the configuration does not match the schema.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at: {config_path}")
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        logger.info(f"Configuration file loaded successfully from {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise

    try:
        config = GlobalConfig(**config_dict)
        logger.info("Configuration validated successfully.")
        return config
    except Exception as e: # Catches Pydantic's ValidationError and others
        logger.error(f"Configuration validation error: {e}")
        # To see detailed Pydantic errors, you might want to print e.errors()
        # if hasattr(e, 'errors'):
        #    logger.error(f"Detailed Pydantic validation errors: {e.errors()}")
        raise

if __name__ == '__main__':
    # Example usage (optional, for direct testing of this module)
    # Create a dummy config.yaml for testing
    dummy_config_content = """
project_name: "Test Project"
pipeline_version: "0.1.0"
random_seed: 42

data_paths:
  database_file: "test_data.db"
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"
  output_dir: "test_outputs"
  model_dir: "test_outputs/models" # Note: using alias 'model_dir'
  metrics_output_file: "test_outputs/metrics.json"
  plot_output_dir: "test_outputs/plots"

data_loading_params:
  table_name: "sample_table"
  target_column: "response"

feature_params:
  numerical_features: ["age", "income"]
  categorical_features: ["gender", "city"]
  features_to_drop: ["id"]

preprocessing_params:
  numerical_imputer_strategy: "median"
  categorical_imputer_strategy: "missing_value" # A custom string if not 'most_frequent' etc.
  scaler_type: "minmax"
  one_hot_encode_categories: true

feature_engineering_params:
  create_polynomial_features: true
  polynomial_degree: 2
  interaction_term_pairs:
    - ["age", "income"]

training_params:
  test_size: 0.25
  cv_folds: 3
  hyperparameter_tuning_strategy: "RandomizedSearchCV"
  tuning_iterations: 5
  models_to_run:
    - name: "logistic_regression"
      hyperparameters: {"C": 1.0, "solver": "liblinear"}
      hyperparameter_tuning_grid: {"C": [0.1, 1, 10]}
    - name: "random_forest"
      hyperparameters: {"n_estimators": 100, "random_state": 42}
      hyperparameter_tuning_grid: {"n_estimators": [50, 100, 150], "max_depth": [null, 10, 20]}

evaluation_params:
  metrics: ["accuracy", "f1_score"]
  positive_label: 1

mlflow_params:
  experiment_name: "Test_Experiment"
  log_models: true
  log_shap_summary: false

fastapi_port: 8001
"""
    logging.basicConfig(level=logging.INFO)
    dummy_config_path = "dummy_config_test.yaml"
    with open(dummy_config_path, "w") as f:
        f.write(dummy_config_content)

    logger.info("Testing config_loader module...")
    try:
        config = load_config(dummy_config_path)
        logger.info(f"Project Name from config: {config.project_name}")
        logger.info(f"Numerical features: {config.feature_params.numerical_features}")
        logger.info(f"Model to run [0] name: {config.training_params.models_to_run[0].name}")
        logger.info(f"MLflow experiment: {config.mlflow_params.experiment_name}")
        logger.info(f"Model output directory: {config.data_paths.model_output_dir}")

        # Test a failing validation (e.g., missing required field)
        faulty_config_content = dummy_config_content.replace("numerical_features:", "num_features:")
        faulty_config_path = "faulty_config_test.yaml"
        with open(faulty_config_path, "w") as f:
            f.write(faulty_config_content)
        try:
            load_config(faulty_config_path)
        except Exception as e: # Pydantic ValidationError is an Exception subclass
            logger.info(f"Successfully caught expected validation error for faulty config: {type(e)}")

    except Exception as e:
        logger.error(f"Error during config_loader test: {e}")
    finally:
        # Clean up
        if os.path.exists(dummy_config_path):
            os.remove(dummy_config_path)
        if os.path.exists(faulty_config_path):
            os.remove(faulty_config_path)
        logger.info("Config_loader module test complete.")