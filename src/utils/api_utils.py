# src/api_utils.py
import logging
import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

import httpx
from httpx import ConnectTimeout, ReadTimeout, RequestError

# Assuming src is in PYTHONPATH or the execution is relative to the project root
# If not, adjust import paths accordingly e.g., from ..utils import load_config, setup_logging
try:
    from src.utils.pipe_utils import load_config, setup_logging
    from src.preprocessing import preprocess_data
except ImportError:
    # Fallback for cases where the script might be run in a context where src is not directly in path
    # This might happen in some testing scenarios or if PYTHONPATH is not set
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.pipe_utils import load_config, setup_logging
    from preprocessing import preprocess_data


# --- Configuration and Logger Setup ---
# Define the path to the deployment configuration file
# Adjust if your project structure is different or if you pass it as an env var
CONFIG_FILE_PATH = os.getenv("API_CONFIG_PATH", "config/deploy_config.yaml")
PIPELINE_CONFIG_PATH = os.getenv("PIPELINE_CONFIG_PATH", "config/pipeline_config.yaml")
PREPROCESSOR_ARTIFACT_PATH = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.joblib") 

# Load deployment configuration
# This will be a global-like variable for the API module
try:
    DEPLOY_CONFIG: Dict[str, Any] = load_config(CONFIG_FILE_PATH)
except Exception as e:
    # Use basic logging if config loading fails, as logger setup depends on config
    logging.basicConfig(level=logging.ERROR)
    logging.critical(f"CRITICAL_ERROR: Failed to load deployment configuration from {CONFIG_FILE_PATH}. Error: {e}", exc_info=True)
    # Exit if config is essential for startup; for some utilities, partial functionality might be okay
    # For an API, config is usually critical.
    sys.exit(f"API cannot start: Failed to load configuration from {CONFIG_FILE_PATH}.")

# Setup API logger using the loaded configuration
try:
    _api_logger_config = DEPLOY_CONFIG.get('logging', {}).get('api')
    _common_logging_config = DEPLOY_CONFIG.get('logging', {})
    if not _api_logger_config:
        raise ValueError("API logging configuration ('logging.api') not found in deploy_config.")
    
    API_LOGGER: logging.Logger = setup_logging(
        logger_specific_config=_api_logger_config,
        common_logging_config=_common_logging_config
    )
    API_LOGGER.info(f"API Logger initialized successfully using config from {CONFIG_FILE_PATH}.")
except Exception as e:
    # Fallback to basic logging if full setup fails
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (fallback_api_logger)')
    API_LOGGER = logging.getLogger("fallback_api_logger")
    API_LOGGER.error(f"Failed to initialize API logger with custom configuration: {e}. Using basic logger.", exc_info=True)
    # Depending on policy, you might want to exit or continue with a basic logger.
    # API_LOGGER.warning("API will continue with a basic logger configuration.")

# Load pipeline configuration
try:
    PIPELINE_CONFIG: Dict[str, Any] = load_config(PIPELINE_CONFIG_PATH)
    API_LOGGER.info(f"Successfully loaded pipeline configuration for API from {PIPELINE_CONFIG_PATH}")
except Exception as e:
    API_LOGGER.critical(f"CRITICAL_ERROR: Failed to load pipeline configuration from {PIPELINE_CONFIG_PATH}. Error: {e}", exc_info=True)
    PIPELINE_CONFIG = None # API should not function without this

# Load preprocessor artifact
try:
    FITTED_PREPROCESSOR = joblib.load(PREPROCESSOR_ARTIFACT_PATH)
    API_LOGGER.info(f"Successfully loaded fitted preprocessor from {PREPROCESSOR_ARTIFACT_PATH}")
except Exception as e:
    API_LOGGER.critical(f"CRITICAL_ERROR: Failed to load preprocessor artifact from {PREPROCESSOR_ARTIFACT_PATH}. Error: {e}", exc_info=True)
    FITTED_PREPROCESSOR = None # API should not function without this


# --- Preprocessing Function for API ---
async def preprocess_for_prediction(
    raw_input_data: Dict[str, Any]
) -> Dict[str, Any] | None:
    """
    Preprocesses raw input data from the API to match the model's expected input format.
    Uses the loaded PIPELINE_CONFIG and FITTED_PREPROCESSOR.
    """
    global API_LOGGER # Access the globally defined API_LOGGER

    if PIPELINE_CONFIG is None:
        API_LOGGER.error("Pipeline configuration (PIPELINE_CONFIG) is not available. Cannot preprocess data.")
        return None
    if FITTED_PREPROCESSOR is None:
        API_LOGGER.error("Fitted preprocessor (FITTED_PREPROCESSOR) is not available. Cannot preprocess data.")
        return None

    try:
        API_LOGGER.debug(f"Received raw input for preprocessing: {raw_input_data}")
        api_to_original_cols_mapping = {
            "age": "Age",
            "job": "Occupation",
            "marital": "Marital Status",
            "education": "Education Level",
            "default": "Credit Default",
            "housing": "Housing Loan",
            "loan": "Personal Loan",
            "contact": "Contact Method",
            "campaign_calls": "Campaign Calls",
            "pdays": "Previous Contact Days"
        }
        # 1. Convert raw input dict to a single-row DataFrame - The keys in raw_input_data should match the original raw feature names
        input_df = pd.DataFrame([raw_input_data])
        API_LOGGER.debug(f"Raw input converted to DataFrame for stage 1 preprocessing:\n{input_df.to_string()}")

        renamed_input_data = {}
        for api_key, value in raw_input_data.items():
            original_col_name = api_to_original_cols_mapping.get(api_key)
            if original_col_name:
                renamed_input_data[original_col_name] = value
            else:
                # Handle any other unexpected API fields if necessary
                API_LOGGER.warning(f"Unexpected API field '{api_key}' received. It will be ignored unless explicitly handled.")
        
        if 'Client ID' not in renamed_input_data:
            renamed_input_data['Client ID'] = 'API_DUMMY_CLIENT_ID' # Or None, if preprocess_data handles None for this
            API_LOGGER.debug("Added dummy 'Client ID' for preprocess_data.")
        
        input_df = pd.DataFrame([renamed_input_data])
        API_LOGGER.debug(f"DataFrame with remapped column names for stage 1 preprocessing (to be passed to preprocess_data):\n{input_df.to_string()}")
        API_LOGGER.debug(f"Columns in input_df for preprocess_data: {input_df.columns.tolist()}")

        # 2. Apply initial preprocessing steps from src/preprocessing.py
        # The `preprocess_data` function requires the full config dictionary
        # as it accesses various nested keys (e.g., config['age_processing'], config['target_column']).
        # Even if some parts like target encoding aren't strictly needed for prediction input,
        # the function might expect them. You might need to adapt preprocess_data or pass a
        # carefully crafted config if it errors due to missing keys not relevant for prediction.
        # For now, we pass the whole PIPELINE_CONFIG.
        preprocessed_df_stage1 = preprocess_data(input_df.copy(), config=PIPELINE_CONFIG)
        API_LOGGER.debug(f"DataFrame after stage 1 preprocessing (preprocess_data):\n{preprocessed_df_stage1.to_string()}")

        # Ensure no target column is present if preprocess_data added it and it's not expected by FITTED_PREPROCESSOR
        target_col_name = PIPELINE_CONFIG.get('target_column')
        if target_col_name and target_col_name in preprocessed_df_stage1.columns:
            API_LOGGER.debug(f"Dropping target column '{target_col_name}' before final transformation if it exists.")
            preprocessed_df_stage1 = preprocessed_df_stage1.drop(columns=[target_col_name])

        # 3. Apply the ColumnTransformer (one-hot encoding, scaling, etc.)
        try:
            # Get the feature names expected by the ColumnTransformer
            # This requires scikit-learn >= 1.0 for feature_names_in_
            # If using an older version, you might need to hardcode this list
            # or save it as an artifact during training.
            expected_cols_for_transformer = FITTED_PREPROCESSOR.feature_names_in_
            API_LOGGER.debug(f"Columns expected by FITTED_PREPROCESSOR: {list(expected_cols_for_transformer)}")
            API_LOGGER.debug(f"Columns available from preprocess_data output: {preprocessed_df_stage1.columns.tolist()}")

            # Ensure all expected columns are present and select them in the correct order
            # Add missing columns with NaN or appropriate default if any are unexpectedly missing
            # (though preprocess_data should ideally produce all of them)
            for col in expected_cols_for_transformer:
                if col not in preprocessed_df_stage1.columns:
                    API_LOGGER.warning(f"Expected column '{col}' for ColumnTransformer not found in preprocess_data output. Adding as NaN.")
                    preprocessed_df_stage1[col] = np.nan # Or a more appropriate default

            # Select and reorder
            input_df_for_transformer = preprocessed_df_stage1[expected_cols_for_transformer]

        except AttributeError:
            # Fallback if feature_names_in_ is not available (older scikit-learn)
            # You MUST ensure this hardcoded list is ACCURATE and reflects the training state.
            API_LOGGER.warning("FITTED_PREPROCESSOR.feature_names_in_ not available. Using hardcoded expected columns. Ensure this list is accurate!")
            hardcoded_expected_cols = [
                'occupation', 'marital_status', 'education_level', 'credit_default',
                'housing_loan', 'personal_loan', 'contact_method', 'campaign_calls',
                'previous_contact_days', 'cleaned_age', 'cc_had_negative_adjustment',
                'previously_contacted'
            ] # This list MUST match the columns FITTED_PREPROCESSOR was trained on.
            API_LOGGER.debug(f"Hardcoded expected columns for ColumnTransformer: {hardcoded_expected_cols}")
            API_LOGGER.debug(f"Columns available from preprocess_data output: {preprocessed_df_stage1.columns.tolist()}")

            for col in hardcoded_expected_cols:
                if col not in preprocessed_df_stage1.columns:
                    API_LOGGER.error(f"CRITICAL: Expected column '{col}' for ColumnTransformer not found in preprocess_data output. Cannot proceed with hardcoded list.")
                    # raise ValueError(f"Missing expected column: {col}") # Or handle error appropriately
                    return None # Stop processing

            input_df_for_transformer = preprocessed_df_stage1[hardcoded_expected_cols]

        API_LOGGER.debug(f"DataFrame ready for ColumnTransformer (FITTED_PREPROCESSOR):\n{input_df_for_transformer.to_string()}")
        API_LOGGER.debug(f"Columns for FITTED_PREPROCESSOR: {input_df_for_transformer.columns.tolist()}")


        # 7. Apply the ColumnTransformer (one-hot encoding, scaling, etc.)
        transformed_array = FITTED_PREPROCESSOR.transform(input_df_for_transformer)
        API_LOGGER.debug(f"Data transformed by FITTED_PREPROCESSOR. Shape: {transformed_array.shape}")

        # 8. Convert the processed array back to a dictionary with correct feature names
        #    These are the names after one-hot encoding, etc.
        processed_feature_names = FITTED_PREPROCESSOR.get_feature_names_out()
        API_LOGGER.debug(f"Feature names from preprocessor after transformation: {list(processed_feature_names)}")

        processed_values = [val.item() if hasattr(val, 'item') else val for val in transformed_array[0]]
        processed_input_dict = dict(zip(processed_feature_names, processed_values))

        API_LOGGER.info("Data preprocessing for prediction successful.")
        API_LOGGER.debug(f"Final processed features for MLflow model: {processed_input_dict}")

        return processed_input_dict

    except Exception as e:
        API_LOGGER.error(f"Error during API data preprocessing: {e}", exc_info=True)
        # Consider re-raising or returning a more specific error response for the API
        return None


# --- MLflow Model Serving Interaction ---
async def call_mlflow_predict_endpoint(
    model_identifier: str,
    processed_input_data_for_model: Dict[str, Any],
    timeout: int = 60  # Default timeout for the request to MLflow server
) -> Tuple[Dict[str, Any] | None, str | None]:
    """
    Calls the prediction endpoint of a specified MLflow model server.

    Args:
        model_identifier (str): The key for the model in DEPLOY_CONFIG['mlflow_model_servers']
                                (e.g., "logistic_regression").
        raw_input_data (Dict[str, Any]): Raw feature data for a single client,
                                         matching the keys defined in
                                         DEPLOY_CONFIG['streamlit_app']['raw_features_form_input'].
        timeout (int): Timeout in seconds for the request to the MLflow server.

    Returns:
        Tuple[Dict[str, Any] | None, str | None]: A tuple containing the prediction
                                                  response (dict) if successful,
                                                  or None. And an error message (str)
                                                  if an error occurred, or None.
    """
    global API_LOGGER, DEPLOY_CONFIG # Access global logger and config

    model_server_configs = DEPLOY_CONFIG.get("mlflow_model_servers", {})
    model_config = model_server_configs.get(model_identifier)

    if not model_config:
        error_msg = f"Configuration for model_identifier '{model_identifier}' not found in 'mlflow_model_servers'."
        API_LOGGER.error(error_msg)
        return None, error_msg

    mlflow_host = model_config.get("host", "127.0.0.1")
    mlflow_port = model_config.get("port")
    # Use timeout from model_config if available, else use function default
    request_timeout = model_config.get("timeout", timeout) 

    if not mlflow_port:
        error_msg = f"Port not configured for MLflow model server: {model_identifier}."
        API_LOGGER.error(error_msg)
        return None, error_msg

    # Construct the MLflow server URL (typically /invocations for predictions)
    # MLflow's built-in server for scikit-learn models uses /invocations
    # The model_uri is used by 'mlflow models serve', not directly in the invocation URL path.
    # Example: "http://127.0.0.1:5002/invocations"
    mlflow_serve_url = f"http://{mlflow_host}:{mlflow_port}/invocations"

    # Prepare payload for MLflow serving
    # Common format for sklearn models is {"dataframe_records": [input_dict]}
    # where input_dict contains {feature_name: value}
    # The raw_input_data should already be a dictionary of features for one client.
    columns = list(processed_input_data_for_model.keys())
    data = [list(processed_input_data_for_model.values())] # Data for one row
    payload = {"dataframe_split": {"columns": columns, "data": data}}
    API_LOGGER.debug(
        f"Attempting to call MLflow server for model '{model_identifier}' at {mlflow_serve_url} "
        f"with duplicated payload (batch of 2) for workaround."
    )
    
    API_LOGGER.debug(f"Attempting to call MLflow server for model '{model_identifier}' at {mlflow_serve_url} with payload: {payload}")

    try:
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            response = await client.post(mlflow_serve_url, json=payload)
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
            
            prediction_result = response.json()
            API_LOGGER.info(f"Successfully received prediction from MLflow model '{model_identifier}'. "
                           f"Response status: {response.status_code}")
            API_LOGGER.debug(f"MLflow response for '{model_identifier}': {prediction_result}")
            return prediction_result, None

    except ConnectTimeout:
        error_msg = (f"Connection timed out while trying to reach MLflow model server "
                     f"'{model_identifier}' at {mlflow_serve_url}.")
        API_LOGGER.error(error_msg, exc_info=True)
        return None, error_msg
    except ReadTimeout:
        error_msg = (f"Read timed out while waiting for response from MLflow model server "
                     f"'{model_identifier}' at {mlflow_serve_url}.")
        API_LOGGER.error(error_msg, exc_info=True)
        return None, error_msg
    except httpx.HTTPStatusError as e:
        error_msg = (f"MLflow model server '{model_identifier}' at {mlflow_serve_url} "
                     f"returned an error: {e.response.status_code} - {e.response.text}")
        API_LOGGER.error(error_msg, exc_info=True)
        API_LOGGER.error(f"Full response body: {e.response.text}")
        return None, f"Model serving error: {e.response.status_code} - {e.response.reason_phrase}"
    except RequestError as e: # Catch other httpx request errors (e.g., DNS resolution, connection refused)
        error_msg = (f"Request error occurred while communicating with MLflow model server "
                     f"'{model_identifier}' at {mlflow_serve_url}: {type(e).__name__} - {e}")
        API_LOGGER.error(error_msg, exc_info=True)
        return None, error_msg
    except Exception as e: # Catch any other unexpected errors
        error_msg = (f"An unexpected error occurred while calling MLflow model server "
                     f"'{model_identifier}' for prediction: {type(e).__name__} - {e}")
        API_LOGGER.error(error_msg, exc_info=True)
        return None, "An unexpected error occurred during prediction."
