# src/streamlit_utils.py
import logging
import os
import sys
from typing import Dict, Any, Tuple, Optional

import requests # Using requests for synchronous calls from Streamlit
from requests.exceptions import RequestException, Timeout, ConnectionError

# Assuming src is in PYTHONPATH or the execution is relative to the project root
try:
    from src.utils.pipe_utils import load_config, setup_logging
    # Schemas might be needed if we construct complex request bodies typed here,
    # but for sending a dict payload, it's not strictly necessary in utils.
    # from src.api.schemas import PredictionRequest # For type hinting if needed
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Go up to src/
    from utils.pipe_utils import load_config, setup_logging
    # from api.schemas import PredictionRequest


# --- Configuration and Logger Setup ---
CONFIG_FILE_PATH = os.getenv("STREAMLIT_CONFIG_PATH", "config/deploy_config.yaml")

try:
    DEPLOY_CONFIG: Dict[str, Any] = load_config(CONFIG_FILE_PATH)
except Exception as e:
    logging.basicConfig(level=logging.ERROR)
    logging.critical(f"CRITICAL_ERROR: Failed to load deployment configuration for Streamlit from {CONFIG_FILE_PATH}. Error: {e}", exc_info=True)
    sys.exit(f"Streamlit app cannot start: Failed to load configuration from {CONFIG_FILE_PATH}.")

try:
    _streamlit_logger_config = DEPLOY_CONFIG.get('logging', {}).get('streamlit')
    _common_logging_config = DEPLOY_CONFIG.get('logging', {})
    if not _streamlit_logger_config:
        raise ValueError("Streamlit logging configuration ('logging.streamlit') not found in deploy_config.")

    STREAMLIT_LOGGER: logging.Logger = setup_logging(
        logger_specific_config=_streamlit_logger_config,
        common_logging_config=_common_logging_config
    )
    STREAMLIT_LOGGER.info(f"Streamlit Logger initialized successfully using config from {CONFIG_FILE_PATH}.")
except Exception as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (fallback_streamlit_logger)')
    STREAMLIT_LOGGER = logging.getLogger("fallback_streamlit_logger")
    STREAMLIT_LOGGER.error(f"Failed to initialize Streamlit logger with custom configuration: {e}. Using basic logger.", exc_info=True)


# --- FastAPI Backend Interaction ---
def call_fastapi_predict(
    model_name: str,
    client_data_dict: Dict[str, Any],
    timeout: int = 60  # Default timeout for the request to FastAPI
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Calls the FastAPI backend's /predict/ endpoint.

    Args:
        model_name (str): The identifier of the model to use (e.g., "logistic_regression").
        client_data_dict (Dict[str, Any]): A dictionary containing the raw client input data.
        timeout (int): Timeout in seconds for the HTTP request.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str]]:
            A tuple containing the JSON response from the API (dict) if successful, or None.
            And an error message (str) if an error occurred, or None.
    """
    global STREAMLIT_LOGGER, DEPLOY_CONFIG

    streamlit_app_config = DEPLOY_CONFIG.get("streamlit_app", {})
    api_base_url = streamlit_app_config.get("api_base_url")
    predict_endpoint_path = streamlit_app_config.get("predict_endpoint")

    if not api_base_url or not predict_endpoint_path:
        error_msg = "FastAPI URL configuration (api_base_url or predict_endpoint) not found in streamlit_app config."
        STREAMLIT_LOGGER.error(error_msg)
        return None, error_msg

    # Ensure no double slashes if predict_endpoint_path starts with /
    if api_base_url.endswith('/') and predict_endpoint_path.startswith('/'):
        full_predict_url = api_base_url[:-1] + predict_endpoint_path
    elif not api_base_url.endswith('/') and not predict_endpoint_path.startswith('/'):
        full_predict_url = api_base_url + '/' + predict_endpoint_path
    else:
        full_predict_url = api_base_url + predict_endpoint_path
        
    # Prepare the payload according to FastAPI's PredictionRequest schema
    payload = {
        "model_name": model_name,
        "client_data": client_data_dict
    }

    STREAMLIT_LOGGER.info(f"Sending prediction request to FastAPI: {full_predict_url} for model '{model_name}'.")
    STREAMLIT_LOGGER.debug(f"Payload: {payload}")

    try:
        response = requests.post(full_predict_url, json=payload, timeout=timeout)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)

        response_data = response.json()
        STREAMLIT_LOGGER.info(f"Successfully received prediction from FastAPI. Status: {response.status_code}")
        STREAMLIT_LOGGER.debug(f"FastAPI response: {response_data}")
        return response_data, None

    except Timeout:
        error_msg = f"Request to FastAPI backend at {full_predict_url} timed out after {timeout} seconds."
        STREAMLIT_LOGGER.error(error_msg, exc_info=True)
        return None, error_msg
    except ConnectionError:
        error_msg = f"Could not connect to FastAPI backend at {full_predict_url}. Ensure the API server is running."
        STREAMLIT_LOGGER.error(error_msg, exc_info=True)
        return None, error_msg
    except requests.HTTPError as e:
        # Try to get more specific error detail from API response if available
        try:
            api_error_detail = e.response.json().get("detail", e.response.text)
        except ValueError: # If response is not JSON
            api_error_detail = e.response.text
        error_msg = (f"FastAPI backend at {full_predict_url} returned an error: "
                     f"{e.response.status_code} - {api_error_detail}")
        STREAMLIT_LOGGER.error(error_msg, exc_info=True) # Log full traceback
        return None, f"API request failed: {e.response.status_code} - {api_error_detail}" # User-friendly part
    except RequestException as e: # Catch other general requests exceptions
        error_msg = f"An unexpected error occurred during request to FastAPI backend at {full_predict_url}: {e}"
        STREAMLIT_LOGGER.error(error_msg, exc_info=True)
        return None, error_msg
    except Exception as e: # Catch-all for other unexpected issues
        error_msg = f"An unexpected error occurred while preparing or handling request to FastAPI: {e}"
        STREAMLIT_LOGGER.error(error_msg, exc_info=True)
        return None, "An unexpected error occurred."


if __name__ == "__main__":
    # This block allows for quick testing of this module's functionalities.
    # Ensure your `config/deploy_config.yaml` is correctly set up and accessible.
    # For `call_fastapi_predict` to succeed, the FastAPI server must be running.

    STREAMLIT_LOGGER.info("streamlit_utils.py loaded. DEPLOY_CONFIG and STREAMLIT_LOGGER available.")
    
    # Example: Print some Streamlit specific config
    st_title = DEPLOY_CONFIG.get("streamlit_app", {}).get("title")
    STREAMLIT_LOGGER.info(f"Configured Streamlit App Title (from DEPLOY_CONFIG): {st_title}")

    # --- Test call_fastapi_predict ---
    # This test requires the FastAPI server to be running.
    # STREAMLIT_LOGGER.info("Attempting to test call_fastapi_predict function...")
    # if DEPLOY_CONFIG.get("streamlit_app", {}).get("api_base_url"):
    #     # Use the schema structure from deploy_config to create sample client data
    #     # This needs to match the fields defined in `src/api/schemas.py -> ClientDataInput`
    #     # and subsequently `deploy_config.yaml -> streamlit_app -> raw_features_form_input`
    #     sample_client_data = {
    #         # Based on the user's updated schemas.py (fewer fields)
    #         "age": 45, "job": "blue-collar", "marital": "married", "education": "basic.9y",
    #         "default": "unknown", "housing": "no", "loan": "no", "contact": "cellular",
    #         "campaign_calls": 1, "pdays": 999, "previous": 0
    #         # Removed fields: month, day_of_week, duration, poutcome, emp_var_rate etc.
    #     }
    #     # Choose a model that your FastAPI server supports
    #     sample_model_name = "logistic_regression" # Or another key from DEPLOY_CONFIG['available_models']

    #     STREAMLIT_LOGGER.info(f"Test: Calling FastAPI for model '{sample_model_name}' with sample data.")
    #     prediction, error = call_fastapi_predict(
    #         model_name=sample_model_name,
    #         client_data_dict=sample_client_data
    #     )

    #     if error:
    #         STREAMLIT_LOGGER.error(f"Test call_fastapi_predict FAILED: {error}")
    #     else:
    #         STREAMLIT_LOGGER.info(f"Test call_fastapi_predict SUCCEEDED. Prediction: {prediction}")
    # else:
    #     STREAMLIT_LOGGER.warning("FastAPI URL not configured in deploy_config; skipping call_fastapi_predict test.")
    
    STREAMLIT_LOGGER.info("--- Finished streamlit_utils.py tests ---")