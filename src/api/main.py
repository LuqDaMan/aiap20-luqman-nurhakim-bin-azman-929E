# src/api/main.py
from fastapi import FastAPI, HTTPException, status, Request as FastAPIRequest
from typing import Any, Optional
from datetime import datetime, timezone

# Assuming src is in PYTHONPATH or the execution is relative to the project root
try:
    from src.api.schemas import (
        PredictionRequest,
        PredictionOutput,
        HealthResponse,
        ModelName,
        ClientDataInput # Though ClientDataInput is part of PredictionRequest
    )
    from src.utils.api_utils import (
        API_LOGGER,
        DEPLOY_CONFIG,
        PIPELINE_CONFIG,
        FITTED_PREPROCESSOR,    
        preprocess_for_prediction,
        call_mlflow_predict_endpoint
    )
except ImportError:
    # Fallback for certain execution contexts (e.g., some test runners or manual script exec)
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # Go up to project root
    from src.api.schemas import (
        PredictionRequest,
        PredictionOutput,
        HealthResponse,
        ModelName,
        ClientDataInput
    )
    from src.utils.api_utils import (
        API_LOGGER,
        DEPLOY_CONFIG,
        PIPELINE_CONFIG,
        FITTED_PREPROCESSOR,
        preprocess_for_prediction,
        call_mlflow_predict_endpoint
    )

# --- FastAPI App Initialization ---
# API metadata from deploy_config.yaml or defaults
api_config = DEPLOY_CONFIG.get("api", {})
app_title = api_config.get("title", "Bank Term Deposit Subscription Prediction API")
app_version = api_config.get("version", "1.1.0") 
app_description = api_config.get(
    "description",
    "API for predicting bank term deposit subscriptions using MLflow-served models."
)

app = FastAPI(
    title=app_title,
    version=app_version,
    description=app_description,
    # Add other FastAPI configurations like root_path if deploying behind a proxy
    # openapi_url="/api/v1/openapi.json" # Example custom OpenAPI path
)

API_LOGGER.info(f"FastAPI application initialized: {app.title} v{app.version}")

# --- API Endpoints ---

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Perform a Health Check",
    description="Returns the operational status of the API."
)
async def health_check():
    """
    Health check endpoint to confirm the API is operational.
    """
    API_LOGGER.info("Health check endpoint called.")
    return HealthResponse(
        status="healthy",
        message=f"{app.title} is up and running.",
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post(
    "/predict/", # FR-DEP-004
    response_model=PredictionOutput,
    tags=["Prediction"],
    summary="Predict Term Deposit Subscription",
    description=(
        "Accepts client data and a selected model name, "
        "then returns a prediction for term deposit subscription likelihood."
    )
)
async def predict_subscription(
    request: PredictionRequest, # FR-DEP-004a, FR-DEP-004c (model_name in body)
    http_request: FastAPIRequest # For logging client info if needed
):
    """
    Prediction endpoint.
    - Takes raw client data and a model identifier.
    - Proxies the request to the appropriate MLflow model server.
    - Returns the prediction result including class and probability.
    """
    client_host = http_request.client.host if http_request.client else "unknown"
    API_LOGGER.info(
        f"Prediction request received from {client_host} for model: {request.model_name.value}. "
        f"Input features (first few): "
        f"{ {k: v for i, (k, v) in enumerate(request.client_data.model_dump().items()) if i < 3} }..."
    )

    # Validate if the selected model is configured for serving
    # ModelName enum already validates if the string is one of the enum members.
    # This checks if the valid enum member is actually configured in deploy_config.
    model_identifier = request.model_name.value
    if model_identifier not in DEPLOY_CONFIG.get("mlflow_model_servers", {}):
        error_msg = f"Model '{model_identifier}' is not configured or available for serving."
        API_LOGGER.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )

    # Convert Pydantic model to dict for MLflow server payload
    # Your updated schemas.py defines ClientDataInput with the correct fields.
    # .model_dump() is preferred for Pydantic v2+
    raw_features_dict = request.client_data.model_dump()

    # --- Call the preprocessing function ---
    API_LOGGER.info(f"Initiating preprocessing for input data for model '{model_identifier}'.")
    processed_data_for_model = await preprocess_for_prediction(raw_features_dict) # Call the new function

    if processed_data_for_model is None:
        error_msg = "Feature preprocessing failed. Cannot proceed with prediction."
        API_LOGGER.error(error_msg)
        # It's important to distinguish this from MLflow errors. This is an internal server error.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Internal server error
            detail=error_msg
        )
    API_LOGGER.info("Feature preprocessing successful.")
    # --- End preprocessing call ---

    # Call the MLflow model serving endpoint via the utility function
    # FR-DEP-006: FastAPI application MUST correctly route requests...
    API_LOGGER.info(f"Sending processed data to MLflow model '{model_identifier}'.")
    mlflow_response, error_message = await call_mlflow_predict_endpoint(
        model_identifier=model_identifier,
        processed_input_data_for_model=processed_data_for_model
    )

    if error_message or mlflow_response is None:
        API_LOGGER.error(f"Error from MLflow service for model '{model_identifier}': {error_message}")
        # Determine appropriate HTTP status code based on error type if possible
        if "timed out" in (error_message or "").lower() or "connection" in (error_message or "").lower():
            http_status_code = status.HTTP_504_GATEWAY_TIMEOUT
        elif "model serving error" in (error_message or "").lower() or "returned an error" in (error_message or "").lower():
             http_status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            http_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(
            status_code=http_status_code,
            detail=f"Prediction failed for model '{model_identifier}': {error_message}"
        )

    # Process the successful MLflow prediction response
    # FR-DEP-005: The prediction endpoint MUST return the prediction result.
    try:
        # Assuming MLflow for sklearn returns: {"predictions": [[prob_class_0, prob_class_1]]}
        # or {"predictions": [class_label_numeric]} if probabilities are not directly output
        # For PRD_II, we need both class and probability (FR-DEP-005a, FR-DEP-005b)
        
        predictions_data = mlflow_response.get("predictions")
        API_LOGGER.info(f"MLflow response 'predictions' field: {predictions_data}")
        if predictions_data is None or not isinstance(predictions_data, list) or not predictions_data:
            raise ValueError("MLflow response 'predictions' field is missing, not a list, or empty.")

        # Assuming the prediction is for a single instance, so we take the first element
        first_prediction = predictions_data[0]
        API_LOGGER.info(f"First prediction data from MLflow: {first_prediction}")
        probability_yes: Optional[float] = None
        predicted_label_numeric: int

        if isinstance(first_prediction, list) and len(first_prediction) == 2:
            # Case: [[prob_class_0, prob_class_1]]
            # Ensure probabilities are floats
            try:
                prob_class_0 = float(first_prediction[0])
                prob_class_1 = float(first_prediction[1])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Could not convert probabilities to float: {first_prediction}. Error: {e}")

            # Class 1 is 'yes' as per typical binary classification setup
            probability_yes = prob_class_1
            predicted_label_numeric = 1 if probability_yes >= 0.5 else 0 # Standard threshold
            API_LOGGER.debug(f"Probabilities received: [P(no)={prob_class_0}, P(yes)={probability_yes}]. Predicted label: {predicted_label_numeric}")

        elif isinstance(first_prediction, (int, float)): # This case should ideally not be hit if workaround is effective
            predicted_label_numeric = int(round(float(first_prediction)))
            API_LOGGER.warning(f"Received single prediction value {first_prediction} even with workaround. "
                            f"Interpreting as class label. Probability for 'yes' may not be available.")
        else:
            raise ValueError(f"Unexpected format for prediction data from MLflow even with workaround: {first_prediction}")


        # Map numeric label to string ('yes'/'no') using config
        class_labels_map = DEPLOY_CONFIG.get("prediction_service", {}).get("output", {}).get("class_labels", {})
        predicted_class_str = class_labels_map.get(str(predicted_label_numeric))

        if predicted_class_str is None:
            raise ValueError(f"Cannot map numeric label '{predicted_label_numeric}' to string class. Check 'class_labels' in config.")

        # Construct and return the response
        # FR-DEP-005c: Structured response (e.g., JSON) - Pydantic handles this.
        response_payload = PredictionOutput(
            model_used=model_identifier,
            predicted_class=predicted_class_str,
            probability_yes=probability_yes if DEPLOY_CONFIG.get("prediction_service",{}).get("output",{}).get("include_probabilities") else None
        )
        API_LOGGER.info(f"Prediction successful for model '{model_identifier}'. Response: {response_payload.model_dump_json(indent=2)}")
        return response_payload

    except ValueError as e:
        error_msg = f"Error processing prediction result from MLflow model '{model_identifier}': {e}"
        API_LOGGER.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )
    except Exception as e:
        error_msg = f"An unexpected error occurred while finalizing prediction for model '{model_identifier}': {e}"
        API_LOGGER.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred on the server while finalizing prediction."
        )

# To run this application locally (for development):
# uvicorn src.api.main:app --reload --host <host_from_config> --port <port_from_config>
# Example from deploy_config.yaml:
# uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    # This block is for direct execution (python src/api/main.py)
    # It's more common to run FastAPI with Uvicorn CLI as shown above.
    API_LOGGER.info("Attempting to run FastAPI app with Uvicorn programmatically (for dev only).")
    
    uvicorn_host = api_config.get("host", "127.0.0.1")
    uvicorn_port = api_config.get("port", 8000)
    uvicorn_reload = api_config.get("reload", True) # From deploy_config
    
    # Note: Reload might not work as effectively when run programmatically this way
    # compared to `uvicorn src.api.main:app --reload`.
    uvicorn.run(app, host=uvicorn_host, port=uvicorn_port)
    # The line above is blocking. Code below it won't run until server stops.
    API_LOGGER.info("FastAPI application server stopped.")