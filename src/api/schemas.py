# src/api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from enum import Enum

# Import API_LOGGER and DEPLOY_CONFIG to dynamically create ModelName enum
# and ModelInput fields if desired, or define them statically.
# For simplicity and clarity, we'll define them statically here
# and the deploy_config.yaml structure we've discussed.
# The api_utils itself should not be imported here to avoid circular dependencies
# if schemas were to be used by api_utils for some reason (not the case here).

# Dynamically getting model names from config for Enum:
# from src.api.api_utils import DEPLOY_CONFIG
# available_model_keys = list(DEPLOY_CONFIG.get("available_models", {}).keys())
# if not available_model_keys: # Fallback if config not loaded or empty
#     available_model_keys = ["logistic_regression", "random_forest", "gradient_boosting"]
# ModelName = Enum("ModelName", {key.upper().replace("_", ""): key for key in available_model_keys})
# This dynamic approach is powerful but makes static analysis harder.
# the models are fixed, so a static Enum is clearer.

class ModelName(str, Enum):
    """
    Enum for available model identifiers.
    These must match the keys in `DEPLOY_CONFIG['mlflow_model_servers']`
    and `DEPLOY_CONFIG['available_models']`.
    """
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"

class ClientDataInput(BaseModel):
    """
    Pydantic model for the raw client input data.
    Fields are derived from `deploy_config.yaml -> streamlit_app -> raw_features_form_input`.
    All fields are required as per typical model input requirements unless specified otherwise.
    Includes example values for API documentation.
    """
    age: int = Field(..., example=42, description="Client's age in years.")
    job: str = Field(..., example="management", description="Type of client's job.")
    marital: str = Field(..., example="married", description="Client's marital status.")
    education: str = Field(..., example="university.degree", description="Client's education level.")
    default: str = Field(..., example="no", description="Does the client have credit in default? ('yes', 'no', 'unknown')")
    housing: str = Field(..., example="yes", description="Does the client have a housing loan? ('yes', 'no', 'unknown')")
    loan: str = Field(..., example="no", description="Does the client have a personal loan? ('yes', 'no', 'unknown')")
    
    contact: str = Field(..., example="cellular", description="Last contact communication type ('cellular', 'telephone').")
    campaign_calls: int = Field(..., example=1, description="Number of contacts performed during this campaign for this client. Your config note: Negative values allowed, to be handled by system (model preprocessor).")
    pdays: int = Field(..., example=999, description="Days passed since client was last contacted (999 means not previously contacted).")

    class Config:
        # schema_extra provides an example for the FastAPI documentation UI
        schema_extra = {
            "example": {
                "age": 30, "job": "technician", "marital": "married", "education": "university.degree",
                "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
                "campaign_calls": 1,
                "pdays": 999
            }
        }

class PredictionRequest(BaseModel):
    """
    Request body for the /predict endpoint.
    It includes the model to use and the client data.
    """
    model_name: ModelName = Field(..., description="Identifier of the model to use for prediction.")
    client_data: ClientDataInput = Field(..., description="Raw input data for a single client.")

class PredictionOutput(BaseModel):
    """
    Response model for a successful prediction.
    """
    model_used: str = Field(..., example="logistic_regression", description="Identifier of the model that was used for this prediction.")
    predicted_class: str = Field(..., example="yes", description="The predicted class label ('yes' or 'no').")
    probability_yes: Optional[float] = Field(
        None,
        ge=0.0, le=1.0, # Probability constraint
        example=0.75,
        description="Predicted probability for the positive class ('yes'). Included if available."
    )
    # You could add more fields here if needed, e.g., request_id, timestamp
    # input_features_received: Optional[Dict[str, Any]] = Field(None, description="A copy of the input features received by the API.")

class HealthResponse(BaseModel):
    """
    Response model for the health check endpoint.
    """
    status: str = Field(..., example="healthy")
    message: str = Field(..., example="API is up and running.")
    timestamp: Optional[str] = Field(None, example="2025-05-11T12:00:00.000Z")

# You might also define a generic error response model if you want to standardize errors further,
# though FastAPI's HTTPException is often sufficient.
# class APIErrorResponse(BaseModel):
#     detail: str
#     error_code: Optional[int] = None