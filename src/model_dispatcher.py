# src/model_dispatcher.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb 

# Type hinting
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, Any, Union, Optional, Type 
import logging
logger = logging.getLogger(__name__)

# Define available models
# The keys are model names as they would appear in the config file.
# The values are the model classes themselves.
# Using Type[ClassifierMixin] indicates that the values are types (classes) that inherit from ClassifierMixin.
_MODELS: Dict[str, Type[ClassifierMixin]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "svc": SVC, # Support Vector Classifier
    "xgb_classifier": xgb.XGBClassifier,
    "knn_classifier": KNeighborsClassifier,
}

def get_model(model_name: str, params: Optional[Dict[str, Any]] = None) -> ClassifierMixin:
    """
    Returns an instantiated model object based on the model name and parameters.

    Args:
        model_name (str): The name of the model to retrieve.
                          Must be a key in the _MODELS dictionary.
        params (Optional[Dict[str, Any]]): A dictionary of hyperparameters for the model.
                                     If None, the model is initialized with its default parameters.
                                     It's crucial to pass 'random_state' through params
                                     for models that support it, to ensure reproducibility.

    Returns:
        ClassifierMixin: An instance of the specified scikit-learn compatible classifier.

    Raises:
        ValueError: If the model_name is not supported or if params are invalid for the model.
        TypeError: If params is not a dictionary when provided.
    """
    model_name_lower = model_name.lower()
    if model_name_lower not in _MODELS:
        logger.error(f"Model '{model_name}' not found in dispatcher. Supported models are: {list(_MODELS.keys())}")
        raise ValueError(f"Unsupported model: {model_name}. Supported models are: {list(_MODELS.keys())}")

    model_class = _MODELS[model_name_lower]
    
    if params is None:
        model_params: Dict[str, Any] = {}
    elif not isinstance(params, dict):
        logger.error(f"Parameters for model '{model_name}' must be a dictionary, got {type(params)}.")
        raise TypeError(f"Parameters for model '{model_name}' must be a dictionary.")
    else:
        model_params = params.copy() # Use a copy to avoid modifying the original dict

    try:
        # Special handling for SVC probability if not explicitly set
        if model_class == SVC and 'probability' not in model_params:
            logger.info("Setting 'probability=True' for SVC to enable probability estimates, if not specified.")
            model_params['probability'] = True # Often needed for AUC scores or predict_proba

        # For XGBoost, if using early stopping, eval_set needs to be provided during fit, not init.
        # We'll assume standard hyperparams are passed here.
        
        model_instance = model_class(**model_params)
        logger.info(f"Successfully dispatched model '{model_name}' with parameters: {model_params}")
    except TypeError as e:
        logger.error(f"TypeError when initializing model '{model_name}' with params {model_params}: {e}")
        logger.error("This might be due to incorrect hyperparameter names or values for the chosen model.")
        raise ValueError(f"Invalid parameters for model {model_name}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while initializing model '{model_name}': {e}")
        raise

    return model_instance

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Starting model_dispatcher example...")

    # Example 1: Logistic Regression with specific params
    try:
        lr_params = {'solver': 'liblinear', 'C': 0.5, 'random_state': 42}
        lr_model = get_model(model_name="logistic_regression", params=lr_params)
        logger.info(f"Logistic Regression instance: {lr_model}")
        logger.info(f"Logistic Regression params: {lr_model.get_params()}")
    except Exception as e:
        logger.error(f"Error dispatching Logistic Regression: {e}")

    # Example 2: Random Forest with default params (except random_state for reproducibility)
    try:
        rf_params = {'random_state': 42, 'n_estimators': 50} # n_estimators is an example, could be default
        rf_model = get_model("random_forest", params=rf_params)
        logger.info(f"Random Forest instance: {rf_model}")
        logger.info(f"Random Forest params: {rf_model.get_params()}")
    except Exception as e:
        logger.error(f"Error dispatching Random Forest: {e}")

    # Example 3: XGBoost Classifier
    try:
        xgb_params = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'}
        # 'use_label_encoder': False is often recommended for newer XGBoost versions to avoid warnings.
        # 'eval_metric' can be set here or during fit.
        xgb_model = get_model("xgb_classifier", params=xgb_params)
        logger.info(f"XGBoost Classifier instance: {xgb_model}")
        logger.info(f"XGBoost Classifier params: {xgb_model.get_params()}")
    except Exception as e:
        logger.error(f"Error dispatching XGBoost Classifier: {e}")


    # Example 4: SVC with probability True (should be set by default logic if not provided)
    try:
        svc_params = {'C': 1.0, 'kernel': 'rbf', 'random_state': 42} # probability will be added by get_model
        svc_model = get_model("svc", params=svc_params)
        logger.info(f"SVC instance: {svc_model}")
        logger.info(f"SVC params: {svc_model.get_params()}")
        assert svc_model.probability is True # type: ignore
    except Exception as e:
        logger.error(f"Error dispatching SVC: {e}")

    # Example 5: Model with no params (uses defaults)
    try:
        default_lr = get_model("logistic_regression") # random_state will be None by default
        logger.info(f"Default Logistic Regression instance: {default_lr}")
        logger.info(f"Default Logistic Regression params: {default_lr.get_params()}")
    except Exception as e:
        logger.error(f"Error dispatching default Logistic Regression: {e}")

    # Example 6: Unsupported model
    try:
        unsupported_model = get_model("unsupported_neural_network")
    except ValueError as e:
        logger.warning(f"Caught expected error for unsupported model: {e}")
    except Exception as e:
        logger.error(f"Unexpected error for unsupported model: {e}")
        
    # Example 7: Model with invalid param
    try:
        invalid_params_rf = get_model("random_forest", params={"non_existent_param": 123, "random_state": 0})
    except ValueError as e: # Changed from TypeError to ValueError to align with get_model's raise
        logger.warning(f"Caught expected error for invalid param: {e}")
    except Exception as e:
        logger.error(f"Unexpected error for invalid param: {e}")


    logger.info("Model_dispatcher example finished.")