# src/model_trainer.py

import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from typing import Tuple, Dict, Any # Changed from 'Any' to 'Dict' for type hint clarity

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline as SKPipeline # Renamed for clarity
from sklearn.metrics import make_scorer, f1_score

logger = logging.getLogger('pipeline.model_trainer')

# Import SMOTE and imblearn pipeline if imblearn is a dependency
try:
    from imblearn.pipeline import Pipeline as ImbPipeline # Renamed for clarity
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    ImbPipeline = None # Define to avoid NameError if not available
    SMOTE = None
    logger.warning("Imbalanced-learn library not found. SMOTE functionality will be unavailable.")



def train_and_tune_models(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    config: dict,
    run_id: str
) -> Tuple[Dict[str, Any], str, Any]:
    """
    Trains and tunes specified machine learning models.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        config (dict): Pipeline configuration dictionary.

    Returns:
        Tuple[Dict[str, Any], str, Any]:
            - trained_model_pipelines (Dict[str, Any]): Dictionary of trained model pipelines.
            - best_model_name (str): Name of the best performing model.
            - best_model_pipeline (Any): The best performing model pipeline object.
    """
    logger.info("Starting model training and hyperparameter tuning with MLflow logging...")

    models_to_train_names = config.get('models_to_train', [])
    param_grids_config = config.get('hyperparameter_tuning', {}).get('param_grids', {})
    cv_folds = config.get('hyperparameter_tuning', {}).get('cv_folds', 5)
    imbalance_config = config.get('imbalance_handling', {})
    random_seed = config.get('random_seed')

    # Assuming 'yes' is encoded as 1 (positive class).
    f1_yes_scorer = make_scorer(f1_score, pos_label=1, average='binary')

    trained_model_pipelines: Dict[str, Any] = {}
    model_best_scores: Dict[str, float] = {}

    if not X_train.empty:
        input_example = X_train.head() # Takes the first 5 rows by default
    else:
        logger.warning("X_train is empty, cannot create input_example for MLflow model signature.")
        input_example = None

    for model_name in models_to_train_names:
        with mlflow.start_run(nested=True, run_name=f"train_{model_name}") as nested_run: # Nested run for each model
            logger.info(f"--- Training and tuning: {model_name} (MLflow Nested Run ID: {nested_run.info.run_id}) ---")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("run_id", run_id)

            base_model_instance: Any # For type hinting
            if model_name == "LogisticRegression":
                base_model_instance = LogisticRegression(random_state=random_seed, solver='liblinear', max_iter=1000)
            elif model_name == "RandomForestClassifier":
                base_model_instance = RandomForestClassifier(random_state=random_seed)
            elif model_name == "GradientBoostingClassifier":
                base_model_instance = GradientBoostingClassifier(random_state=random_seed)
            else:
                logger.warning(f"Model '{model_name}' not recognized or supported. Skipping.")
                continue

            pipeline_steps = []
            current_pipeline_class = SKPipeline # Default to scikit-learn pipeline

            # Handle class imbalance strategy
            imbalance_method = imbalance_config.get('method', 'None').upper()

            if imbalance_method == 'SMOTE':
                if IMBLEARN_AVAILABLE and SMOTE is not None and ImbPipeline is not None:
                    smote_k = imbalance_config.get('smote_k_neighbors', 5)
                    pipeline_steps.append(('smote', SMOTE(random_state=random_seed, k_neighbors=smote_k)))
                    current_pipeline_class = ImbPipeline # Use imblearn pipeline for SMOTE
                    logger.info(f"Using SMOTE (k_neighbors={smote_k}) for {model_name}.")
                else:
                    logger.warning("SMOTE selected but imblearn is not available. Proceeding without SMOTE for {model_name}.")
            elif imbalance_method == 'CLASS_WEIGHT':
                if hasattr(base_model_instance, 'class_weight'):
                    # Check if 'classifier__class_weight' is in the param grid. If so, GridSearchCV will handle it.
                    # If not, set it directly on the base model instance.
                    if f'classifier__class_weight' not in param_grids_config.get(model_name, {}):
                        try:
                            base_model_instance.set_params(class_weight='balanced')
                            logger.info(f"Set class_weight='balanced' for {model_name} directly on the model.")
                        except ValueError as e:
                            logger.warning(f"Could not set class_weight='balanced' for {model_name}: {e}. It might not support 'balanced' string or other issues.")
                else:
                    logger.warning(f"Class imbalance method 'CLASS_WEIGHT' chosen, but model {model_name} does not have 'class_weight' attribute.")
            elif imbalance_method != 'NONE':
                 logger.warning(f"Unknown imbalance handling method: '{imbalance_method}'. Proceeding without specific imbalance handling for {model_name}.")


            pipeline_steps.append(('classifier', base_model_instance))
            model_pipeline = current_pipeline_class(pipeline_steps)

            current_param_grid = param_grids_config.get(model_name, {})
            if current_param_grid:
                mlflow.log_params({f"param_grid_{k}": v for k, v in current_param_grid.items()}) # Log param grid
            else:
                mlflow.log_param("param_grid", "default_model_params")


            strat_k_fold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
            
            logger.debug(f"Parameter grid for {model_name}: {current_param_grid}")
            
            # Setup GridSearchCV
            # Note: If current_param_grid is empty, GridSearchCV will train with default parameters of the estimator.
            grid_search = GridSearchCV(
                estimator=model_pipeline,
                param_grid=current_param_grid,
                scoring=f1_yes_scorer,
                cv=strat_k_fold,
                n_jobs=-1, # Use all available cores
                verbose=1  # Set to 0 for less verbosity, 1 or higher for more.
            )

            try:
                grid_search.fit(X_train, y_train)
                trained_model_pipelines[model_name] = grid_search.best_estimator_
                model_best_scores[model_name] = float(grid_search.best_score_) 
                mlflow.log_metric(f"cv_best_score_{config.get('hyperparameter_tuning', {}).get('scoring_metric_for_tuning', 'f1_yes_scorer')}", grid_search.best_score_)
                mlflow.log_params(grid_search.best_params_)
                predictions_proba = grid_search.best_estimator_.predict_proba(input_example)
                signature = infer_signature(input_example, predictions_proba)
                mlflow.sklearn.log_model(
                        sk_model=grid_search.best_estimator_,
                        artifact_path=model_name, # This creates a 'models/{model_name}' like structure in MLflow artifacts
                        input_example=input_example,
                        signature=signature,
                        pyfunc_predict_fn="predict_proba",
                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE 
                    )
                logger.info(f"Best F1-score (yes class) for {model_name} (from CV): {grid_search.best_score_:.4f}")
                logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                logger.info(f"Logged {model_name} model and params to MLflow.")
            except Exception as e:
                logger.error(f"Error during GridSearchCV for {model_name}: {e}", exc_info=True)
                trained_model_pipelines[model_name] = None # Indicate failure
                model_best_scores[model_name] = -1.0     # Indicate failure or very poor performance
                mlflow.log_param(f"{model_name}_training_status", "failed")
                mlflow.set_tag(f"{model_name}_error", str(e))

    if not trained_model_pipelines or not any(pipeline is not None for pipeline in trained_model_pipelines.values()):
        logger.error("No models were successfully trained. Cannot determine the best model.")
        return {}, "", None # Return empty dict, empty name, and None object

    # Determine overall best model from successfully trained models
    valid_scores = {name: score for name, score in model_best_scores.items() if trained_model_pipelines.get(name) is not None and score != -1.0}
    if not valid_scores:
        logger.error("No models have valid scores after training. Cannot determine best model.")
        return trained_model_pipelines, "", None

    best_model_name = max(valid_scores, key=valid_scores.get)
    overall_best_score = valid_scores[best_model_name]
    best_model_pipeline = trained_model_pipelines[best_model_name]

    logger.info("--- Overall Best Model Selection (based on F1 for 'yes' class during CV) ---")
    logger.info(f"Best Model Name: {best_model_name}")
    logger.info(f"Best Cross-Validated F1-score (yes class): {overall_best_score:.4f}")
    if best_model_pipeline:
        logger.debug(f"Best Model Pipeline details: {best_model_pipeline}")
    else: # Should not happen if best_model_name was derived from valid_scores/trained_pipelines
        logger.error(f"Inconsistency: Best model '{best_model_name}' found, but its pipeline is None.")
        return trained_model_pipelines, "", None


    return trained_model_pipelines, best_model_name, best_model_pipeline