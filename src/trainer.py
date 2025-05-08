# src/trainer.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, Any, Union, Optional, Tuple

import mlflow
import mlflow.sklearn # For logging sklearn models

# Assuming utils.py is in the same src directory
from utils import save_object # For the save_model method

import logging
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Manages the model training process, including hyperparameter tuning
    and cross-validation.
    """
    def __init__(self,
                 preprocessor: SklearnPipeline,
                 feature_engineer: SklearnPipeline,
                 model: ClassifierMixin,
                 model_name: str,
                 training_config: Dict[str, Any],
                 param_distributions: Optional[Dict[str, Any]] = None):
        """
        Initializes the ModelTrainer.

        Args:
            preprocessor (SklearnPipeline): The pre-built preprocessing pipeline.
            feature_engineer (SklearnPipeline): The pre-built feature engineering pipeline.
            model (ClassifierMixin): The model instance (e.g., from model_dispatcher).
            model_name (str): Name of the model (for logging purposes).
            training_config (Dict[str, Any]): Configuration for training, including
                CV settings, tuning method, scoring metric, etc.
                Example:
                {
                    "cv_type": "stratified_kfold",
                    "cv_folds": 5,
                    "cv_shuffle": True,
                    "random_seed": 42,
                    "hyperparameter_tuning": {
                        "method": "randomized_search_cv", // or "grid_search_cv" or "none"
                        "n_iter": 50, // For RandomizedSearchCV
                        "scoring_metric": "roc_auc"
                    }
                }
            param_distributions (Optional[Dict[str, Any]]): Parameter distributions for
                hyperparameter tuning (e.g., for RandomizedSearchCV).
                Keys should be prefixed with 'model__' (e.g., 'model__n_estimators').
                Required if tuning method is not 'none'.
        """
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.model = model
        self.model_name = model_name # For logging and identification
        self.training_config = training_config
        self.param_distributions = param_distributions if param_distributions else {}

        self.full_pipeline: Optional[SklearnPipeline] = None
        self.best_estimator_: Optional[SklearnPipeline] = None

    def _get_cv_strategy(self) -> Union[StratifiedKFold, KFold]:
        """
        Creates a cross-validation strategy object based on configuration.
        """
        cv_type = self.training_config.get("cv_type", "stratified_kfold").lower()
        folds = self.training_config.get("cv_folds", 5)
        shuffle = self.training_config.get("cv_shuffle", True)
        random_state = self.training_config.get("random_seed", None)

        if cv_type == "stratified_kfold":
            return StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
        elif cv_type == "kfold":
            return KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
        else:
            logger.warning(f"Unsupported CV type: {cv_type}. Defaulting to StratifiedKFold.")
            return StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> SklearnPipeline:
        """
        Constructs the full pipeline, performs hyperparameter tuning (if configured),
        trains the model, and logs results to MLflow.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.

        Returns:
            SklearnPipeline: The best trained (and possibly tuned) full pipeline.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        self.full_pipeline = SklearnPipeline([
            ('preprocessing', self.preprocessor),
            ('feature_engineering', self.feature_engineer),
            ('model', self.model) # The 'model' step for hyperparameter tuning
        ])
        logger.info(f"Full pipeline for {self.model_name} constructed: {self.full_pipeline.steps}")

        tuning_config = self.training_config.get("hyperparameter_tuning", {})
        tuning_method = tuning_config.get("method", "none").lower()
        scoring_metric = tuning_config.get("scoring_metric", "roc_auc") # Default scoring metric
        random_seed = self.training_config.get("random_seed", None)

        cv_strategy = self._get_cv_strategy()
        logger.info(f"Using CV strategy: {cv_strategy} with scoring metric: {scoring_metric}")

        mlflow.log_param("model_name", self.model_name)
        mlflow.log_param("cv_type", cv_strategy.__class__.__name__)
        mlflow.log_param("cv_folds", cv_strategy.get_n_splits())
        mlflow.log_param("cv_shuffle", self.training_config.get("cv_shuffle", True)) # logging actual config
        mlflow.log_param("tuning_method", tuning_method)
        mlflow.log_param("tuning_scoring_metric", scoring_metric)


        if tuning_method == "none":
            logger.info(f"No hyperparameter tuning specified for {self.model_name}. Fitting with default/provided parameters.")
            # Log initial model parameters if no tuning
            initial_model_params = {f"initial_{k}": v for k,v in self.model.get_params().items()}
            mlflow.log_params(initial_model_params)

            self.full_pipeline.fit(X_train, y_train)
            self.best_estimator_ = self.full_pipeline
            # Note: CV scores are not available without a CV-based tuner or manual CV.
            # Consider adding manual cross_val_score if 'none' but still want CV metrics.
            logger.info(f"{self.model_name} fitted without hyperparameter tuning.")

        elif tuning_method in ["randomized_search_cv", "grid_search_cv"]:
            if not self.param_distributions:
                logger.error(f"Parameter distributions are required for {tuning_method} but not provided.")
                raise ValueError(f"param_distributions missing for {tuning_method}")

            if tuning_method == "randomized_search_cv":
                n_iter = tuning_config.get("n_iter", 10) # Default n_iter for RandomizedSearchCV
                mlflow.log_param("tuning_n_iter", n_iter)
                tuner = RandomizedSearchCV(
                    estimator=self.full_pipeline,
                    param_distributions=self.param_distributions,
                    n_iter=n_iter,
                    cv=cv_strategy,
                    scoring=scoring_metric,
                    random_state=random_seed,
                    n_jobs=-1, # Use all available cores
                    verbose=1 # Or configurable
                )
                logger.info(f"Starting RandomizedSearchCV for {self.model_name} with {n_iter} iterations.")
            else: # grid_search_cv
                tuner = GridSearchCV(
                    estimator=self.full_pipeline,
                    param_grid=self.param_distributions, # param_grid for GridSearchCV
                    cv=cv_strategy,
                    scoring=scoring_metric,
                    n_jobs=-1,
                    verbose=1
                )
                logger.info(f"Starting GridSearchCV for {self.model_name}.")

            tuner.fit(X_train, y_train)

            self.best_estimator_ = tuner.best_estimator_
            best_params = {k.replace("model__", ""): v for k, v in tuner.best_params_.items()} # Clean prefix for logging
            
            logger.info(f"Best parameters found for {self.model_name}: {best_params}")
            logger.info(f"Best CV score ({scoring_metric}): {tuner.best_score_:.4f}")

            mlflow.log_params({f"best_param_{k}": v for k,v in best_params.items()})
            mlflow.log_metric(f"best_cv_{scoring_metric}", tuner.best_score_)

            # Log all CV results for more detailed analysis (optional)
            # cv_results_df = pd.DataFrame(tuner.cv_results_)
            # mlflow.log_text(cv_results_df.to_string(), "cv_results_summary.txt")
            # for i in range(cv_strategy.get_n_splits()):
            #    mlflow.log_metric(f"cv_split{i}_test_score", tuner.cv_results_[f"split{i}_test_score"][tuner.best_index_])

        else:
            # Placeholder for other tuning methods like Optuna, Hyperopt
            logger.error(f"Unsupported hyperparameter tuning method: {tuning_method}")
            raise ValueError(f"Unsupported tuning method: {tuning_method}")

        if self.best_estimator_ is None:
             logger.error(f"Model training failed for {self.model_name}, best_estimator_ is None.")
             raise RuntimeError(f"Training failed for {self.model_name}")

        logger.info(f"Training completed for {self.model_name}. Best estimator obtained.")
        return self.best_estimator_

    def save_trained_model(self, file_path: str) -> None:
        """
        Saves the best trained model pipeline to a file using joblib.

        Args:
            file_path (str): The path (including filename) to save the model.
        """
        if self.best_estimator_ is None:
            logger.error("No best_estimator_ found. Train the model first before saving.")
            raise RuntimeError("Model not trained yet. Call train() first.")
        
        try:
            save_object(self.best_estimator_, file_path)
            logger.info(f"Trained model for {self.model_name} saved to {file_path}")
            # Optionally log as MLflow artifact if not using mlflow.sklearn.log_model in main pipeline
            # mlflow.log_artifact(file_path, artifact_path="trained_model")
        except Exception as e:
            logger.error(f"Error saving model {self.model_name} to {file_path}: {e}")
            raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Starting ModelTrainer example...")

    # Dummy data for example
    X_dummy = pd.DataFrame({
        'num_feat1': np.random.rand(100),
        'num_feat2': np.random.rand(100) * 10,
        'cat_feat1': np.random.choice(['A', 'B', 'C'], size=100),
        'cat_feat2': np.random.choice(['X', 'Y', 'Z', np.nan], size=100)
    })
    y_dummy = pd.Series(np.random.randint(0, 2, size=100))

    # Dummy preprocessor and feature engineer (replace with actual ones)
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from preprocessing import NumericalImputer, CategoricalImputer # Assuming they are defined

    # Define a simple preprocessor pipeline
    numerical_features = ['num_feat1', 'num_feat2']
    categorical_features = ['cat_feat1', 'cat_feat2']

    # Preprocessing steps
    numerical_pipeline = SklearnPipeline([
        ('imputer', NumericalImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = SklearnPipeline([
        ('imputer', CategoricalImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor_ct = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ], remainder='passthrough' # or 'drop'
    )
    preprocessor_full_pipe = SklearnPipeline([('col_transformer', preprocessor_ct)])

    # Dummy feature engineering (identity for this example)
    feature_engineer_pipe = SklearnPipeline([]) # No actual feature engineering steps

    # Dummy model (Logistic Regression)
    from sklearn.linear_model import LogisticRegression
    dummy_model_instance = LogisticRegression(solver='liblinear', random_state=42)
    dummy_model_name = "logistic_regression_example"

    # Training configuration
    example_training_config = {
        "cv_type": "stratified_kfold",
        "cv_folds": 3, # Reduced for quick example
        "cv_shuffle": True,
        "random_seed": 42,
        "hyperparameter_tuning": {
            "method": "randomized_search_cv",
            "n_iter": 5, # Reduced for quick example
            "scoring_metric": "accuracy" # Using accuracy for simplicity here
        }
    }
    example_param_dist = {
        'model__C': [0.1, 1, 10], # Note the 'model__' prefix
        'model__penalty': ['l1', 'l2']
    }

    # MLflow setup for example (usually done in a main script)
    # For local testing, MLflow will create an 'mlruns' directory.
    mlflow.set_experiment("ModelTrainer_Example_Experiment")
    
    with mlflow.start_run(run_name=f"Train_{dummy_model_name}") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_param("example_run", "True")

        try:
            trainer = ModelTrainer(
                preprocessor=preprocessor_full_pipe,
                feature_engineer=feature_engineer_pipe,
                model=dummy_model_instance,
                model_name=dummy_model_name,
                training_config=example_training_config,
                param_distributions=example_param_dist
            )

            logger.info("Training the dummy model...")
            best_pipeline = trainer.train(X_dummy, y_dummy)
            logger.info(f"Best pipeline obtained: {best_pipeline}")
            logger.info(f"Best model parameters: {best_pipeline.named_steps['model'].get_params()}")

            # Example of saving the model
            model_save_path = f"{dummy_model_name}_best_pipeline.joblib"
            trainer.save_trained_model(model_save_path)
            logger.info(f"Model saved to {model_save_path}. Check your local directory.")
            # Log the model using MLflow's sklearn integration
            mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                artifact_path=f"model_{dummy_model_name}",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            )
            logger.info(f"Model also logged to MLflow artifact store under 'model_{dummy_model_name}'.")


        except Exception as e:
            logger.error(f"An error occurred during the ModelTrainer example: {e}", exc_info=True)
            mlflow.set_tag("mlflow.runName", f"FAILED_Train_{dummy_model_name}") # Mark run as failed

    logger.info("ModelTrainer example finished.")