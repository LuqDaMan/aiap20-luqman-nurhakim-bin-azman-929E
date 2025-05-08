# src/predict.py
import os
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin # The loaded model is expected to be a classifier pipeline
from typing import Any, Optional, List

# Assuming utils.py is in the same src directory
from utils import load_object # For loading the serialized model pipeline

import logging
logger = logging.getLogger(__name__)

class PredictionService:
    """
    Handles loading a trained model pipeline and making predictions on new data.
    """
    def __init__(self, model_path: str):
        """
        Initializes the PredictionService by loading the serialized model pipeline.

        Args:
            model_path (str): The path to the serialized model pipeline file
                              (e.g., created by ModelTrainer.save_trained_model).
        
        Raises:
            FileNotFoundError: If the model_path does not exist.
            Exception: If there's an error loading the model.
        """
        self.model_path = model_path
        self.model_pipeline: Optional[ClassifierMixin] = None
        self._load_model()
        
        # Store expected feature names if available from the pipeline
        self.expected_input_features_: Optional[List[str]] = None
        if self.model_pipeline and hasattr(self.model_pipeline, 'feature_names_in_'):
            self.expected_input_features_ = list(self.model_pipeline.feature_names_in_)
            logger.info(f"Model pipeline expects input features: {self.expected_input_features_}")
        elif self.model_pipeline and hasattr(self.model_pipeline, 'steps'):
            # Try to get from the first step if it's a ColumnTransformer or similar
            first_step_transformer = self.model_pipeline.steps[0][1]
            if hasattr(first_step_transformer, 'feature_names_in_'):
                 self.expected_input_features_ = list(first_step_transformer.feature_names_in_)
                 logger.info(f"First pipeline step expects input features: {self.expected_input_features_}")
            elif hasattr(first_step_transformer, 'transformers') and first_step_transformer.transformers:
                # Heuristic for ColumnTransformer: collect all input columns from its transformers
                # This might not be perfectly robust for all ColumnTransformer configurations
                all_cols = set()
                for _, _, cols in first_step_transformer.transformers_:
                    if isinstance(cols, (list, tuple)):
                        all_cols.update(c for c in cols if isinstance(c, str))
                    elif isinstance(cols, str):
                        all_cols.add(cols)
                if all_cols:
                    self.expected_input_features_ = sorted(list(all_cols))
                    logger.info(f"Inferred expected input features from ColumnTransformer: {self.expected_input_features_}")


    def _load_model(self) -> None:
        """Loads the model pipeline from the specified path."""
        logger.info(f"Loading model pipeline from: {self.model_path}")
        try:
            self.model_pipeline = load_object(self.model_path)
            if self.model_pipeline is None:
                raise ValueError("Loaded model is None. Check the model file and load_object utility.")
            if not (hasattr(self.model_pipeline, 'predict') and hasattr(self.model_pipeline, 'predict_proba')):
                 logger.warning("Loaded model pipeline does not have predict/predict_proba methods. Ensure it's a scikit-learn compatible classifier pipeline.")
                 # For this project, we expect a classifier pipeline.
                 # If it's just a transformer, it wouldn't be suitable here directly.
            logger.info("Model pipeline loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model file not found at path: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}", exc_info=True)
            raise

    def _validate_input_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs basic validation on the input DataFrame.
        The loaded scikit-learn pipeline will perform more detailed structural
        and value-based validation internally based on how it was trained.
        """
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        
        if input_data.empty:
            raise ValueError("Input data DataFrame is empty.")

        # Basic check for expected columns if schema was inferred
        if self.expected_input_features_:
            missing_cols = [col for col in self.expected_input_features_ if col not in input_data.columns]
            if missing_cols:
                logger.warning(f"Input data is missing expected columns: {missing_cols}. "
                                "The pipeline might fail if these are critical and not handled by imputation of entire columns.")
                # Depending on pipeline design, this might be an error or handled.
                # For now, we issue a warning and let the pipeline try.
                # raise ValueError(f"Input data is missing expected columns: {missing_cols}")
            
            # Optional: Reorder columns to match expected order if known (can prevent some subtle issues)
            # This is usually handled well if ColumnTransformer used named columns.
            # If order matters critically and isn't handled by pipeline:
            # try:
            #     input_data = input_data[self.expected_input_features_]
            # except KeyError:
            #     logger.error("Could not reorder columns due to missing columns. This should have been caught earlier.")
            #     # This would re-raise if missing_cols check above was made an error
        
        return input_data.copy() # Return a copy to avoid modifying original DataFrame


    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions on the new input data using the loaded model pipeline.

        Args:
            input_data (pd.DataFrame): New data for prediction. Its schema should
                                       match the raw input schema expected by the
                                       trained pipeline.

        Returns:
            np.ndarray: Prediction results (e.g., class labels).
        
        Raises:
            RuntimeError: If the model is not loaded.
            TypeError: If input_data is not a DataFrame.
            ValueError: If input_data is empty or has critical issues.
            Exception: For errors during the prediction process by the pipeline.
        """
        if self.model_pipeline is None:
            logger.error("Model pipeline is not loaded. Cannot make predictions.")
            raise RuntimeError("Model not loaded. Initialize PredictionService with a valid model_path.")

        logger.info(f"Received {len(input_data)} records for prediction.")
        validated_data = self._validate_input_data(input_data)
        
        try:
            logger.debug("Calling predict method on the loaded model pipeline...")
            predictions = self.model_pipeline.predict(validated_data)
            logger.info(f"Successfully generated {len(predictions)} predictions.")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            # This could be due to data issues not caught by basic validation but problematic for pipeline steps.
            raise

    def predict_proba(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Makes probability predictions on the new input data.

        Args:
            input_data (pd.DataFrame): New data for prediction.

        Returns:
            np.ndarray: Prediction probabilities for each class.

        Raises:
            RuntimeError: If the model is not loaded or doesn't support predict_proba.
            TypeError: If input_data is not a DataFrame.
            ValueError: If input_data is empty or has critical issues.
            Exception: For errors during the prediction process.
        """
        if self.model_pipeline is None:
            logger.error("Model pipeline is not loaded. Cannot make probability predictions.")
            raise RuntimeError("Model not loaded.")
        
        if not hasattr(self.model_pipeline, 'predict_proba'):
            logger.error("Loaded model pipeline does not have a 'predict_proba' method.")
            raise AttributeError("Model pipeline does not support predict_proba.")

        logger.info(f"Received {len(input_data)} records for probability prediction.")
        validated_data = self._validate_input_data(input_data)

        try:
            logger.debug("Calling predict_proba method on the loaded model pipeline...")
            probabilities = self.model_pipeline.predict_proba(validated_data)
            logger.info(f"Successfully generated probabilities for {len(probabilities)} records.")
            return probabilities
        except Exception as e:
            logger.error(f"Error during probability prediction: {e}", exc_info=True)
            raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Starting PredictionService example...")

    # --- Setup for Example ---
    # This example assumes a model has been trained and saved by ModelTrainer.
    # We'll create a dummy model pipeline and save it for this example to work standalone.
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from utils import save_object # Assuming utils is in sibling directory for direct run
    
    dummy_model_file = "dummy_trained_pipeline.joblib"

    # Create and save a dummy pipeline
    try:
        # A very simple pipeline for demonstration: only scales one feature and then logistic regression
        # In reality, this would be the complex pipeline from your ModelTrainer
        example_data_for_fit = pd.DataFrame({'feature1': np.random.rand(10), 'feature2': np.random.rand(10)})
        example_target_for_fit = pd.Series(np.random.randint(0, 2, 10))

        # Define a simple ColumnTransformer (even if it uses only one feature type for simplicity)
        from sklearn.compose import ColumnTransformer
        ct = ColumnTransformer([('scaler', StandardScaler(), ['feature1'])], remainder='passthrough')
        
        dummy_pipeline = SklearnPipeline([
            ('preprocessor', ct), # A minimal preprocessor
            ('classifier', LogisticRegression(solver='liblinear'))
        ])
        dummy_pipeline.fit(example_data_for_fit, example_target_for_fit)
        save_object(dummy_pipeline, dummy_model_file)
        logger.info(f"Dummy model pipeline saved to {dummy_model_file} for example.")
    except Exception as e:
        logger.error(f"Could not create/save dummy model for example: {e}. The rest of the example might fail.")
        dummy_pipeline = None # Ensure it's None if saving failed

    if os.path.exists(dummy_model_file):
        # 1. Initialize PredictionService
        try:
            prediction_service = PredictionService(model_path=dummy_model_file)
            logger.info("PredictionService initialized successfully.")

            # 2. Prepare sample new data for prediction
            # Schema should match what the dummy_pipeline expects at its input
            # (i.e., before its 'preprocessor' step)
            new_data = pd.DataFrame({
                'feature1': [0.1, 0.5, 0.9, -0.2],
                'feature2': [10, 20, 5, 15], # feature2 will be passed through by ColumnTransformer's remainder
                'extra_column_not_used': ['a', 'b', 'c', 'd'] # Should be ignored if remainder='passthrough' or 'drop' and not selected
            })
            logger.info(f"\nSample new data for prediction:\n{new_data}")

            # 3. Make predictions
            predictions = prediction_service.predict(new_data)
            logger.info(f"\nPredictions:\n{predictions}")

            # 4. Make probability predictions
            if hasattr(prediction_service.model_pipeline, 'predict_proba'):
                probabilities = prediction_service.predict_proba(new_data)
                logger.info(f"\nProbabilities:\n{probabilities}")
            else:
                logger.info("Model does not support predict_proba.")

            # Example with missing critical column (if `expected_input_features_` was set robustly)
            new_data_missing_col = pd.DataFrame({'feature2': [11, 22]})
            logger.info(f"\nSample new data with missing 'feature1':\n{new_data_missing_col}")
            try:
                preds_missing = prediction_service.predict(new_data_missing_col)
                logger.info(f"Predictions with missing col (pipeline might have failed or handled it): {preds_missing}")
            except Exception as e:
                logger.warning(f"Caught expected error or pipeline failure for data with missing column: {e}")
                
        except Exception as e:
            logger.error(f"An error occurred during the PredictionService example: {e}", exc_info=True)
        finally:
            # Clean up the dummy model file
            if os.path.exists(dummy_model_file):
                try:
                    os.remove(dummy_model_file)
                    logger.info(f"Cleaned up dummy model file: {dummy_model_file}")
                except Exception as e_del:
                    logger.warning(f"Could not delete dummy model file {dummy_model_file}: {e_del}")
    else:
        logger.warning(f"Dummy model file {dummy_model_file} was not created. Skipping PredictionService example execution.")
        
    logger.info("PredictionService example finished.")