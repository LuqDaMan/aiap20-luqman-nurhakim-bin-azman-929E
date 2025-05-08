# src/evaluator.py
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin # The model is expected to be a classifier pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.inspection import PartialDependenceDisplay # For PDP plots

import matplotlib.pyplot as plt
import seaborn as sns # For confusion matrix plotting
import shap # For SHAP plots
import mlflow
import os # For creating directories and joining paths

# Type hinting
from typing import Dict, Any, List, Optional, Union, Tuple

import logging
logger = logging.getLogger(__name__)

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class ModelEvaluator:
    """
    Evaluates a trained model pipeline on a test set and generates
    interpretability reports.
    """
    def __init__(self,
                 model_pipeline: ClassifierMixin, # This is the fitted Scikit-learn pipeline
                 eval_config: Dict[str, Any]):
        """
        Initializes the ModelEvaluator.

        Args:
            model_pipeline (ClassifierMixin): The trained scikit-learn compatible model pipeline.
            eval_config (Dict[str, Any]): Configuration for evaluation.
                Example:
                {
                    "metrics": ["accuracy", "roc_auc", "f1_macro", ...],
                    "positive_label": 1, # For binary classification metrics if not default
                    "interpretability": {
                        "shap_summary_plot": {
                            "enabled": True,
                            "plot_path": "interpretability/shap_summary.png",
                            "max_display_shap": 15 # Number of features to display in SHAP summary
                        },
                        "pdp_plots": {
                            "enabled": True,
                            "features": ["feature1", "feature2"], # Original feature names
                            "plot_path_template": "interpretability/pdp_plot_{feature}.png"
                        },
                        "confusion_matrix_plot": {
                            "enabled": True,
                            "plot_path": "interpretability/confusion_matrix.png"
                        }
                    },
                    "artifact_base_path": "evaluation_artifacts" // Base for saving plots before logging
                }
        """
        if not hasattr(model_pipeline, 'predict') or not hasattr(model_pipeline, 'predict_proba'):
            raise ValueError("Provided model_pipeline must have 'predict' and 'predict_proba' methods.")
        self.model_pipeline = model_pipeline
        self.eval_config = eval_config
        self.artifact_base_path = self.eval_config.get("artifact_base_path", "evaluation_artifacts")
        
        # Create artifact directory if it doesn't exist
        if not os.path.exists(self.artifact_base_path):
            os.makedirs(self.artifact_base_path, exist_ok=True)
            logger.info(f"Created artifact directory: {self.artifact_base_path}")


    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Makes predictions on the test set, calculates evaluation metrics,
        and logs them to MLflow.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target variable.

        Returns:
            Dict[str, float]: A dictionary of calculated evaluation metrics.
        """
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test must be a pandas DataFrame.")
        if not isinstance(y_test, pd.Series):
            raise TypeError("y_test must be a pandas Series.")

        logger.info("Making predictions on the test set...")
        try:
            y_pred = self.model_pipeline.predict(X_test)
            y_pred_proba = self.model_pipeline.predict_proba(X_test)
        except Exception as e:
            logger.error(f"Error during prediction on test set: {e}", exc_info=True)
            raise

        # Assuming binary classification for proba, take probability of positive class
        # The positive class index depends on model.classes_
        positive_class_idx = 1
        if hasattr(self.model_pipeline, 'classes_'):
            classes = self.model_pipeline.classes_
            if len(classes) == 2: # Binary classification
                 # Find index of the positive label specified in config, or default to 1
                positive_label = self.eval_config.get("positive_label", 1)
                try:
                    positive_class_idx = list(classes).index(positive_label)
                except ValueError:
                    logger.warning(f"Positive label {positive_label} not in model classes {classes}. Defaulting to class 1 index if binary.")
                    positive_class_idx = 1 # Fallback, assuming classes are [0, 1] or similar
            else: # Multiclass
                 logger.info(f"Multiclass classification detected (classes: {classes}). ROC AUC and PR AUC might behave differently or require specific averaging.")
        
        y_pred_proba_positive = y_pred_proba[:, positive_class_idx] if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1 else y_pred_proba


        metrics_to_calc = self.eval_config.get("metrics", ["accuracy", "roc_auc", "f1_macro"])
        pos_label_for_metrics = self.eval_config.get("positive_label", 1) # For precision, recall, f1 specific to positive class
        
        results: Dict[str, float] = {}
        logger.info(f"Calculating evaluation metrics for positive_label='{pos_label_for_metrics}': {metrics_to_calc}")

        if "accuracy" in metrics_to_calc:
            results["accuracy"] = accuracy_score(y_test, y_pred)
        if "roc_auc" in metrics_to_calc:
            try:
                results["roc_auc"] = roc_auc_score(y_test, y_pred_proba_positive)
            except ValueError as e: # Handles cases like only one class present in y_test
                logger.warning(f"Could not calculate ROC AUC: {e}. Setting to NaN.")
                results["roc_auc"] = np.nan
        if "pr_auc" in metrics_to_calc: # Precision-Recall AUC
             try:
                results["pr_auc"] = average_precision_score(y_test, y_pred_proba_positive, pos_label=pos_label_for_metrics)
             except ValueError as e:
                logger.warning(f"Could not calculate PR AUC: {e}. Setting to NaN.")
                results["pr_auc"] = np.nan

        # Metrics specific to positive class (assuming binary or one-vs-rest context)
        # For multiclass, 'average' parameter would be important (e.g., 'macro', 'micro', 'weighted')
        # The EDA indicates focus on positive class (subscription).
        if "f1_positive" in metrics_to_calc:
            results["f1_positive"] = f1_score(y_test, y_pred, pos_label=pos_label_for_metrics, zero_division=0)
        if "precision_positive" in metrics_to_calc:
            results["precision_positive"] = precision_score(y_test, y_pred, pos_label=pos_label_for_metrics, zero_division=0)
        if "recall_positive" in metrics_to_calc:
            results["recall_positive"] = recall_score(y_test, y_pred, pos_label=pos_label_for_metrics, zero_division=0)
        
        # General F1/Precision/Recall (e.g., macro averaged for multiclass or if specified for binary)
        if "f1_macro" in metrics_to_calc:
            results["f1_macro"] = f1_score(y_test, y_pred, average='macro', zero_division=0)
        if "precision_macro" in metrics_to_calc:
            results["precision_macro"] = precision_score(y_test, y_pred, average='macro', zero_division=0)
        if "recall_macro" in metrics_to_calc:
            results["recall_macro"] = recall_score(y_test, y_pred, average='macro', zero_division=0)

        logger.info(f"Calculated test set metrics: {results}")
        mlflow.log_metrics({f"test_{k}": v for k, v in results.items() if not np.isnan(v)})

        # Log classification report as text
        try:
            report = classification_report(y_test, y_pred, zero_division=0)
            logger.info(f"Classification Report:\n{report}")
            mlflow.log_text(report, "test_classification_report.txt")
        except Exception as e:
            logger.warning(f"Could not generate or log classification report: {e}")


        # --- Generate Interpretability Artifacts ---
        interpretability_cfg = self.eval_config.get("interpretability", {})
        
        if interpretability_cfg.get("confusion_matrix_plot", {}).get("enabled", False):
            self.generate_confusion_matrix_plot(y_test, y_pred, interpretability_cfg["confusion_matrix_plot"])

        if interpretability_cfg.get("shap_summary_plot", {}).get("enabled", False):
            self.generate_shap_summary_plot(X_test, interpretability_cfg["shap_summary_plot"])
        
        if interpretability_cfg.get("pdp_plots", {}).get("enabled", False):
            pdp_cfg = interpretability_cfg["pdp_plots"]
            self.generate_pdp_plots(X_test, pdp_cfg.get("features", []), pdp_cfg)
            
        # LIME is instance-specific and can be more complex to integrate generically here.
        # If needed, a separate method or script might handle LIME for specific instances.

        return results

    def _get_transformed_data_and_model(self, X_original: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        """
        Helper to get the data transformed by all steps except the final model,
        and the final model itself.
        """
        if not hasattr(self.model_pipeline, 'steps'):
            logger.warning("Model pipeline does not have 'steps'. SHAP/PDP might not work as expected if not a scikit-learn Pipeline.")
            # If not a Pipeline, assume model_pipeline IS the final model and X_original is already transformed (less likely for this project)
            return X_original, self.model_pipeline

        # Transform data using all steps except the last one (the model)
        if len(self.model_pipeline.steps) > 1:
            transformer_pipeline = SklearnPipeline(self.model_pipeline.steps[:-1])
            X_transformed = transformer_pipeline.transform(X_original)
            final_model_estimator = self.model_pipeline.steps[-1][1]
        else: # Pipeline only has the model step
            X_transformed = X_original # Assume X is already processed
            final_model_estimator = self.model_pipeline.steps[0][1]
        
        # If X_transformed is numpy, convert to DataFrame for SHAP/PDP if feature names are needed by explainers
        # This requires feature names from the last transformation step
        try:
            # Attempt to get feature names from the last transformer in the preprocessing/feature engineering part
            # This can be complex if the transformer pipeline has nested ColumnTransformers, etc.
            # For simplicity, we assume the output of the transformer_pipeline is a DataFrame or can be converted.
            # If it's a NumPy array, feature names might be lost for some SHAP explainers.
            if isinstance(X_transformed, np.ndarray):
                # Try to get feature names from the last step of the transformer pipeline
                last_transformer_step = transformer_pipeline.steps[-1][1]
                if hasattr(last_transformer_step, 'get_feature_names_out'):
                    feature_names = last_transformer_step.get_feature_names_out()
                    X_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X_original.index)
                else:
                    logger.warning("Transformed data is a NumPy array and feature names could not be retrieved. SHAP plots might use generic feature names.")
                    X_transformed = pd.DataFrame(X_transformed, index=X_original.index, columns=[f"feature_{i}" for i in range(X_transformed.shape[1])])
        except Exception as e:
            logger.warning(f"Could not get feature names for transformed data. Using generic names if needed. Error: {e}")
            if isinstance(X_transformed, np.ndarray): # Ensure it's a DataFrame
                 X_transformed = pd.DataFrame(X_transformed, index=X_original.index, columns=[f"feature_{i}" for i in range(X_transformed.shape[1])])


        return X_transformed, final_model_estimator

    def generate_confusion_matrix_plot(self, y_true: pd.Series, y_pred: pd.Series, cm_config: Dict[str, Any]):
        """Generates and logs a confusion matrix plot."""
        plot_path_rel = cm_config.get("plot_path", "interpretability/confusion_matrix.png")
        plot_path_abs = os.path.join(self.artifact_base_path, plot_path_rel)
        os.makedirs(os.path.dirname(plot_path_abs), exist_ok=True)

        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=self.model_pipeline.classes_ if hasattr(self.model_pipeline, 'classes_') else ['0','1'],
                        yticklabels=self.model_pipeline.classes_ if hasattr(self.model_pipeline, 'classes_') else ['0','1'])
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            plt.savefig(plot_path_abs)
            plt.close()
            mlflow.log_artifact(plot_path_abs, os.path.dirname(plot_path_rel) or ".") # Log to interpretability/ or root
            logger.info(f"Confusion matrix plot saved to {plot_path_abs} and logged to MLflow.")
        except Exception as e:
            logger.error(f"Failed to generate or log confusion matrix plot: {e}", exc_info=True)


    def generate_shap_summary_plot(self, X_test_original: pd.DataFrame, shap_config: Dict[str, Any]):
        """
        Generates SHAP summary plot and logs it.
        X_test_original is the data BEFORE any transformations by the model_pipeline.
        """
        plot_path_rel = shap_config.get("plot_path", "interpretability/shap_summary.png")
        plot_path_abs = os.path.join(self.artifact_base_path, plot_path_rel)
        os.makedirs(os.path.dirname(plot_path_abs), exist_ok=True)
        max_display = shap_config.get("max_display_shap", 15)

        logger.info("Generating SHAP summary plot...")
        try:
            X_test_transformed, final_model = self._get_transformed_data_and_model(X_test_original)
            
            # SHAP explainer selection based on model type
            if isinstance(final_model, (xgb.XGBModel, RandomForestClassifier, GradientBoostingClassifier)): # Add other tree-based models if used
                explainer = shap.TreeExplainer(final_model, feature_perturbation="tree_path_dependent")
            elif isinstance(final_model, LogisticRegression) or isinstance(final_model, SVC):
                 # For linear models and SVMs, KernelExplainer is often used. It can be slow.
                 # Masking with a subset of data for KernelExplainer background improves performance.
                 # Ensure X_test_transformed is a DataFrame for feature names.
                 if not isinstance(X_test_transformed, pd.DataFrame):
                     X_test_transformed = pd.DataFrame(X_test_transformed)

                 # Using a subset of transformed data as background for KernelExplainer
                 # shap.sample can create a weighted sample.
                 # Using X_test_transformed directly can be very slow if large.
                 # Let's try to use shap.maskers.Independent for tabular data.
                 # Need to properly handle masker for KernelExplainer
                 med = X_test_transformed.median().values.reshape(1, -1)
                 X_background = X_test_transformed.iloc[:100] if len(X_test_transformed) > 100 else X_test_transformed
                 # The KernelExplainer expects data directly, not a masker
                 explainer = shap.KernelExplainer(final_model.predict_proba, X_background)
                 # Alternatively, if final_model is linear: explainer = shap.LinearExplainer(final_model, X_test_transformed)

            else: # Fallback to generic explainer if model type not specifically handled
                logger.warning(f"Using generic shap.Explainer for model type {type(final_model)}. This might be slow or require specific masker.")
                # It's often better to handle specific model types.
                # The generic explainer tries to pick the best one.
                explainer = shap.Explainer(final_model.predict_proba, X_test_transformed)

            shap_values = explainer(X_test_transformed) # For KernelExplainer with predict_proba output, this can be a list of arrays (one per class)

            # For binary classification from predict_proba, shap_values might be a list [shap_values_class0, shap_values_class1]
            # Or it could be an Explanation object with .values attribute.
            # We typically plot SHAP values for the positive class.
            
            shap_values_for_plot = shap_values
            if isinstance(shap_values, list) and len(shap_values) > 1: # Multi-output (e.g., predict_proba)
                positive_class_idx_shap = 1 # Assuming positive class is index 1
                shap_values_for_plot = shap_values[positive_class_idx_shap]
            elif hasattr(shap_values, 'values') and isinstance(shap_values.values, np.ndarray) and shap_values.values.ndim == X_test_transformed.ndim +1 :
                # This is common for Explanation objects from newer SHAP versions for multi-class
                positive_class_idx_shap = 1
                shap_values_for_plot = shap_values[..., positive_class_idx_shap]


            plt.figure() # Ensure a new figure context for SHAP plot
            # If shap_values_for_plot is an Explanation object, it might directly support plotting:
            if hasattr(shap_values_for_plot, 'base_values'): # Checks if it's likely an Explanation object
                 shap.summary_plot(shap_values_for_plot, X_test_transformed, plot_type="bar", max_display=max_display, show=False)
            else: # If it's a raw numpy array of SHAP values
                 shap.summary_plot(shap_values_for_plot, X_test_transformed, plot_type="bar", max_display=max_display, show=False)

            plt.tight_layout()
            plt.savefig(plot_path_abs)
            plt.close()
            mlflow.log_artifact(plot_path_abs, os.path.dirname(plot_path_rel) or ".")
            logger.info(f"SHAP summary plot saved to {plot_path_abs} and logged to MLflow.")

        except Exception as e:
            logger.error(f"Failed to generate or log SHAP summary plot: {e}", exc_info=True)

    def generate_pdp_plots(self, X_test_original: pd.DataFrame, features_for_pdp: List[Union[str, int, Tuple[Union[str, int], ...]]], pdp_config: Dict[str, Any]):
        """
        Generates Partial Dependence Plots (PDP) for specified features.
        X_test_original is the data BEFORE transformations by the model_pipeline.
        """
        plot_path_template_rel = pdp_config.get("plot_path_template", "interpretability/pdp_plot_{feature}.png")
        
        logger.info(f"Generating PDP plots for features: {features_for_pdp}...")
        if not features_for_pdp:
            logger.warning("No features specified for PDP plots.")
            return

        try:
            # PDP typically works well with scikit-learn pipelines directly
            # The 'features' argument should refer to original column names or indices if X is original
            # If X_test_original has categorical features not yet encoded by the pipeline, PDP might struggle
            # It's generally safer if features_for_pdp are those that are numeric after initial pipeline steps,
            # or if the pipeline handles them robustly for PDP.
            # For `from_estimator`, features can be names if X_test_original is a DataFrame with these columns.
            
            # Check if features are valid for X_test_original before transformation
            valid_features_for_pdp = []
            for f_spec in features_for_pdp:
                if isinstance(f_spec, (str, int)):
                    if isinstance(f_spec, str) and f_spec not in X_test_original.columns:
                        logger.warning(f"Feature '{f_spec}' for PDP not in X_test_original columns. Skipping.")
                        continue
                    valid_features_for_pdp.append(f_spec)
                elif isinstance(f_spec, tuple): # Interaction PDP
                    valid_pair = True
                    for f_item in f_spec:
                         if isinstance(f_item, str) and f_item not in X_test_original.columns:
                            logger.warning(f"Feature '{f_item}' in pair {f_spec} for PDP not in X_test_original columns. Skipping pair.")
                            valid_pair = False
                            break
                    if valid_pair:
                        valid_features_for_pdp.append(f_spec)
            
            if not valid_features_for_pdp:
                 logger.warning("No valid features found for PDP plots after checking against X_test_original.")
                 return

            # `from_estimator` handles the necessary transformations if `model_pipeline` is a Pipeline
            # And X is the original, untransformed data.
            display = PartialDependenceDisplay.from_estimator(
                self.model_pipeline,
                X_test_original,
                features=valid_features_for_pdp,
                kind='average', # 'average' for individual PDPs, 'both' for ICE and PDP
                # For binary classification, PDP is usually for the positive class probability
                # `response_method='predict_proba'` and targeting the positive class often makes sense
                # `target` param is not directly available in from_estimator, but it uses predict_decfunc or predict_proba
            )
            
            # The display object might contain multiple axes if multiple features are plotted.
            # We save each one or the combined figure. `display.plot()` returns the axes.
            # For simplicity, let's assume from_estimator creates a figure we can save.
            fig_ = display.figure_
            plot_path_rel_combined = plot_path_template_rel.replace("_{feature}", "_combined")
            plot_path_abs_combined = os.path.join(self.artifact_base_path, plot_path_rel_combined)
            os.makedirs(os.path.dirname(plot_path_abs_combined), exist_ok=True)

            fig_.suptitle("Partial Dependence Plots", fontsize=16)
            fig_.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
            fig_.savefig(plot_path_abs_combined)
            plt.close(fig_)
            mlflow.log_artifact(plot_path_abs_combined, os.path.dirname(plot_path_rel_combined) or ".")
            logger.info(f"Combined PDP plot saved to {plot_path_abs_combined} and logged to MLflow.")

            # If you want individual plots (more complex to manage here if display.plot() handles all)
            # You might need to iterate and call display.plot(features=[single_feature_spec], ...)

        except Exception as e:
            logger.error(f"Failed to generate or log PDP plots: {e}", exc_info=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Starting ModelEvaluator example...")

    # --- Setup for Example ---
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from preprocessing import NumericalImputer, CategoricalImputer 

    # Dummy data
    X_dummy_test = pd.DataFrame({
        'age': np.random.randint(20, 60, 100),
        'balance': np.random.normal(5000, 2000, 100),
        'duration': np.random.randint(50, 1000, 100),
        'job': np.random.choice(['admin.', 'technician', 'services', 'management', 'retired', 'student'], size=100),
        'education': np.random.choice(['secondary', 'tertiary', 'primary', 'unknown'], size=100),
        'cat_feat_nan': np.random.choice(['X', 'Y', np.nan], size=100)
    })
    X_dummy_test['balance'] = X_dummy_test['balance'].clip(lower=0) # Ensure balance is non-negative
    y_dummy_test = pd.Series(np.random.randint(0, 2, size=100))

    # Dummy fitted pipeline (mimicking output from ModelTrainer)
    numerical_features_ex = ['age', 'balance', 'duration']
    categorical_features_ex = ['job', 'education', 'cat_feat_nan']

    # Using transformers that should be defined in your preprocessing.py
    # For this example, using sklearn's directly for simplicity if not available
    num_pipe = SklearnPipeline([('imputer', NumericalImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = SklearnPipeline([('imputer', CategoricalImputer(strategy='most_frequent')), 
                                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor_ct_ex = ColumnTransformer(
        transformers=[
            ('num', num_pipe, numerical_features_ex),
            ('cat', cat_pipe, categorical_features_ex)
        ], remainder='drop'
    )
    
    # Create a dummy Logistic Regression model and fit it within a pipeline
    # This model needs to be 'trained' on some data to be evaluable
    dummy_model = LogisticRegression(solver='liblinear', random_state=42)
    # For a proper example, this pipeline should be fitted on training data first.
    # Here, we'll fit it on the dummy test data for demonstration purposes ONLY.
    # In a real scenario, it would be already fitted.
    
    # Create dummy training data to fit the pipeline for example
    X_dummy_train = X_dummy_test.sample(frac=0.8, random_state=42) # Not really training data
    y_dummy_train = y_dummy_test.loc[X_dummy_train.index]

    fitted_pipeline_example = SklearnPipeline([
        ('preprocessor', preprocessor_ct_ex),
        ('model', dummy_model)
    ])
    try:
        fitted_pipeline_example.fit(X_dummy_train, y_dummy_train)
        logger.info("Dummy pipeline fitted for evaluator example.")
    except Exception as e:
        logger.error(f"Could not fit dummy pipeline for example: {e}")
        # If fitting fails, evaluator might not run correctly.
        # This can happen if NumericalImputer/CategoricalImputer are not correctly found.
        # For standalone running, ensure these imports work or mock them.
        # Let's use sklearn's SimpleImputer if custom ones fail for the example run.
        from sklearn.impute import SimpleImputer
        num_pipe_fallback = SklearnPipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        cat_pipe_fallback = SklearnPipeline([('imputer', SimpleImputer(strategy='most_frequent')), 
                                             ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        preprocessor_ct_fallback = ColumnTransformer(
            transformers=[('num', num_pipe_fallback, numerical_features_ex), ('cat', cat_pipe_fallback, categorical_features_ex)], 
            remainder='drop'
        )
        fitted_pipeline_example = SklearnPipeline([('preprocessor', preprocessor_ct_fallback), ('model', dummy_model)])
        try:
             fitted_pipeline_example.fit(X_dummy_train, y_dummy_train)
             logger.info("Dummy pipeline (with fallback imputers) fitted for evaluator example.")
        except Exception as e_fallback:
            logger.error(f"Fallback dummy pipeline fitting failed: {e_fallback}. Evaluator example may not run.")
            fitted_pipeline_example = None # Mark as None if fitting fails

    # Evaluation configuration
    example_eval_config = {
        "metrics": ["accuracy", "roc_auc", "f1_positive", "precision_positive", "recall_positive", "pr_auc"],
        "positive_label": 1,
        "interpretability": {
            "shap_summary_plot": {"enabled": True, "max_display_shap": 10}, # plot_path will use default
            "pdp_plots": {"enabled": True, "features": ['age', ('age', 'balance'), 'duration']}, # plot_path_template will use default
            "confusion_matrix_plot": {"enabled": True} # plot_path will use default
        },
        "artifact_base_path": "example_evaluation_artifacts" # For local saving
    }

    if fitted_pipeline_example:
        # MLflow setup for example
        mlflow.set_experiment("ModelEvaluator_Example_Experiment")
        with mlflow.start_run(run_name="Evaluate_ExampleModel") as run:
            logger.info(f"MLflow Run ID for Evaluator example: {run.info.run_id}")
            mlflow.log_param("evaluator_example_run", "True")
            try:
                evaluator = ModelEvaluator(
                    model_pipeline=fitted_pipeline_example,
                    eval_config=example_eval_config
                )
                test_metrics = evaluator.evaluate(X_dummy_test, y_dummy_test)
                logger.info(f"Evaluator example finished. Test Metrics: {test_metrics}")
                logger.info(f"Check the '{example_eval_config['artifact_base_path']}' directory and MLflow UI for artifacts.")

            except Exception as e:
                logger.error(f"An error occurred during the ModelEvaluator example: {e}", exc_info=True)
                mlflow.set_tag("mlflow.runName", "FAILED_Evaluate_ExampleModel")
    else:
        logger.error("Skipping ModelEvaluator example as the dummy pipeline could not be fitted.")