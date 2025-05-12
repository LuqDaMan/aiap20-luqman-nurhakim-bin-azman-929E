# src/model_evaluator.py

import logging
import pandas as pd
import numpy as np
import os
import mlflow
from typing import Dict, Any

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    auc,
    roc_curve,
    classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger('pipeline.model_evaluator')

# Attempt to import matplotlib for plotting, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not found. Plotting functionality will be disabled in model_evaluator.")

# Attempt to import SHAP, but make it optional
try:
    import shap
    SHAP_AVAILABLE = True
    shap.initjs()
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP library not found. SHAP plotting will be disabled.")


def evaluate_model_on_test_set(
    model_pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    config: dict,
    run_id: str
) -> Dict[str, Any]: # Changed return value to Dict[str, Any] to include model_name
    """
    Evaluates the given model pipeline on the test set and logs/returns metrics.

    Args:
        model_pipeline (Any): The trained model pipeline (e.g., from scikit-learn or imblearn).
        X_test (pd.DataFrame): The processed test features.
        y_test (pd.Series): The test target variable.
        model_name (str): The name of the model being evaluated.
        config (dict): The pipeline configuration dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing the evaluation metrics.
    """
    if model_pipeline is None:
        logger.error(f"Model pipeline for '{model_name}' is None. Cannot evaluate.")
        return {"model_name": model_name, "error": "Model pipeline is None."}

    logger.info(f"--- Evaluating model: {model_name} on the test set ---")

    evaluation_results: Dict[str, Any] = {"model_name": model_name}
    metrics_to_calculate = config.get('evaluation_metrics', [])
    # Assuming 'yes' class (positive class) is encoded as 1, as per typical setup
    pos_label = 1 

    try:
        # Make predictions
        y_pred = model_pipeline.predict(X_test)

        # Get probabilities for the positive class
        y_pred_proba = None
        if hasattr(model_pipeline, "predict_proba"):
            try:
                y_pred_proba_all_classes = model_pipeline.predict_proba(X_test)
                if y_pred_proba_all_classes.ndim == 2 and y_pred_proba_all_classes.shape[1] >= 2:
                    y_pred_proba = y_pred_proba_all_classes[:, 1] # Probability of positive class
                elif y_pred_proba_all_classes.ndim == 1: # Some classifiers might return 1D if only one class in training y
                     logger.warning(f"predict_proba for {model_name} returned 1D array. This might indicate issues with model training or data. ROC/AUPRC might be affected.")
                     # Try to use it if it seems like positive class probabilities, otherwise set to None
                     # This case needs careful handling based on model behavior. For now, we assume it's prob of pos class if 1D.
                     y_pred_proba = y_pred_proba_all_classes
                else:
                    logger.warning(f"predict_proba for {model_name} returned unexpected shape: {y_pred_proba_all_classes.shape}. ROC-AUC and AUPRC may not be reliable.")
            except Exception as e_proba:
                logger.error(f"Error obtaining prediction probabilities for {model_name}: {e_proba}")
        else:
            logger.warning(f"Model {model_name} does not have predict_proba method. ROC-AUC and AUPRC cannot be calculated.")

        # --- Calculate Metrics ---
        if "f1_score_yes_class" in metrics_to_calculate:
            evaluation_results["f1_score_yes_class"] = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        if "roc_auc_score" in metrics_to_calculate:
            if y_pred_proba is not None:
                evaluation_results["roc_auc_score"] = roc_auc_score(y_test, y_pred_proba)
            else:
                evaluation_results["roc_auc_score"] = np.nan # Or some other indicator of unavailability
        if "recall_yes_class" in metrics_to_calculate:
            evaluation_results["recall_yes_class"] = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        if "precision_yes_class" in metrics_to_calculate:
            evaluation_results["precision_yes_class"] = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
        if "auprc_score" in metrics_to_calculate:
            if y_pred_proba is not None:
                precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba, pos_label=pos_label)
                evaluation_results["auprc_score"] = auc(recall_vals, precision_vals)
            else:
                evaluation_results["auprc_score"] = np.nan

        # Log detailed classification report and confusion matrix
        report_str = classification_report(y_test, y_pred, target_names=['no (0)', 'yes (1)'], zero_division=0)
        logger.info(f"\nClassification Report for {model_name} on Test Set:\n{report_str}")
        evaluation_results["classification_report"] = report_str # Store for potential later use

        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix for {model_name} on Test Set:\n{cm}")
        evaluation_results["confusion_matrix"] = cm.tolist() # Store as list of lists

        # --- Plotting ---
        plots_dir_config = config.get("output_paths", {}).get("plots_dir", "plots/")
        # MLflow will create artifact paths, local saving is optional if primarily relying on MLflow UI
        # Create a run-specific subdirectory for plots if saving locally and not just to MLflow
        current_run_id = mlflow.active_run().info.run_id # Get current run ID
        plots_dir = os.path.join(plots_dir_config, current_run_id, model_name) # Example: plots/RUN_ID/MODEL_NAME/

        if MATPLOTLIB_AVAILABLE and plots_dir:
            if not os.path.exists(plots_dir):
                try:
                    os.makedirs(plots_dir)
                    logger.info(f"Created plots directory: {plots_dir}")
                except OSError as e:
                    logger.error(f"Could not create plots directory {plots_dir}: {e}")
                    plots_dir = None 

            if plots_dir and y_pred_proba is not None:
                # Plot ROC Curve
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                roc_auc_val = evaluation_results.get('roc_auc_score', np.nan)
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title(f'ROC - {model_name} (Test Set)') # Update title for clarity
                ax_roc.legend(loc="lower right")
                ax_roc.grid(True)
                local_roc_plot_path = os.path.join(plots_dir, f"{model_name}_test_roc_curve.png")
                try:
                    fig_roc.savefig(local_roc_plot_path)
                    logger.info(f"ROC curve for {model_name} saved to {local_roc_plot_path}")
                    mlflow.log_figure(fig_roc, artifact_file=f"plots/{model_name}_test_roc_curve.png")
                except Exception as e_plot:
                    logger.error(f"Failed to save/log ROC curve plot for {model_name}: {e_plot}")
                plt.close(fig_roc)

                # Plot Precision-Recall Curve
                auprc_val = evaluation_results.get('auprc_score', np.nan)
                if not np.isnan(auprc_val): # Ensure precision_vals, recall_vals were computed
                    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
                    temp_precision_vals, temp_recall_vals, _ = precision_recall_curve(y_test, y_pred_proba, pos_label=pos_label)
                    ax_pr.plot(temp_recall_vals, temp_precision_vals, color='blue', lw=2, label=f'PR curve (AUPRC = {auprc_val:.2f})')
                    ax_pr.set_xlabel('Recall')
                    ax_pr.set_ylabel('Precision')
                    ax_pr.set_ylim([0.0, 1.05])
                    ax_pr.set_xlim([0.0, 1.0])
                    ax_pr.set_title(f'Precision-Recall - {model_name} (Test Set)')
                    ax_pr.legend(loc="lower left")
                    ax_pr.grid(True)
                    local_pr_plot_path = os.path.join(plots_dir, f"{model_name}_test_pr_curve.png")
                    try:
                        fig_pr.savefig(local_pr_plot_path)
                        logger.info(f"PR curve for {model_name} saved to {local_pr_plot_path}")
                        mlflow.log_figure(fig_pr, artifact_file=f"plots/{model_name}_test_pr_curve.png")
                        # mlflow.log_artifact(local_pr_plot_path, artifact_path=f"plots/{model_name}") # Alternative
                    except Exception as e_plot:
                        logger.error(f"Failed to save/log PR curve plot for {model_name}: {e_plot}")
                    plt.close(fig_pr)

        # --- SHAP Feature Importance ---
        if SHAP_AVAILABLE and MATPLOTLIB_AVAILABLE and plots_dir:
            logger.info(f"Calculating and plotting SHAP values for {model_name}...")
            try:
                if 'classifier' in model_pipeline.named_steps:
                    model_to_explain = model_pipeline.named_steps['classifier']
                elif 'model' in model_pipeline.named_steps: # Common alternative
                    model_to_explain = model_pipeline.named_steps['model']
                else: # Fallback or error if model step not found by common names
                    logger.warning(f"Could not find classifier/model step in pipeline for SHAP. Attempting to use final step.")
                    model_to_explain = model_pipeline.steps[-1][1]


                explainer = None 
                # --- Subsample X_test for SHAP calculation for performance ---
                # Decide on a sample size. For visualization, 500-2000 instances are often enough.
                shap_sample_size = min(500, X_test.shape[0]) # Adjust sample size as needed
                if X_test.shape[0] > shap_sample_size:
                    logger.info(f"Using a subsample of {shap_sample_size} instances from X_test for SHAP calculation.")
                    X_test_shap_sample = X_test.sample(n=shap_sample_size, random_state=config.get("random_seed", 42))
                else:
                    X_test_shap_sample = X_test.copy()
                # --- End Subsample ---
                shap_values = None

                if isinstance(model_to_explain, (RandomForestClassifier, GradientBoostingClassifier)):
                    explainer = shap.TreeExplainer(model_to_explain, data=X_test.copy(), model_output="probability" if hasattr(model_to_explain, 'predict_proba') else "raw", feature_perturbation="interventional")
                    if explainer:
                        shap_values = explainer.shap_values(X_test_shap_sample, check_additivity=False) # check_additivity is valid here
                elif isinstance(model_to_explain, LogisticRegression):
                    explainer = shap.LinearExplainer(model_to_explain, masker=X_test.copy())
                    if explainer:
                        shap_values = explainer.shap_values(X_test_shap_sample) # No check_additivity here
                else:
                    logger.info(f"Using shap.KernelExplainer for {model_name}. This might be slow.")
                    predict_fn_shap = lambda x: model_pipeline.predict_proba(pd.DataFrame(x, columns=X_test_shap_sample.columns))[:,1] if hasattr(model_pipeline, 'predict_proba') else model_pipeline.predict(pd.DataFrame(x, columns=X_test_shap_sample.columns))
                    background_kernel_size = min(100, X_test.shape[0]) # Use a sample from the original X_test or X_train
                    background_data_kernel = shap.sample(X_test, background_kernel_size, random_state=config.get("random_seed", 42))
                    explainer = shap.KernelExplainer(predict_fn_shap, background_data_kernel)
                    if explainer:
                        logger.info(f"Calculating SHAP values for {X_test_shap_sample.shape[0]} sample instances (KernelExplainer)...")
                        shap_values = explainer.shap_values(X_test_shap_sample)

                # --- Logging after SHAP value computation attempt ---
                logger.info(f"SHAP State Post-Calculation for {model_name}:")
                logger.info(f"  explainer is None: {explainer is None}")
                if explainer is not None:
                    logger.info(f"  explainer type: {type(explainer)}")
                logger.info(f"  shap_values is None: {shap_values is None}")
                if shap_values is not None:
                    logger.info(f"  shap_values type: {type(shap_values)}")
                    if isinstance(shap_values, np.ndarray):
                        logger.info(f"  shap_values shape: {shap_values.shape}")
                    elif isinstance(shap_values, list):
                        logger.info(f"  shap_values list length: {len(shap_values)}")
                        for i_sv_post, item_sv_post in enumerate(shap_values):
                            logger.info(f"    shap_values[{i_sv_post}] type: {type(item_sv_post)}")
                            if isinstance(item_sv_post, np.ndarray):
                                logger.info(f"    shap_values[{i_sv_post}] shape: {item_sv_post.shape}")

                # This is the section to be modified, after shap_values = explainer.shap_values(...)
                    if explainer and shap_values is not None:
                        shap_values_for_plot = None
                        expected_value_for_plot = None

                        # --- MODIFIED logic to determine shap_values_for_plot and expected_value_for_plot ---
                        if isinstance(shap_values, list) and len(shap_values) == 2:
                            # Standard case: list of [shap_class0, shap_class1], each should be (N, M)
                            _raw_class1_shaps = shap_values[1] # Candidate for positive class

                            if isinstance(_raw_class1_shaps, np.ndarray):
                                if _raw_class1_shaps.ndim == 2:
                                    shap_values_for_plot = _raw_class1_shaps
                                elif _raw_class1_shaps.ndim == 3 and _raw_class1_shaps.shape[-1] == 1: # Shape (N, M, 1)
                                    shap_values_for_plot = np.squeeze(_raw_class1_shaps, axis=-1)
                                    logger.info(f"SHAP values from list (shap_values[1]) had shape {_raw_class1_shaps.shape}, squeezed to {shap_values_for_plot.shape} for {model_name}.")
                                elif _raw_class1_shaps.ndim == 3 and _raw_class1_shaps.shape[-1] == 2 and \
                                     list(_raw_class1_shaps.shape[:2]) == [X_test_shap_sample.shape[0], X_test_shap_sample.shape[1]]:
                                    # This handles if shap_values[1] itself is (N, M, 2)
                                    logger.warning(f"shap_values[1] for {model_name} has shape {_raw_class1_shaps.shape}. "
                                                 f"This implies the SHAP values for the positive class are multi-component. "
                                                 f"Attempting to use slice [:, :, 1]. Review if this is the correct component for your model's positive class explanation.")
                                    shap_values_for_plot = _raw_class1_shaps[:, :, 1]
                                else:
                                    # Not 2D, or 3D but not (N,M,1) or expected (N,M,2). Let validation catch.
                                    shap_values_for_plot = _raw_class1_shaps
                                    logger.warning(f"shap_values[1] for {model_name} has an unhandled shape: {_raw_class1_shaps.shape if isinstance(_raw_class1_shaps, np.ndarray) else type(_raw_class1_shaps)}. Will proceed to validation.")
                            else: # _raw_class1_shaps is not a numpy array
                                shap_values_for_plot = _raw_class1_shaps # Let validation catch this type issue

                            # Corresponding expected value for the positive class
                            if hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) == 2:
                                expected_value_for_plot = explainer.expected_value[1]
                            elif hasattr(explainer, 'expected_value'):
                                expected_value_for_plot = explainer.expected_value

                        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                            # Case: explainer.shap_values() is a single 3D array (n_samples, n_features, n_classes_or_outputs)
                            # This matches the user's logged shape (500, 33, 2) if `shap_values` was this 3D array.
                            n_samples_s, n_features_s, n_outputs_s = shap_values.shape
                            
                            if n_samples_s != X_test_shap_sample.shape[0] or n_features_s != X_test_shap_sample.shape[1]:
                                logger.error(f"SHAP values 3D array dimensions ({n_samples_s},{n_features_s}) for {model_name} mismatch "
                                             f"X_test_shap_sample ({X_test_shap_sample.shape[0]},{X_test_shap_sample.shape[1]}). Skipping SHAP plots.")
                                shap_values_for_plot = shap_values # Mark to fail validation
                            elif n_outputs_s >= 2: # Assuming the last dimension is for classes; take index 1 for positive class
                                logger.info(f"Raw SHAP values for {model_name} are a 3D numpy array with shape {shap_values.shape}. "
                                            f"Using slice [:, :, 1] for the positive class.")
                                shap_values_for_plot = shap_values[:, :, 1] 
                                if hasattr(explainer, 'expected_value'):
                                    if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) == n_outputs_s:
                                        expected_value_for_plot = explainer.expected_value[1]
                                    else: 
                                        expected_value_for_plot = explainer.expected_value 
                                        logger.warning(f"Using provided expected_value (type: {type(explainer.expected_value)}) "
                                                     f"for {model_name} with 3D SHAP array slice. Ensure it corresponds to class 1.")
                            elif n_outputs_s == 1: # Shape (N, M, 1)
                                shap_values_for_plot = np.squeeze(shap_values, axis=-1)
                                expected_value_for_plot = explainer.expected_value
                                logger.info(f"Raw SHAP values for {model_name} were 3D array {shap_values.shape}, squeezed to 2D.")
                            else: # n_outputs_s == 0
                                logger.warning(f"Raw SHAP values for {model_name} are 3D array {shap_values.shape} with 0 outputs in last dim. Cannot plot.")
                                shap_values_for_plot = shap_values # Mark to fail validation
                        
                        else: # Fallback: shap_values is already 2D (e.g., regression) or an unhandled structure.
                            shap_values_for_plot = shap_values
                            if hasattr(explainer, 'expected_value'):
                                expected_value_for_plot = explainer.expected_value
                        # --- End MODIFIED logic ---

                        # Determine feature_names from X_test_shap_sample
                        if hasattr(X_test_shap_sample, 'columns'):
                            feature_names = X_test_shap_sample.columns.tolist()
                        elif isinstance(X_test_shap_sample, np.ndarray) and X_test_shap_sample.ndim == 2 :
                            feature_names = [f"feature_{i}" for i in range(X_test_shap_sample.shape[1])]
                        else:
                            logger.error(f"X_test_shap_sample for {model_name} is not a DataFrame or 2D NumPy array. Cannot determine feature names. Skipping SHAP plots.")
                            feature_names = [] 

                        # --- VALIDATION STEP (this was the previously added validation block) ---
                        can_create_explanation = True
                        if not isinstance(shap_values_for_plot, np.ndarray) or shap_values_for_plot.ndim != 2:
                            logger.warning(f"After processing, SHAP values for {model_name} are still not a 2D numpy array (actual shape: {getattr(shap_values_for_plot, 'shape', 'N/A')}, type: {type(shap_values_for_plot)}). Skipping SHAP plots.")
                            can_create_explanation = False
                        elif not feature_names: 
                            logger.warning(f"Could not determine feature names for SHAP ({model_name}), or no features in input. Skipping SHAP plots.")
                            can_create_explanation = False
                        # The following two checks are now more specific after attempting to make shap_values_for_plot 2D
                        elif shap_values_for_plot.shape[0] != X_test_shap_sample.shape[0]: # Check number of samples
                             logger.warning(f"SHAP values for {model_name} have incorrect number of samples "
                                          f"(expected {X_test_shap_sample.shape[0]}, got {shap_values_for_plot.shape[0]}). Skipping SHAP plots.")
                             can_create_explanation = False
                        elif shap_values_for_plot.shape[1] == 0 and len(feature_names) > 0 : # SHAP values have 0 features, but input data has features
                            logger.warning(f"SHAP values calculated for {model_name} resulted in 0 features, though input data has {len(feature_names)} features. Skipping SHAP plots.")
                            can_create_explanation = False
                        elif shap_values_for_plot.shape[1] != len(feature_names): # Mismatch in feature count
                            logger.error(f"Mismatch for {model_name}: SHAP values feature count ({shap_values_for_plot.shape[1]}) "
                                         f"differs from data's feature name count ({len(feature_names)}). Skipping SHAP plots.")
                            can_create_explanation = False
                         # --- End VALIDATION STEP ---

                        if can_create_explanation and hasattr(shap, 'Explanation'):
                            data_for_explanation = X_test_shap_sample.to_numpy() if isinstance(X_test_shap_sample, pd.DataFrame) else X_test_shap_sample
                            
                            # Ensure expected_value_for_plot is a scalar or 1D array of base values for the explanation
                            if isinstance(expected_value_for_plot, (list, np.ndarray)) and np.asarray(expected_value_for_plot).ndim > 0 and len(np.asarray(expected_value_for_plot)) != 1 and len(np.asarray(expected_value_for_plot)) != shap_values_for_plot.shape[0]:
                                logger.warning(f"expected_value_for_plot for {model_name} has shape {np.asarray(expected_value_for_plot).shape}, "
                                             f"which is not scalar or matching number of samples. Using its mean or first element if possible.")
                                if np.asarray(expected_value_for_plot).size > 0:
                                     base_val_final = np.mean(expected_value_for_plot) # Or np.asarray(expected_value_for_plot)[0] - depends on expected structure
                                else:
                                     base_val_final = 0 # Default base value if problematic
                                logger.warning(f"Adjusted base_value for SHAP Explanation to: {base_val_final}")
                            else:
                                base_val_final = expected_value_for_plot

                        # --- Create SHAP Explanation Object (Modern API) ---
                                shap_explanation = shap.Explanation(
                                    values=shap_values_for_plot,
                                    base_values=base_val_final,
                                    data=data_for_explanation,
                                    feature_names=feature_names
                                )

                                # 1. SHAP Bar Plot
                                num_features_to_display = min(20, len(feature_names))
                                plt.figure(figsize=(10, max(6, num_features_to_display * 0.4))) # Adjusted height logic
                                shap.plots.bar(shap_explanation, max_display=num_features_to_display, show=False)
                                plt.title(f"SHAP Feature Importance (Bar) - {model_name}")
                                plt.tight_layout()
                                shap_bar_plot_path = os.path.join(plots_dir, f"{model_name}_shap_summary_bar.png")
                                plt.savefig(shap_bar_plot_path)
                                logger.info(f"SHAP bar plot for {model_name} saved to {shap_bar_plot_path}")
                                if mlflow.active_run():
                                    mlflow.log_artifact(shap_bar_plot_path, artifact_path=f"plots/shap/{model_name}")
                                plt.close()

                                # 2. SHAP Beeswarm Plot
                                plt.figure(figsize=(10, max(6, num_features_to_display * 0.4))) # Adjusted height logic
                                shap.plots.beeswarm(shap_explanation, max_display=num_features_to_display, show=False)
                                plt.title(f"SHAP Feature Importance (Beeswarm) - {model_name}")
                                plt.tight_layout()
                                shap_summary_plot_path = os.path.join(plots_dir, f"{model_name}_shap_summary_beeswarm.png")
                                plt.savefig(shap_summary_plot_path)
                                logger.info(f"SHAP beeswarm plot for {model_name} saved to {shap_summary_plot_path}")
                                if mlflow.active_run():
                                    mlflow.log_artifact(shap_summary_plot_path, artifact_path=f"plots/shap/{model_name}")
                                plt.close()

                        elif explainer and shap_values is None:
                            logger.warning(f"SHAP values could not be computed for {model_name}, even though explainer was initialized. Skipping SHAP plots.")
                        elif not explainer and X_test.shape[0] > 0: # only warn if we intended to create an explainer
                            logger.warning(f"SHAP explainer could not be initialized for {model_name}. Skipping SHAP plots.")

            except Exception as e_shap:
                logger.error(f"Error during SHAP value calculation or plotting for {model_name}: {e_shap}", exc_info=True)
        elif plots_dir and (not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE):
            logger.info("SHAP or Matplotlib not available. Skipping SHAP plot generation.")

        elif plots_dir and not MATPLOTLIB_AVAILABLE:
            logger.info("Plotting directory specified but Matplotlib is not available. Skipping plot generation.")

    except Exception as e:
        logger.error(f"An error occurred during evaluation of {model_name}: {e}", exc_info=True)
        evaluation_results["error"] = str(e)

    # Log summary of metrics
    log_metrics_summary = {k: v for k, v in evaluation_results.items() if isinstance(v, (int, float, str))}
    logger.info(f"Final evaluation metrics for {model_name} on test set: {log_metrics_summary}")
    
    return evaluation_results