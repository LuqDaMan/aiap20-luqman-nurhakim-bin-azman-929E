# src/model_evaluator.py

import logging
import pandas as pd
import numpy as np
import os
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

logger = logging.getLogger('pipeline.model_evaluator')

# Attempt to import matplotlib for plotting, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not found. Plotting functionality will be disabled in model_evaluator.")


def evaluate_model_on_test_set(
    model_pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    config: dict
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
    # Assuming 'yes' class (positive class) is encoded as 1, as per PRD and typical setup
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

        # --- Plotting (Optional) ---
        plots_dir = config.get("output_paths", {}).get("plots_dir")
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
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=pos_label)
                roc_auc_val = evaluation_results.get('roc_auc_score', np.nan)
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}')
                plt.legend(loc="lower right")
                plt.grid(True)
                roc_plot_path = os.path.join(plots_dir, f"{model_name}_test_roc_curve.png")
                try:
                    plt.savefig(roc_plot_path)
                    logger.info(f"ROC curve for {model_name} saved to {roc_plot_path}")
                except Exception as e_plot:
                    logger.error(f"Failed to save ROC curve plot for {model_name}: {e_plot}")
                plt.close()

                # Plot Precision-Recall Curve
                auprc_val = evaluation_results.get('auprc_score', np.nan)
                if not np.isnan(auprc_val): # Ensure precision_vals, recall_vals were computed
                    plt.figure(figsize=(8, 6))
                    # We need precision_vals and recall_vals from earlier for AUPRC if we didn't store them
                    # Recompute if necessary for plotting, or ensure they are available
                    temp_precision_vals, temp_recall_vals, _ = precision_recall_curve(y_test, y_pred_proba, pos_label=pos_label)
                    plt.plot(temp_recall_vals, temp_precision_vals, color='blue', lw=2, label=f'PR curve (AUPRC = {auprc_val:.2f})')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.ylim([0.0, 1.05])
                    plt.xlim([0.0, 1.0])
                    plt.title(f'Precision-Recall Curve - {model_name}')
                    plt.legend(loc="lower left")
                    plt.grid(True)
                    pr_plot_path = os.path.join(plots_dir, f"{model_name}_test_pr_curve.png")
                    try:
                        plt.savefig(pr_plot_path)
                        logger.info(f"Precision-Recall curve for {model_name} saved to {pr_plot_path}")
                    except Exception as e_plot:
                        logger.error(f"Failed to save PR curve plot for {model_name}: {e_plot}")
                    plt.close()
        elif plots_dir and not MATPLOTLIB_AVAILABLE:
            logger.info("Plotting directory specified but Matplotlib is not available. Skipping plot generation.")

    except Exception as e:
        logger.error(f"An error occurred during evaluation of {model_name}: {e}", exc_info=True)
        evaluation_results["error"] = str(e)

    # Log summary of metrics
    log_metrics_summary = {k: v for k, v in evaluation_results.items() if isinstance(v, (int, float, str))}
    logger.info(f"Final evaluation metrics for {model_name} on test set: {log_metrics_summary}")
    
    return evaluation_results