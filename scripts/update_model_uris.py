# scripts/update_model_uris.py
import mlflow
import yaml
import os
import sys
from typing import Dict, Optional, List

# Ensure src is in PYTHONPATH to import utils if needed for config loading,
# though this script primarily loads deploy_config itself.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.pipe_utils import load_config # For loading deploy_config path

# --- Configuration ---
DEPLOY_CONFIG_PATH = "config/deploy_config.yaml"

# Mapping of model server keys in deploy_config.yaml to the expected artifact paths
# logged by the training pipeline.
# These artifact paths are relative to the run's artifact root.
# Example: 'logistic_regression' key in deploy_config uses the artifact 'LogisticRegression'
MODEL_KEY_TO_ARTIFACT_PATH_MAP: Dict[str, str] = {
    "logistic_regression": "LogisticRegression",
    "random_forest": "RandomForestClassifier",
    "gradient_boosting": "GradientBoostingClassifier"
}

def get_latest_run_id_for_artifact(
    mlflow_client: mlflow.tracking.MlflowClient,
    experiment_id: str,
    expected_model_artifact_folder: str
) -> Optional[str]:
    """
    Finds the latest run ID in a given experiment that contains an artifact
    with the specified path segment.

    Args:
        mlflow_client: Initialized MLflow Tracking Client.
        experiment_id: ID of the experiment to search within.
        artifact_path_segment: The unique path segment of the artifact to look for
                               (e.g., "LogisticRegression", "RandomForestClassifier").

    Returns:
        The run ID (str) if found, else None.
    """
    print(f"Searching for latest run with artifact containing '{expected_model_artifact_folder}' in experiment '{experiment_id}'...")
    
    # Search for runs ordered by start time descending (latest first)
    # We look for runs that have successfully completed.
    runs: List[mlflow.entities.Run] = mlflow_client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status = 'FINISHED'", # Consider only finished runs
        order_by=["start_time DESC"]
    )

    expected_mlmodel_full_path = f"{expected_model_artifact_folder}/MLmodel"
    print(f"The script will look for this exact artifact file path: '{expected_mlmodel_full_path}'")

    for run in runs:
        run_id = run.info.run_id
        print(f"\nChecking run_id: {run_id} (started at {run.info.start_time}) for model file '{expected_mlmodel_full_path}'")
        try:
            # List artifacts. We can list from the expected_model_artifact_folder.
            # The paths returned will be relative to the run's artifact root.
            artifacts_in_specific_folder = mlflow_client.list_artifacts(
                run_id=run_id,
                path=expected_model_artifact_folder # e.g., "LogisticRegression"
            )
            
            # --- Debugging output for artifacts ---
            print(f"  DEBUG: Items listed by list_artifacts(path='{expected_model_artifact_folder}') for run {run_id}: {len(artifacts_in_specific_folder)}")
            found_expected_path_in_debug = False
            for i, artifact_info in enumerate(artifacts_in_specific_folder):
                # artifact_info.path is relative to the run's artifact root.
                # e.g., "LogisticRegression/MLmodel", "LogisticRegression/conda.yaml"
                if i < 10 or expected_model_artifact_folder in artifact_info.path : # Print some items
                     print(f"    Item[{i}]: Path='{artifact_info.path}', IsDir={artifact_info.is_dir}")
                if artifact_info.path == expected_mlmodel_full_path and not artifact_info.is_dir:
                    found_expected_path_in_debug = True # Note if we see it during debug print
            
            if found_expected_path_in_debug:
                print(f"  DEBUG: The exact path '{expected_mlmodel_full_path}' WAS SEEN in the listing above for run {run_id}.")
            else:
                print(f"  DEBUG: The exact path '{expected_mlmodel_full_path}' WAS NOT SEEN in the listing above for run {run_id}.")
            # --- End Debugging ---

            found_mlmodel_file = False
            for item in artifacts_in_specific_folder:
                # Check if item.path (which is absolute from artifact root)
                # matches the expected full path of the MLmodel file.
                if item.path == expected_mlmodel_full_path and not item.is_dir:
                    found_mlmodel_file = True
                    break
            
            if found_mlmodel_file:
                print(f"  SUCCESS: Found '{expected_mlmodel_full_path}' for run_id: {run_id}.")
                return run_id
            else:
                print(f"  INFO: Run {run_id}: Did not find '{expected_mlmodel_full_path}' when listing contents of '{expected_model_artifact_folder}'.")

        except mlflow.exceptions.MlflowException as e:
            if " Aucun artefact trouvé à l'emplacement " in str(e) or "No artifacts found at path" in str(e) or "does not exist" in str(e).lower():
                print(f"  INFO: Model artifact folder '{expected_model_artifact_folder}' does not exist or is empty for run {run_id}.")
            else:
                print(f"  WARNING: MLflowException while checking artifacts for run {run_id}, target path '{expected_model_artifact_folder}'. Error: {e}")
            continue
        except Exception as e:
            print(f"  WARNING: An unexpected error occurred while checking artifacts for run {run_id}, target path '{expected_model_artifact_folder}'. Error: {e}")
            continue
            
    print(f"\nNo suitable run found containing the valid model file '{expected_mlmodel_full_path}' across all checked runs.")
    return None

def update_deploy_config_model_uris(
    deploy_config_file: str,
    model_mapping: Dict[str, str],
    mlflow_tracking_uri: str,
    experiment_name: str
):
    """
    Updates the model_uri fields in the deployment configuration file
    with the latest run IDs from MLflow.
    """
    print(f"Starting update of model URIs in: {deploy_config_file}")
    print(f"Using MLflow Tracking URI: {mlflow_tracking_uri}")
    print(f"Target Experiment Name: {experiment_name}")

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()
    except Exception as e:
        print(f"Error: Could not connect to MLflow tracking server at '{mlflow_tracking_uri}'. Exception: {e}")
        sys.exit(1)

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Error: MLflow Experiment '{experiment_name}' not found.")
            sys.exit(1)
        experiment_id = experiment.experiment_id
        print(f"Found Experiment ID: {experiment_id} for name '{experiment_name}'.")
    except Exception as e:
        print(f"Error accessing MLflow experiment '{experiment_name}'. Exception: {e}")
        sys.exit(1)

    try:
        with open(deploy_config_file, 'r') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Deployment config file '{deploy_config_file}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Could not parse YAML from '{deploy_config_file}'. Exception: {e}")
        sys.exit(1)

    if 'mlflow_model_servers' not in config_data:
        print(f"Error: 'mlflow_model_servers' section not found in '{deploy_config_file}'.")
        sys.exit(1)

    updated_count = 0
    for model_key, artifact_path in model_mapping.items():
        print(f"\nProcessing model key: '{model_key}' (expected artifact: '{artifact_path}')")
        latest_run_id = get_latest_run_id_for_artifact(client, experiment_id, artifact_path)

        if latest_run_id:
            new_model_uri = f"runs:/{latest_run_id}/{artifact_path}"
            if model_key in config_data['mlflow_model_servers']:
                if config_data['mlflow_model_servers'][model_key].get('model_uri') != new_model_uri:
                    config_data['mlflow_model_servers'][model_key]['model_uri'] = new_model_uri
                    print(f"Updating '{model_key}' URI to: {new_model_uri}")
                    updated_count += 1
                else:
                    print(f"URI for '{model_key}' is already up-to-date: {new_model_uri}")
            else:
                print(f"Warning: Model key '{model_key}' not found under 'mlflow_model_servers' in config. Skipping update.")
        else:
            print(f"Warning: Could not find a latest run for model_key '{model_key}' with artifact '{artifact_path}'. URI not updated.")

    if updated_count > 0:
        try:
            with open(deploy_config_file, 'w') as f:
                yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)
            print(f"\nSuccessfully updated {updated_count} model URIs in '{deploy_config_file}'.")
        except Exception as e:
            print(f"Error: Could not write updated configuration to '{deploy_config_file}'. Exception: {e}")
            sys.exit(1)
    else:
        print("\nNo model URIs needed an update or no new models found.")

if __name__ == "__main__":
    print("--- Script to Update Model URIs in Deployment Configuration ---")
    
    # Load deploy_config to get MLflow tracking URI and experiment name
    # This creates a small dependency but avoids hardcoding them here.
    try:
        cfg = load_config(DEPLOY_CONFIG_PATH) # Uses src.utils.load_config
        mlflow_config = cfg.get("mlflow", {})
        tracking_uri = mlflow_config.get("tracking_server_uri")
        exp_name = mlflow_config.get("experiment_name")

        if not tracking_uri or not exp_name:
            print("Error: 'mlflow.tracking_server_uri' or 'mlflow.experiment_name' not found in "
                  f"'{DEPLOY_CONFIG_PATH}'. Please configure them.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error loading initial configuration from {DEPLOY_CONFIG_PATH} to get MLflow settings: {e}")
        sys.exit(1)

    update_deploy_config_model_uris(
        deploy_config_file=DEPLOY_CONFIG_PATH,
        model_mapping=MODEL_KEY_TO_ARTIFACT_PATH_MAP,
        mlflow_tracking_uri=tracking_uri,
        experiment_name=exp_name
    )
    print("--- Update script finished ---")