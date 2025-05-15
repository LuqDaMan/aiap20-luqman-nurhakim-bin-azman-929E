#!/bin/bash
# run_app.sh
# Script to launch all services for the Bank Term Deposit Prediction local deployment.
# Attempts to dynamically load URIs/ports from config/deploy_config.yaml using 'yq'.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_CONFIG_FILE="$PROJECT_ROOT/config/deploy_config.yaml"
echo "Project Root: $PROJECT_ROOT"
echo "Using Deployment Configuration: $DEPLOY_CONFIG_FILE"
cd "$PROJECT_ROOT" # Ensure commands run from project root

VENV_PATH="$PROJECT_ROOT/.venv/bin/activate" # Common .venv path

# Function to read value from YAML using yq
# Usage: get_yaml_value <key_path_for_yq> <default_value>
get_yaml_value() {
    local key_path_input="$1"       # e.g., '.mlflow.tracking_server_uri'
    local default_value="$2"
    local config_file_to_read="${3:-$DEPLOY_CONFIG_FILE}" # Use 3rd arg or fallback to DEPLOY_CONFIG_FILE

    local value=""
    local yq_filter_path="$key_path_input"
    local yq_exit_code=0

    # 1. Check if yq and jq commands are available
    if ! command -v yq &> /dev/null || ! command -v jq &> /dev/null; then
        # Optional: Add a one-time warning if yq or jq is missing, if desired.
        # echo "Warning: yq or jq not found. Using default for $key_path_input." >&2
        echo "$default_value"
        return
    fi

    # 2. Check if the configuration file exists
    if [ ! -f "$config_file_to_read" ]; then
        # Optional: Add a warning if the config file is missing.
        # echo "Warning: Config file $config_file_to_read not found for $key_path_input. Using default." >&2
        echo "$default_value"
        return
    fi

    # 3. Execute yq with raw output option.
    # Suppress stderr for common cases like path not found (yq -r might still exit 0).
    value=$(yq -r "$yq_filter_path" "$config_file_to_read" 2>/dev/null)
    yq_exit_code=$?

    # 4. Determine if the default value should be used
    # Use default if:
    #   a) yq exited with an error.
    #   b) yq exited successfully BUT returned an empty string (and that empty string is not a valid "false" boolean).
    #   c) yq returned the literal string "null".
    if [ $yq_exit_code -ne 0 ] || \
       { [ $yq_exit_code -eq 0 ] && [ -z "$value" ] && [ "$value" != "false" ]; } || \
       [ "$value" = "null" ]; then
        echo "$default_value"
    else
        echo "$value" # Value is successfully retrieved and should be raw
    fi
}

# --- Default/Fallback values ---
# These are used if yq is not available or if keys are missing from deploy_config.yaml
DEFAULT_MLFLOW_TRACKING_URI="http://localhost:5001"

DEFAULT_LR_MODEL_HOST="127.0.0.1"; DEFAULT_LR_MODEL_PORT="5002" # Ensure defaults are strings if yq might return numbers
DEFAULT_LR_MODEL_URI="runs:/d2ed5511d9b1462c9ce34d85a913a9e2/LogisticRegression"; DEFAULT_LR_MODEL_WORKERS="1"

DEFAULT_RF_MODEL_HOST="127.0.0.1"; DEFAULT_RF_MODEL_PORT="5003"
DEFAULT_RF_MODEL_URI="runs:/4178130253334dddbe71b2a3820b5687/RandomForestClassifier"; DEFAULT_RF_MODEL_WORKERS="1"

DEFAULT_GB_MODEL_HOST="127.0.0.1"; DEFAULT_GB_MODEL_PORT="5004"
DEFAULT_GB_MODEL_URI="runs:/8fdb30a541d14ae0800c7adf683a62e7/GradientBoostingClassifier"; DEFAULT_GB_MODEL_WORKERS="1"

DEFAULT_API_HOST="0.0.0.0"; DEFAULT_API_PORT="8000"
DEFAULT_API_WORKERS="4"; DEFAULT_API_RELOAD="true" # 'true' as a string for bash comparison

DEFAULT_STREAMLIT_APP_FILE="src/app/streamlit_main.py"
DEFAULT_STREAMLIT_SERVER_ADDRESS="0.0.0.0"; DEFAULT_STREAMLIT_SERVER_PORT="8501"

# --- Load values dynamically or use defaults ---
echo ""
echo "Loading configurations..."
if ! command -v yq &> /dev/null; then
    echo "WARNING: 'yq' command not found. Using default/fallback configurations for services."
    echo "         Consider installing yq (e.g., 'pip install yq' or from https://github.com/mikefarah/yq)"
    echo "         for dynamic configuration loading from '$DEPLOY_CONFIG_FILE'."
fi

MLFLOW_TRACKING_URI=$(get_yaml_value '.mlflow.tracking_server_uri' "$DEFAULT_MLFLOW_TRACKING_URI")

LR_MODEL_HOST=$(get_yaml_value '.mlflow_model_servers.logistic_regression.host' "$DEFAULT_LR_MODEL_HOST")
LR_MODEL_PORT=$(get_yaml_value '.mlflow_model_servers.logistic_regression.port' "$DEFAULT_LR_MODEL_PORT")
LR_MODEL_URI=$(get_yaml_value '.mlflow_model_servers.logistic_regression.model_uri' "$DEFAULT_LR_MODEL_URI")
LR_MODEL_WORKERS=$(get_yaml_value '.mlflow_model_servers.logistic_regression.workers' "$DEFAULT_LR_MODEL_WORKERS")

RF_MODEL_HOST=$(get_yaml_value '.mlflow_model_servers.random_forest.host' "$DEFAULT_RF_MODEL_HOST")
RF_MODEL_PORT=$(get_yaml_value '.mlflow_model_servers.random_forest.port' "$DEFAULT_RF_MODEL_PORT")
RF_MODEL_URI=$(get_yaml_value '.mlflow_model_servers.random_forest.model_uri' "$DEFAULT_RF_MODEL_URI")
RF_MODEL_WORKERS=$(get_yaml_value '.mlflow_model_servers.random_forest.workers' "$DEFAULT_RF_MODEL_WORKERS")

GB_MODEL_HOST=$(get_yaml_value '.mlflow_model_servers.gradient_boosting.host' "$DEFAULT_GB_MODEL_HOST")
GB_MODEL_PORT=$(get_yaml_value '.mlflow_model_servers.gradient_boosting.port' "$DEFAULT_GB_MODEL_PORT")
GB_MODEL_URI=$(get_yaml_value '.mlflow_model_servers.gradient_boosting.model_uri' "$DEFAULT_GB_MODEL_URI")
GB_MODEL_WORKERS=$(get_yaml_value '.mlflow_model_servers.gradient_boosting.workers' "$DEFAULT_GB_MODEL_WORKERS")

API_HOST=$(get_yaml_value '.api.host' "$DEFAULT_API_HOST")
API_PORT=$(get_yaml_value '.api.port' "$DEFAULT_API_PORT")
API_WORKERS=$(get_yaml_value '.api.workers' "$DEFAULT_API_WORKERS")
API_RELOAD_STR=$(get_yaml_value '.api.reload' "$DEFAULT_API_RELOAD") # yq returns boolean as 'true'/'false' strings

# Streamlit config: Add these to your deploy_config.yaml under 'streamlit_app' if you want to control them from there
# For example: streamlit_app: { title: "...", api_base_url: "...", server_address: "0.0.0.0", server_port: 8501 }
STREAMLIT_SERVER_ADDRESS=$(get_yaml_value '.streamlit_app.server_address' "$DEFAULT_STREAMLIT_SERVER_ADDRESS")
STREAMLIT_SERVER_PORT=$(get_yaml_value '.streamlit_app.server_port' "$DEFAULT_STREAMLIT_SERVER_PORT")
STREAMLIT_APP_FILE="$PROJECT_ROOT/src/app/streamlit_main.py" # Path is usually fixed


# --- Log Effective Configurations ---
echo ""
echo "--- Effective Configurations Being Used ---"
echo "MLflow Tracking URI: $MLFLOW_TRACKING_URI"
echo "Logistic Regression Server: URI='$LR_MODEL_URI', Host='$LR_MODEL_HOST', Port='$LR_MODEL_PORT', Workers='$LR_MODEL_WORKERS'"
echo "Random Forest Server: URI='$RF_MODEL_URI', Host='$RF_MODEL_HOST', Port='$RF_MODEL_PORT', Workers='$RF_MODEL_WORKERS'"
echo "Gradient Boosting Server: URI='$GB_MODEL_URI', Host='$GB_MODEL_HOST', Port='$GB_MODEL_PORT', Workers='$GB_MODEL_WORKERS'"
echo "FastAPI Server: Host='$API_HOST', Port='$API_PORT', Workers='$API_WORKERS', Reload='$API_RELOAD_STR'"
echo "Streamlit App: File='$STREAMLIT_APP_FILE', Port='$STREAMLIT_SERVER_PORT', Address='$STREAMLIT_SERVER_ADDRESS'"
echo "-----------------------------------------"
echo ""


# Log Directories
LOG_DIR_MLFLOW_SERVERS="$PROJECT_ROOT/logs/mlflow_model_servers"
LOG_DIR_API="$PROJECT_ROOT/logs/api" # Used by FastAPI app's logger
LOG_DIR_STREAMLIT="$PROJECT_ROOT/logs/streamlit" # Used by Streamlit app's logger

# PIDs of background processes for cleanup
PID_MLFLOW_LR=""
PID_MLFLOW_RF=""
PID_MLFLOW_GB=""
PID_FASTAPI=""

cleanup() {
    echo ""
    echo "Initiating shutdown of services..."
    # Kill processes by PID. Added check if PID is not empty and some error suppression for kill.
    if [ -n "$PID_FASTAPI" ]; then kill "$PID_FASTAPI" &>/dev/null && echo "FastAPI server (PID $PID_FASTAPI) terminated." || echo "Attempted to stop FastAPI server (PID $PID_FASTAPI)."; fi
    if [ -n "$PID_MLFLOW_GB" ]; then kill "$PID_MLFLOW_GB" &>/dev/null && echo "MLflow GB server (PID $PID_MLFLOW_GB) terminated." || echo "Attempted to stop MLflow GB server (PID $PID_MLFLOW_GB)."; fi
    if [ -n "$PID_MLFLOW_RF" ]; then kill "$PID_MLFLOW_RF" &>/dev/null && echo "MLflow RF server (PID $PID_MLFLOW_RF) terminated." || echo "Attempted to stop MLflow RF server (PID $PID_MLFLOW_RF)."; fi
    if [ -n "$PID_MLFLOW_LR" ]; then kill "$PID_MLFLOW_LR" &>/dev/null && echo "MLflow LR server (PID $PID_MLFLOW_LR) terminated." || echo "Attempted to stop MLflow LR server (PID $PID_MLFLOW_LR)."; fi
    echo "All background services signaled to stop."
    exit 0 # Exit script after cleanup
}

trap cleanup SIGINT SIGTERM

# --- Preparations ---
echo "Preparing environment and directories..."
if [ -f "$VENV_PATH" ]; then
    echo "Activating Python virtual environment from $VENV_PATH..."
    # shellcheck disable=SC1090
    source "$VENV_PATH"
else
    echo "Warning: Python virtual environment not found at $VENV_PATH."
    echo "Please ensure all dependencies are installed in your current Python environment."
fi

mkdir -p "$LOG_DIR_MLFLOW_SERVERS"
mkdir -p "$LOG_DIR_API"
mkdir -p "$LOG_DIR_STREAMLIT"

# --- Start Services ---
echo ""
echo "Starting local deployment services..."
echo "Setting MLflow tracking URI environment variable..."
export MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI"
echo "NOTE: Ensure the MLflow Tracking Server is running at $MLFLOW_TRACKING_URI for 'runs:/' URIs."
echo ""

# 1. Start MLflow Model Servers (in background)
echo "Starting MLflow Model Server for Logistic Regression (Port: $LR_MODEL_PORT)..."
mlflow models serve -m "$LR_MODEL_URI" -h "$LR_MODEL_HOST" -p "$LR_MODEL_PORT" -w "$LR_MODEL_WORKERS" --no-conda > "$LOG_DIR_MLFLOW_SERVERS/lr_model_server.log" 2>&1 &
PID_MLFLOW_LR=$!

echo "Starting MLflow Model Server for Random Forest (Port: $RF_MODEL_PORT)..."
mlflow models serve -m "$RF_MODEL_URI" -h "$RF_MODEL_HOST" -p "$RF_MODEL_PORT" -w "$RF_MODEL_WORKERS" --no-conda > "$LOG_DIR_MLFLOW_SERVERS/rf_model_server.log" 2>&1 &
PID_MLFLOW_RF=$!

echo "Starting MLflow Model Server for Gradient Boosting (Port: $GB_MODEL_PORT)..."
mlflow models serve -m "$GB_MODEL_URI" -h "$GB_MODEL_HOST" -p "$GB_MODEL_PORT" -w "$GB_MODEL_WORKERS" --no-conda > "$LOG_DIR_MLFLOW_SERVERS/gb_model_server.log" 2>&1 &
PID_MLFLOW_GB=$!
echo "MLflow servers launched with PIDs: LR=$PID_MLFLOW_LR, RF=$PID_MLFLOW_RF, GB=$PID_MLFLOW_GB."
echo "Logs: $LOG_DIR_MLFLOW_SERVERS/"
sleep 5

# 2. Start FastAPI Backend Server (in background)
echo "Starting FastAPI backend server (Host: $API_HOST, Port: $API_PORT)..."
UVICORN_CMD_ARRAY=(uvicorn src.api.main:app --host "$API_HOST" --port "$API_PORT" --workers "$API_WORKERS")
# yq returns boolean 'true'/'false' as strings
if [[ "$API_RELOAD_STR" == "true" || "$API_RELOAD_STR" == "True" ]]; then
    UVICORN_CMD_ARRAY+=(--reload)
fi
"${UVICORN_CMD_ARRAY[@]}" > "$LOG_DIR_API/uvicorn_server.log" 2>&1 &
PID_FASTAPI=$!
echo "FastAPI server launched with PID $PID_FASTAPI. Uvicorn logs: $LOG_DIR_API/uvicorn_server.log. App logs in $LOG_DIR_API."
sleep 3

# 3. Start Streamlit Frontend Application (in foreground)
echo "Starting Streamlit frontend application (Port: $STREAMLIT_SERVER_PORT)..."
echo "Access Streamlit UI at: http://$STREAMLIT_SERVER_ADDRESS:$STREAMLIT_SERVER_PORT (or http://localhost:$STREAMLIT_SERVER_PORT if address is 0.0.0.0)"
streamlit run "$STREAMLIT_APP_FILE" \
    --server.port "$STREAMLIT_SERVER_PORT" \
    --server.address "$STREAMLIT_SERVER_ADDRESS" \
    --browser.serverAddress "$STREAMLIT_SERVER_ADDRESS" # Helps Streamlit open the correct browser URL

echo "Streamlit application has been shut down."
cleanup # Call cleanup explicitly when Streamlit (the foreground process) exits