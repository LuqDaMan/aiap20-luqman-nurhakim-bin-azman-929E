#!/bin/bash

# Script to run the Bank Term Deposit Subscription Prediction Pipeline

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
DEFAULT_CONFIG_PATH="config/pipeline_config.yaml"
CONFIG_PATH="${1:-$DEFAULT_CONFIG_PATH}"
PYTHON_EXECUTABLE="python3"

get_yaml_value() {
    local key_path_input="$1"       # e.g., '.mlflow_server.host'
    local default_value="$2"
    local config_file_to_read="$3" # Expecting $CONFIG_PATH to be passed here

    local value=""
    local yq_filter_path="$key_path_input"
    local yq_exit_code=0

    # 1. Check if yq and jq commands are available
    if ! command -v yq &> /dev/null || ! command -v jq &> /dev/null; then
        # echo "Warning: yq or jq not found. Using default for $key_path_input from $config_file_to_read." >&2
        echo "$default_value"
        return
    fi

    # 2. Check if the configuration file exists
    if [ ! -f "$config_file_to_read" ]; then
        # echo "Warning: Config file $config_file_to_read not found for $key_path_input. Using default." >&2
        echo "$default_value"
        return
    fi

    # 3. Execute yq with raw output option.
    value=$(yq -r "$yq_filter_path" "$config_file_to_read" 2>/dev/null)
    yq_exit_code=$?

    # 4. Determine if the default value should be used
    if [ $yq_exit_code -ne 0 ] || \
       { [ $yq_exit_code -eq 0 ] && [ -z "$value" ] && [ "$value" != "false" ]; } || \
       [ "$value" = "null" ]; then
        echo "$default_value"
    else
        echo "$value"
    fi
}

# --- MLflow Server Default/Fallback values ---
# These are used if yq/jq are missing, config file missing, or keys not found.
DEFAULT_MLFLOW_SERVER_HOST="127.0.0.1"
DEFAULT_MLFLOW_SERVER_PORT="5001"
DEFAULT_MLFLOW_BACKEND_STORE_URI="./mlruns"
DEFAULT_MLFLOW_DEFAULT_ARTIFACT_ROOT="./mlruns"

# --- Load MLflow Server configurations dynamically ---
echo "===================================================="
echo "Loading MLflow Server Configurations from: ${CONFIG_PATH}"

MLFLOW_SERVER_HOST=$(get_yaml_value '.mlflow_server.host' "$DEFAULT_MLFLOW_SERVER_HOST" "$CONFIG_PATH")
MLFLOW_SERVER_PORT=$(get_yaml_value '.mlflow_server.port' "$DEFAULT_MLFLOW_SERVER_PORT" "$CONFIG_PATH")
MLFLOW_BACKEND_STORE_URI=$(get_yaml_value '.mlflow_server.backend_store_uri' "$DEFAULT_MLFLOW_BACKEND_STORE_URI" "$CONFIG_PATH")
MLFLOW_DEFAULT_ARTIFACT_ROOT=$(get_yaml_value '.mlflow_server.default_artifact_root' "$DEFAULT_MLFLOW_DEFAULT_ARTIFACT_ROOT" "$CONFIG_PATH")

# --- Environment Setup ---
echo "Environment Setup"
echo "===================================================="
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    "${PYTHON_EXECUTABLE}" -m venv .venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install required dependencies (mlflow must be in requirements.txt)
echo "Installing required dependencies from requirements.txt..."
pip install -r requirements.txt --quiet
echo "Dependencies installed."
echo

# --- MLflow Server Setup ---
echo "===================================================="
echo "MLflow Server Setup"
echo "===================================================="

# Set MLFLOW_TRACKING_URI for the pipeline script
export MLFLOW_TRACKING_URI="http://${MLFLOW_SERVER_HOST}:${MLFLOW_SERVER_PORT}"
echo "MLFLOW_TRACKING_URI set to: ${MLFLOW_TRACKING_URI}"

# Ensure MLflow server log directory exists
MLFLOW_SERVER_LOG_DIR="logs/mlflow_server"
mkdir -p "${MLFLOW_SERVER_LOG_DIR}"
MLFLOW_SERVER_LOG_FILE="${MLFLOW_SERVER_LOG_DIR}/mlflow_server_$(date +%Y%m%d_%H%M%S).log"
MLFLOW_SERVER_PID_FILE="${MLFLOW_SERVER_LOG_DIR}/mlflow_server.pid" # File to store PID

# Check if mlflow command is available
if ! command -v mlflow &> /dev/null; then
    echo "----------------------------------------------------"
    echo "ERROR: mlflow command could not be found."
    echo "Please ensure 'mlflow' is listed in your requirements.txt and was installed correctly."
    echo "----------------------------------------------------"
    exit 1
fi

# Check if MLflow server is already running on the specified port
SERVER_ALREADY_RUNNING=false
if command -v nc &> /dev/null && nc -zw1 "${MLFLOW_SERVER_HOST}" "${MLFLOW_SERVER_PORT}" > /dev/null 2>&1; then
    SERVER_ALREADY_RUNNING=true
elif command -v ss &> /dev/null && ss -tuln | grep -q "${MLFLOW_SERVER_HOST}:${MLFLOW_SERVER_PORT}"; then
    SERVER_ALREADY_RUNNING=true
fi

if ${SERVER_ALREADY_RUNNING}; then
    echo "MLflow server appears to be already running on ${MLFLOW_TRACKING_URI}."
else
    echo "Starting MLflow Tracking Server..."
    echo "  Host: ${MLFLOW_SERVER_HOST}"
    echo "  Port: ${MLFLOW_SERVER_PORT}"
    echo "  Backend Store URI: ${MLFLOW_BACKEND_STORE_URI}"
    echo "  Default Artifact Root: ${MLFLOW_DEFAULT_ARTIFACT_ROOT}"
    echo "  Server logs will be at: ${MLFLOW_SERVER_LOG_FILE}"

    # Create artifact root and backend store URI if they are local paths and don't exist
    # (MLflow server usually creates these, but good practice to ensure parent dirs if needed)
    if [[ "${MLFLOW_DEFAULT_ARTIFACT_ROOT}" == ./* ]] || [[ "${MLFLOW_DEFAULT_ARTIFACT_ROOT}" == /* ]]; then
        mkdir -p "${MLFLOW_DEFAULT_ARTIFACT_ROOT}"
    fi
    if [[ "${MLFLOW_BACKEND_STORE_URI}" == ./* ]] && [[ "${MLFLOW_BACKEND_STORE_URI}" != sqlite* ]]; then
        mkdir -p "${MLFLOW_BACKEND_STORE_URI}"
    fi

    nohup mlflow server \
        --host "${MLFLOW_SERVER_HOST}" \
        --port "${MLFLOW_SERVER_PORT}" \
        --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
        --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}" \
        > "${MLFLOW_SERVER_LOG_FILE}" 2>&1 &

    SERVER_PID=$!
    echo "${SERVER_PID}" > "${MLFLOW_SERVER_PID_FILE}"
    echo "MLflow server started in background with PID ${SERVER_PID}."
    echo "Waiting for MLflow server to initialize (5 seconds)..."
    sleep 5

    # Check again if the server started successfully
    SERVER_STARTED_SUCCESSFULLY=false
    if command -v nc &> /dev/null && nc -zw1 "${MLFLOW_SERVER_HOST}" "${MLFLOW_SERVER_PORT}" > /dev/null 2>&1; then
        SERVER_STARTED_SUCCESSFULLY=true
    elif command -v ss &> /dev/null && ss -tuln | grep -q "${MLFLOW_SERVER_HOST}:${MLFLOW_SERVER_PORT}"; then
        SERVER_STARTED_SUCCESSFULLY=true
    fi

    if ! ${SERVER_STARTED_SUCCESSFULLY}; then
        echo "----------------------------------------------------"
        echo "ERROR: MLflow server failed to start or is not accessible after attempting to start."
        echo "Please check server logs at: ${MLFLOW_SERVER_LOG_FILE}"
        echo "And ensure host '${MLFLOW_SERVER_HOST}' and port '${MLFLOW_SERVER_PORT}' are correct and available."
        echo "----------------------------------------------------"
        if [ -f "${MLFLOW_SERVER_PID_FILE}" ]; then rm "${MLFLOW_SERVER_PID_FILE}"; fi
        exit 1
    else
        echo "MLflow server confirmed to be accessible."
    fi
fi
echo

# --- Script Execution ---
echo "===================================================="
echo "Starting Bank Term Deposit Subscription Prediction Pipeline"
echo "Using configuration file: ${CONFIG_PATH}"
echo "Current time: $(date)"
echo "===================================================="
echo

PYTHON_COMMAND="${PYTHON_EXECUTABLE} src/pipeline.py --config ${CONFIG_PATH}"

# Execute the command
echo "Executing command: ${PYTHON_COMMAND}"
echo "----------------------------------------------------"
${PYTHON_COMMAND}
echo "----------------------------------------------------"

# Capture exit status
EXIT_STATUS=$?

if [ "${EXIT_STATUS}" -eq 0 ]; then
  echo "Pipeline executed successfully."
else
  echo "Pipeline execution failed with exit code ${EXIT_STATUS}."
fi
echo

echo "===================================================="
echo "Pipeline run finished at: $(date)"
echo "===================================================="
echo

# --- Optional: Stop the MLflow server ---
# if [ -f "${MLFLOW_SERVER_PID_FILE}" ] && ! ${SERVER_ALREADY_RUNNING} ; then
#   # Only stop it if this script started it (PID file exists and server wasn't already running)
#   STORED_PID=$(cat "${MLFLOW_SERVER_PID_FILE}")
#   echo "Attempting to stop MLflow server (PID: ${STORED_PID}) started by this script..."
#   if kill "${STORED_PID}" ; then
#       echo "Kill signal sent to MLflow server PID ${STORED_PID}."
#       TIMEOUT=10; ELAPSED=0
#       while kill -0 "${STORED_PID}" 2>/dev/null && [ "${ELAPSED}" -lt "${TIMEOUT}" ]; do
#           sleep 1; ELAPSED=$((ELAPSED + 1)); done
#       if ! kill -0 "${STORED_PID}" 2>/dev/null; then echo "MLflow server stopped."; else
#           echo "MLflow server (PID ${STORED_PID}) did not stop gracefully. Consider 'kill -9 ${STORED_PID}'."; fi
#   else
#       echo "Failed to send kill signal to MLflow server PID ${STORED_PID}. It might have already stopped."
#   fi
#   rm -f "${MLFLOW_SERVER_PID_FILE}"
# else
#   if ${SERVER_ALREADY_RUNNING}; then
#       echo "MLflow server was already running; not stopping it."
#   else # No PID file, but server was not detected as already running (should not happen if start logic is correct)
#       echo "MLflow server was not started by this script or its PID was not captured."
#   fi
#   echo "If needed, stop any MLflow server manually (e.g., pkill -f \"mlflow server.*--port ${MLFLOW_SERVER_PORT}\")."
# fi

exit ${EXIT_STATUS}