#!/bin/bash

# Script to run the Bank Term Deposit Subscription Prediction Pipeline

# --- Configuration ---
DEFAULT_CONFIG_PATH="config/pipeline_config.yaml"
CONFIG_PATH="${1:-$DEFAULT_CONFIG_PATH}"

# --- Environment Setup ---
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install required dependencies
echo "Installing required dependencies..."
pip install -r requirements.txt

# --- Script Execution ---
echo "===================================================="
echo "Starting Bank Term Deposit Subscription Prediction Pipeline"
echo "Using configuration file: ${CONFIG_PATH}"
echo "Current time: $(date)"
echo "===================================================="
echo

PYTHON_COMMAND="python3 src/pipeline.py --config ${CONFIG_PATH}"

# Execute the command
echo "Executing command: ${PYTHON_COMMAND}"
echo "----------------------------------------------------"
${PYTHON_COMMAND}
echo "----------------------------------------------------"

# Capture exit status
EXIT_STATUS=$?

if [ ${EXIT_STATUS} -eq 0 ]; then
  echo "Pipeline executed successfully."
else
  echo "Pipeline execution failed with exit code ${EXIT_STATUS}."
fi

echo "===================================================="
echo "Pipeline run finished at: $(date)"
echo "===================================================="

exit ${EXIT_STATUS}