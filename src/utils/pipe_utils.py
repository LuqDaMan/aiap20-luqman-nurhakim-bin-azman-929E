# src/utils/pipe_utils.py
import yaml
import logging
import os
import sys
import random
import numpy as np
from logging.handlers import RotatingFileHandler
from datetime import datetime


def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading config from {config_path}: {e}")
        raise

def setup_logging(
    logger_specific_config: dict,
    common_logging_config: dict,
) -> logging.Logger:
    """
    Sets up a specific logger based on the provided configurations.
    Logs to both console and a rotating file with optional timestamp in filename.

    Args:
        logger_specific_config (dict): Configuration specific to this logger.
            Expected keys:
            - 'logger_name' (str): Name for the logger.
            - 'level' (str, e.g., "INFO", "DEBUG"): Logging level.
            - 'log_dir' (str): Directory to store log files.
            - 'log_file_base_name' (str): Base name for the log file.
        common_logging_config (dict): Common logging configurations.
            Expected keys:
            - 'format' (str): Log format string.
            - 'date_format' (str, optional): Date format for log messages.
            - 'timestamp_log_files' (bool, optional): If True, appends a timestamp to log filenames.
            - 'max_size_bytes' (int, optional): Max log file size in bytes for rotation.
            - 'backup_count' (int, optional): Number of backup log files for rotation.

    Returns:
        logging.Logger: Configured logger.

    Raises:
        ValueError: If essential configuration keys are missing.
    """
    try:
        # Extract logger-specific config values
        logger_name = logger_specific_config.get('logger_name')
        if not logger_name:
            raise ValueError("'logger_name' is not specified in logger_specific_config.")

        log_level_str = logger_specific_config.get('level', 'INFO').upper()
        log_dir = logger_specific_config.get('log_dir')
        if not log_dir:
            raise ValueError(f"'log_dir' is not specified for logger '{logger_name}'.")
        log_file_base_name = logger_specific_config.get('log_file_base_name')
        if not log_file_base_name:
            raise ValueError(f"'log_file_base_name' is not specified for logger '{logger_name}'.")

        # Extract common logging config values
        log_format_str = common_logging_config.get(
            'format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        date_format_str = common_logging_config.get('date_format', '%Y-%m-%d %H:%M:%S')
        timestamp_log_files = common_logging_config.get('timestamp_log_files', False)
        max_bytes = common_logging_config.get('max_size_bytes', 10 * 1024 * 1024)  # Default 10MB
        backup_count = common_logging_config.get('backup_count', 3)

        # Get log level from logging module
        log_level = getattr(logging, log_level_str, logging.INFO)

        # Get the named logger instance
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level) # Set level for the logger

        # Clear existing handlers to prevent duplicate logging if re-configured
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Propagate logs to parent loggers or not. Default is True.
        # For application specific loggers, you might want to set this to False
        # if you have a root logger also configured and want to avoid duplicate messages.
        # For now, we keep the default (True).
        # logger.propagate = False

        # Create formatter
        formatter = logging.Formatter(log_format_str, datefmt=date_format_str)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level) # Handler level can also be set
        logger.addHandler(console_handler)

        # File handler (with rotation and optional timestamp)
        os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists

        if timestamp_log_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(log_file_base_name)
            log_file_name = f"{base}_{timestamp}{ext}"
        else:
            log_file_name = log_file_base_name
        
        log_file_path = os.path.join(log_dir, log_file_name)

        # Set up rotating file handler
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            mode='a', # Append mode
            encoding='utf-8' # Specify encoding
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level) # Handler level
        logger.addHandler(file_handler)

        logger.info(
            f"Logging for '{logger_name}' setup complete. Level: {log_level_str}. "
            f"Log file: {log_file_path}"
        )
        return logger

    except Exception as e:
        # Fallback to basic logging configuration if custom setup fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s (fallback)',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Attempt to use the intended logger name or a default if not available
        fallback_logger_name = logger_name if 'logger_name' in locals() and logger_name else "utils_fallback_logger"
        logger = logging.getLogger(fallback_logger_name)
        
        error_msg = (f"CRITICAL: Error setting up custom logging for '{fallback_logger_name}': {e}. "
                     f"Falling back to basicConfig. Some logs may be missed or improperly formatted.")
        logger.error(error_msg, exc_info=True) # Log with exception info
        # Also print to stderr as a direct warning to the console user
        print(error_msg, file=sys.stderr)
        return logger


def set_global_random_seed(seed_value: int) -> None:
    """
    Sets the random seed for common libraries to ensure reproducibility.
    Libraries covered: random, numpy.

    Args:
        seed_value (int): The seed value to use.
    """
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        # If using TensorFlow or PyTorch, add their seed settings here:
        # For TensorFlow:
        # import tensorflow as tf
        # tf.random.set_seed(seed_value)
        #
        # For PyTorch:
        # import torch
        # torch.manual_seed(seed_value)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(seed_value) # for multi-GPU.
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False

        logging.info(f"Global random seed set to {seed_value} for random and numpy.")
    except Exception as e:
        logging.error(f"Error setting global random seed: {e}")
        raise

# Example usage (optional, for testing the module directly)
if __name__ == '__main__':
    # This part will only run if utils.py is executed directly
    # Create a dummy config for testing logging
    mock_config_for_logging = {
        "logging_level": "DEBUG",
        "output_paths": {
            "log_file": "logs/utils_test.log"
        },
        "random_seed": 123
    }
    
    # Setup logging using the mock config
    setup_logging(mock_config_for_logging) # Logger will be set up using this config

    logging.debug("This is a debug message from utils.py direct execution.")
    logging.info("This is an info message from utils.py direct execution.")
    
    # Test config loading (assuming a dummy config.yaml exists at the specified path)
    # For this example, we'll just show how you might call it.
    # You'd need to create a 'config/pipeline_config.yaml' or adjust path for this test.
    try:
        # Ensure a dummy config exists or point to the actual one if available
        # For isolated testing of utils.py, you might create a temporary dummy config
        # For now, we'll assume a simple config loading test that might fail if config doesn't exist
        # but shows the intent.
        
        # Create a dummy config file for the test
        if not os.path.exists("config"):
            os.makedirs("config")
        with open("config/pipeline_config.yaml", "w") as f:
            yaml.dump({"test_param": "hello world", **mock_config_for_logging}, f)

        cfg = load_config("config/pipeline_config.yaml") # Now uses the logger set by setup_logging
        logging.info(f"Test config loaded: {cfg.get('test_param')}")

        set_global_random_seed(cfg.get("random_seed"))
        logging.info(f"Numpy random: {np.random.rand(1)}")


    except Exception as e:
        logging.error(f"Error in utils.py __main__ example: {e}")