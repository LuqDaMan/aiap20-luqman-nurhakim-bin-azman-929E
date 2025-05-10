# trc/utils.py
import yaml
import logging
import os
import sys
import random
import numpy as np
from logging.handlers import RotatingFileHandler


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
        with open(config_path, "r") as f:
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

def setup_logging(config: dict, logger_name: str = None) -> logging.Logger:
    """
    Sets up logging for the pipeline based on the configuration.
    Logs to both console and a file with rotation support.
    
    Args:
        config (dict): The pipeline configuration dictionary.
                       Expected keys:
                       - 'logging' containing:
                           - 'level' (e.g., "INFO", "DEBUG")
                           - 'file' (path to the log file)
                           - 'format' (optional log format)
                           - 'max_size_mb' (optional, max log file size in MB)
                           - 'backup_count' (optional, number of backup files)
        logger_name (str, optional): Name for the logger. If None, uses root logger.
    
    Returns:
        logging.Logger: Configured logger
    """
    try:
        # Extract config values with sensible defaults
        log_config = config.get('logging', {})
        log_level_str = log_config.get('level', 'DEBUG').upper()
        log_file_path = log_config.get('file')
        log_format_str = log_config.get('format', 
                                       '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        max_size_mb = log_config.get('max_size_mb', 10)
        backup_count = log_config.get('backup_count', 3)
        
        # Get log level
        log_level = getattr(logging, log_level_str, logging.DEBUG)
        
        # Get logger (root or named)
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        
        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(log_format_str)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (with rotation)
        if log_file_path:
            # Create log directory if needed
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            
            # Set up rotating file handler
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=max_size_mb * 1024 * 1024,  # Convert MB to bytes
                backupCount=backup_count,
                mode='a'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        logger.info(f"Logging setup complete. Level: {log_level_str}. "
                   f"Log file: {log_file_path if log_file_path else 'None'}")
        return logger
        
    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(logger_name)
        logger.error(f"Error setting up custom logging: {e}. Falling back to basicConfig.")
        print(f"CRITICAL: Error setting up logging: {e}", file=sys.stderr)
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