# src/utils.py

import os
import random
import joblib
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def set_seeds(seed: int = 42) -> None:
    """
    Sets the random seeds for Python, NumPy, and other potential libraries
    to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seeds set to: {seed}")

    # Add for other libraries if used, e.g., TensorFlow, PyTorch
    # try:
    #     import tensorflow as tf
    #     tf.random.set_seed(seed)
    #     logger.info(f"TensorFlow random seed set to: {seed}")
    # except ImportError:
    #     pass # TensorFlow not installed
    #
    # try:
    #     import torch
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed_all(seed) # for multi-GPU
    #         torch.backends.cudnn.deterministic = True
    #         torch.backends.cudnn.benchmark = False
    #     logger.info(f"PyTorch random seed set to: {seed}")
    # except ImportError:
    #     pass # PyTorch not installed

def save_object(obj: any, filepath: str) -> None:
    """
    Saves a Python object to a file using joblib.

    Args:
        obj (any): The Python object to save.
        filepath (str): The path to the file where the object will be saved.
                        Directory will be created if it doesn't exist.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(obj, filepath)
        logger.info(f"Object saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Error saving object to {filepath}: {e}")
        raise

def load_object(filepath: str) -> any:
    """
    Loads a Python object from a file using joblib.

    Args:
        filepath (str): The path to the file from which to load the object.

    Returns:
        any: The loaded Python object.
    """
    try:
        obj = joblib.load(filepath)
        logger.info(f"Object loaded successfully from {filepath}")
        return obj
    except FileNotFoundError:
        logger.error(f"Error loading object: File not found at {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading object from {filepath}: {e}")
        raise

if __name__ == '__main__':
    # Example usage (optional, for direct testing of this module)
    logger.info("Testing utils module...")

    # Test set_seeds
    set_seeds(123)
    logger.info(f"Numpy random number after seed: {np.random.rand()}")

    # Test save_object and load_object
    test_obj = {"key": "value", "numbers": [1, 2, 3]}
    temp_filepath = "temp/temp_test_object.joblib"

    save_object(test_obj, temp_filepath)
    loaded_obj = load_object(temp_filepath)

    assert test_obj == loaded_obj
    logger.info(f"Loaded object: {loaded_obj}")

    # Clean up
    if os.path.exists(temp_filepath):
        os.remove(temp_filepath)
    logger.info("Utils module test complete.")