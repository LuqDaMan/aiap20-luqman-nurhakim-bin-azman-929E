# src/data_loader.py
import pandas as pd
from sqlalchemy import create_engine, text
import logging
import os

# Configure logging
# It's good practice to configure logging in a central place (e.g., main.py or pipeline.py)
# For a module, usually, you just get the logger.
# However, if this script is run standalone for testing, basicConfig can be useful.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading data from a SQLite database.
    """
    def __init__(self, db_path: str, table_name: str):
        """
        Initializes the DataLoader with the database path and table name.

        Args:
            db_path (str): The relative path to the SQLite database file
                           (e.g., 'data/your_database_name.db').
                           This path should be relative to the project root.
            table_name (str): The name of the table to load data from.
        """
        # Construct the absolute path to the database from the relative path
        # This assumes db_path is relative to the current working directory
        # when the script/pipeline is run (usually the project root).
        self.db_uri = f"sqlite:///{os.path.abspath(db_path)}"
        self.table_name = table_name
        self.engine = None
        try:
            self.engine = create_engine(self.db_uri)
            logger.info(f"SQLAlchemy engine created for database: {self.db_uri}")
        except Exception as e:
            logger.error(f"Failed to create SQLAlchemy engine for URI '{self.db_uri}': {e}")
            raise ConnectionError(f"Failed to create SQLAlchemy engine: {e}")

    def load_data(self) -> pd.DataFrame:
        """
        Connects to the SQLite database, loads data from the specified table,
        and returns it as a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data.

        Raises:
            ConnectionError: If the database engine was not initialized.
            ValueError: If the table is not found in the database.
            Exception: For other unexpected errors during data loading (e.g., SQL errors).
        """
        if not self.engine:
            logger.error("SQLAlchemy engine not initialized. Cannot load data.")
            raise ConnectionError("Database engine not initialized. Call __init__ first.")

        logger.info(f"Attempting to load data from table '{self.table_name}' using URI '{self.db_uri}'.")
        try:
            # Check if table exists
            with self.engine.connect() as connection:
                # For SQLite, sqlite_master (or sqlite_temp_master for temp tables) holds schema
                result = connection.execute(
                    text(f"SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"),
                    {"table_name": self.table_name}
                ).fetchone()
                if result is None:
                    logger.error(f"Table '{self.table_name}' not found in the database '{self.db_uri}'.")
                    raise ValueError(f"Table '{self.table_name}' not found in the database.")

            # Load data from the table
            query = f"SELECT * FROM {self.table_name}" # Ensure table_name is sanitized if from user input outside config
            df = pd.read_sql_query(query, self.engine)
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns from table '{self.table_name}'.")
            if df.empty:
                logger.warning(f"Loaded DataFrame from '{self.table_name}' is empty.")
            return df
        except ValueError as ve: # Re-raise ValueError if table not found
            logger.error(f"ValueError during data loading: {ve}")
            raise ve
        except Exception as e: # Catch other pandas or SQLAlchemy errors
            logger.error(f"Error loading data from table '{self.table_name}': {e}")
            # Wrap in a custom exception or re-raise
            raise Exception(f"Could not load data from table '{self.table_name}': {e}")

if __name__ == '__main__':
    # This is an example of how to use the DataLoader.
    # It assumes you have a 'data' folder in your project root with 'bmarket.db'.
    # For the actual pipeline, DataLoader would be instantiated and used within pipeline.py or main.py,
    # with configuration typically coming from a config file.

    logger.info("Starting DataLoader example execution...")

    # Configuration for the example (these would come from config files in a real pipeline)
    db_file_path = os.path.join("data", "bmarket.db")
    
    actual_table_name = "bank_marketing" 

    logger.info(f"Attempting to load data from DB: '{db_file_path}', Table: '{actual_table_name}'")

    # Check if the database file exists
    if not os.path.exists(db_file_path):
        logger.error(f"Database file not found at '{os.path.abspath(db_file_path)}'.")
        logger.error("Please ensure the database 'bank_marketing_data.db' exists in the 'data' directory "
                     "relative to your project root.")
        logger.error("If you are running this script directly, ensure you are in the project root directory.")
    else:
        logger.info(f"Database file found at '{os.path.abspath(db_file_path)}'.")
        try:
            # Instantiate the DataLoader
            data_loader = DataLoader(db_path=db_file_path, table_name=actual_table_name)
            
            # Load the data
            client_data_df = data_loader.load_data()
            
            logger.info(f"\nSuccessfully loaded data from '{actual_table_name}':")
            logger.info(f"Data head:\n{client_data_df.head()}")
            logger.info(f"Data shape: {client_data_df.shape}")
            logger.info(f"Data columns: {client_data_df.columns.tolist()}")

            # Example of trying to load a non-existent table to test error handling
            non_existent_table = "this_table_does_not_exist"
            logger.info(f"\nAttempting to load a non-existent table ('{non_existent_table}') (expected to fail)...")
            try:
                error_loader = DataLoader(db_path=db_file_path, table_name=non_existent_table)
                error_loader.load_data()
            except ValueError as e:
                logger.warning(f"Caught expected ValueError for non-existent table: {e}")
            except Exception as e:
                logger.error(f"Caught unexpected error while testing non-existent table: {e}")

        except ConnectionError as e:
            logger.error(f"A ConnectionError occurred: {e}")
            logger.error("This might be due to issues with the SQLite driver or database file path.")
        except ValueError as e:
            logger.error(f"A ValueError occurred during loading (e.g., table not found): {e}")
            logger.error(f"Ensure the table name '{actual_table_name}' is correct for the database '{db_file_path}'.")
        except FileNotFoundError as e: # Should be caught by the os.path.exists check, but as a safeguard
            logger.error(f"Database file not found. Please ensure '{db_file_path}' exists. Error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during the DataLoader example: {e}")
            logger.error("Traceback:", exc_info=True)

    logger.info("DataLoader example execution finished.")