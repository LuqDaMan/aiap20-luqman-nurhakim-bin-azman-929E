# src/data_ingestion.py
import logging
import sqlite3
import pandas as pd
import os # For checking if DB file exists


def ingest_data(data_source_config: dict) -> pd.DataFrame:
    """
    Ingests data from the specified SQLite database and table.

    Args:
        data_source_config (dict): A dictionary containing data source parameters.
                                   Expected keys: 'db_path', 'table_name'.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the database file does not exist.
        sqlite3.Error: If there's an error related to SQLite database operations.
        KeyError: If required keys are missing from data_source_config.
        Exception: For any other unexpected errors during data ingestion.
    """
    try:
        db_path = data_source_config['db_path']
        table_name = data_source_config['table_name']

        # SETUP LOGGING
        logger = logging.getLogger('pipeline.data_ingestion')
        
        logger.info(f"Starting data ingestion from database: {db_path}, table: {table_name}")

        if not os.path.exists(db_path):
            logger.error(f"Database file not found at path: {db_path}")
            raise FileNotFoundError(f"Database file not found at path: {db_path}")

        conn = None # Initialize conn to None to ensure it's defined in finally block
        try:
            conn = sqlite3.connect(db_path)
            query = f"SELECT * FROM {table_name}"
            logger.debug(f"Executing query: {query}")
            
            df = pd.read_sql_query(query, conn)
            
            logger.info(f"Data ingested successfully from table '{table_name}'.")
            logger.info(f"DataFrame shape: {df.shape}")
            if df.empty:
                logger.warning(f"The DataFrame loaded from table '{table_name}' is empty.")
            else:
                logger.debug(f"First 5 rows of ingested data:\n{df.head().to_string()}")
            
            return df

        except sqlite3.Error as e:
            logger.error(f"SQLite error during data ingestion from table '{table_name}': {e}")
            raise 
        finally:
            if conn:
                conn.close()
                logger.debug(f"Database connection to {db_path} closed.")
                
    except KeyError as e:
        logger.error(f"Missing required key in data_source_config: {e}")
        raise 
    except Exception as e:
        logger.error(f"An unexpected error occurred during data ingestion: {e}")
        raise 