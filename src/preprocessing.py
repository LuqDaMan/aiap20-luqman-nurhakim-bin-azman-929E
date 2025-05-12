# src/preprocessing.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('pipeline.preprocessing') 

# --- Helper Functions for Preprocessing Steps ---

def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes all column names to lowercase_with_underscores."""
    original_columns = df.columns.tolist()
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^0-9a-zA-Z_]', '', regex=True)
    standardized_columns = df.columns.tolist()
    if original_columns != standardized_columns:
        logger.info("Standardized column names.")
        logger.debug(f"Original columns: {original_columns}")
        logger.debug(f"Standardized columns: {standardized_columns}")
    return df

def _preprocess_age(df: pd.DataFrame, age_config: dict) -> pd.DataFrame:
    """Processes the age column: removes ' years', converts to numeric, handles anomalies and NaNs."""
    age_col = age_config.get('original_col', 'age')
    cleaned_age_col = age_config.get('cleaned_col', 'cleaned_age')
    anomaly_threshold = age_config.get('anomaly_threshold', 100)

    if age_col not in df.columns:
        logger.warning(f"Original age column '{age_col}' not found. Skipping age preprocessing.")
        return df

    logger.debug(f"Processing age column: '{age_col}' into '{cleaned_age_col}'. Anomaly threshold: {anomaly_threshold}")
    
    # Ensure it's string type before replacing, then coerce to numeric
    df[age_col] = df[age_col].astype(str).str.replace(' years', '', regex=False)
    df[age_col] = pd.to_numeric(df[age_col], errors='coerce') # Coerce will turn empty strings and non-numeric to NaN

    # Calculate median on valid ages (<= threshold and not NaN)
    valid_ages = df.loc[(df[age_col] <= anomaly_threshold) & (df[age_col].notna()), age_col]
    if not valid_ages.empty:
        age_median = valid_ages.median()
        logger.debug(f"Median age (for values <= {anomaly_threshold} and non-NaN): {age_median}")
    else:
        age_median = df[age_col].median() # Median of all numeric ages, whatever they are
        logger.warning(f"No ages found <= {anomaly_threshold}. Using overall median {age_median} for imputation.")

    # Create cleaned_age column
    df[cleaned_age_col] = df[age_col]
    
    # Handle anomalies (> threshold)
    anomalous_age_count = df[df[cleaned_age_col] > anomaly_threshold].shape[0]
    if anomalous_age_count > 0:
        logger.info(f"Found {anomalous_age_count} age values > {anomaly_threshold}. Imputing with median: {age_median}.")
        df.loc[df[cleaned_age_col] > anomaly_threshold, cleaned_age_col] = age_median
        
    # Handle NaNs (which might have been original NaNs, empty strings, or non-numeric strings)
    nan_age_count = df[cleaned_age_col].isna().sum()
    if nan_age_count > 0:
        logger.info(f"Found {nan_age_count} NaN values in '{cleaned_age_col}'. Imputing with median: {age_median}.")
        df[cleaned_age_col] = df[cleaned_age_col].fillna(age_median)

    df = df.drop(columns=[age_col])
    logger.info(f"Age column '{age_col}' processed into '{cleaned_age_col}'.")
    return df

def _standardize_categorical_strings(df: pd.DataFrame, cat_cols_config: list, contact_method_config: dict) -> pd.DataFrame:
    """Standardizes string values in specified categorical columns (lowercase, strip whitespace, consolidate)."""
    logger.debug(f"Standardizing categorical columns: {cat_cols_config}")
    for col in cat_cols_config:
        if col in df.columns:
            original_dtype = df[col].dtype
            # Convert to string type first to handle potential mixed types gracefully
            df[col] = df[col].astype(str).str.lower().str.strip()
            
            if col == 'contact_method' and contact_method_config:
                for K, V in contact_method_config.items():
                    df[col] = df[col].replace(K, V)
                logger.info(f"Consolidated values in 'contact_method' using mapping: {contact_method_config}")
            
            # "unknown" string values are standardized here. Their treatment as a distinct category happens in encoding.
            logger.debug(f"Standardized categorical column: '{col}'. 'unknown' values standardized if present.")
        else:
            logger.warning(f"Categorical column '{col}' for standardization not found in DataFrame. Skipping.")
    return df

def _handle_loan_columns(df: pd.DataFrame, loan_cols_config: dict) -> pd.DataFrame:
    """Handles missing (NaN) and 'unknown' string values in loan columns by mapping them to a specified category."""
    logger.debug(f"Handling missing/unknown in loan columns: {list(loan_cols_config.keys())}")
    for col, replacement_value in loan_cols_config.items():
        if col in df.columns:
            # Ensure column is string type for consistent 'unknown' string handling
            # Fill NaNs with the string 'nan_temp_marker' to differentiate from actual "unknown" strings before lowercasing
            df[col] = df[col].fillna('nan_temp_marker')
            df[col] = df[col].astype(str).str.lower().str.strip()
            
            # Map both original 'unknown' strings and NaNs (now 'nan_temp_marker') to the target replacement_value
            condition = (df[col] == 'unknown') | (df[col] == 'nan_temp_marker')
            num_affected = df[condition].shape[0]
            if num_affected > 0:
                 logger.info(f"Mapping {num_affected} NaN or 'unknown' string values in '{col}' to '{replacement_value}'.")
            df.loc[condition, col] = replacement_value.lower().strip()
           
        else:
            logger.warning(f"Loan column '{col}' for missing/unknown handling not found. Skipping.")
    return df

def _preprocess_campaign_calls(df: pd.DataFrame, campaign_calls_config: dict) -> pd.DataFrame:
    """Handles negative values in 'campaign_calls' and creates an indicator column."""
    original_col = campaign_calls_config.get('original_col', 'campaign_calls')
    indicator_col = campaign_calls_config.get('negative_adjustment_indicator_col', 'cc_had_negative_adjustment')

    if original_col not in df.columns:
        logger.warning(f"Campaign calls column '{original_col}' not found. Skipping its preprocessing.")
        return df

    # Ensure numeric, coercing errors. This handles if column was read as object.
    df[original_col] = pd.to_numeric(df[original_col], errors='coerce')
    
    nan_before_abs = df[original_col].isna().sum()
    
    # Create indicator for negative values (before abs())
    # Consider NaNs as not having a negative adjustment
    df[indicator_col] = np.where(df[original_col] < 0, True, False)
    df.loc[df[original_col].isna(), indicator_col] = False # NaNs did not have negative adjustment
    
    negative_count = (df[original_col] < 0).sum()
    if negative_count > 0:
        logger.info(f"Found {negative_count} negative values in '{original_col}'. Taking absolute values and created '{indicator_col}'.")
    
    df[original_col] = df[original_col].abs()
    
    # Handle NaNs that might have been coerced or were original (impute with median of absolute values)
    # It's better to handle NaNs here if they can occur to prevent issues downstream.
    # PRD does not specify imputation for campaign_calls, but it's safer.
    # Using median of non-NaN absolute values.
    if df[original_col].isna().sum() > 0 :
        median_calls = df[original_col].median()
        if pd.isna(median_calls): median_calls = 0 # Fallback if all are NaN
        logger.info(f"Found {df[original_col].isna().sum()} NaN values in '{original_col}' after abs(). Imputing with median: {median_calls}.")
        df[original_col] = df[original_col].fillna(median_calls)
        
    logger.info(f"Processed '{original_col}': negatives handled, '{indicator_col}' created.")
    return df

def _preprocess_previous_contact_days(df: pd.DataFrame, prev_contact_config: dict) -> pd.DataFrame:
    """Handles special value 999 in 'previous_contact_days', creates 'previously_contacted' indicator."""
    original_col = prev_contact_config.get('original_col', 'previous_contact_days')
    special_value = prev_contact_config.get('special_value_no_contact', 999)
    indicator_col = prev_contact_config.get('previously_contacted_indicator_col', 'previously_contacted')
    replacement_val = prev_contact_config.get('value_for_no_contact_after_indicator', 0)

    if original_col not in df.columns:
        logger.warning(f"Previous contact days column '{original_col}' not found. Skipping its preprocessing.")
        return df
    
    # Ensure numeric, coercing errors
    df[original_col] = pd.to_numeric(df[original_col], errors='coerce')
    
    # Create 'previously_contacted' indicator
    # True if not special_value AND not NaN. False if special_value. False if NaN (treat NaN as not previously contacted for indicator)
    df[indicator_col] = np.where(
        (df[original_col] != special_value) & (df[original_col].notna()), 
        True, 
        False
    )
    
    count_999 = (df[original_col] == special_value).sum()
    logger.info(f"Created '{indicator_col}' based on '{original_col}' (special value {special_value}).")
    
    # Replace special_value (e.g., 999) with another value (e.g., 0 or NaN) as per DS-FE-001
    # "Replace 999 with 0 after creation of new feature."
    if count_999 > 0:
        logger.info(f"Replacing {count_999} occurrences of {special_value} in '{original_col}' with {replacement_val}.")
    df.loc[df[original_col] == special_value, original_col] = replacement_val
    
    # Handle original NaNs in 'previous_contact_days' if any, after 999 replacement.
    # PRD DS-FE-001 "For clients who were previously contacted, this feature is numeric (days). Replace 999 with 0"
    # This implies if it was NaN originally, and indicator is False, it should probably be 0 as well, or handled carefully.
    # If it was NaN and indicator is False, let's set it to `replacement_val` too.
    nan_mask = df[original_col].isna()
    if nan_mask.sum() > 0:
        logger.info(f"Found {nan_mask.sum()} NaN values in '{original_col}'. Setting to {replacement_val} as they are also 'not previously contacted'.")
        df.loc[nan_mask, original_col] = replacement_val

    logger.info(f"Processed '{original_col}': '{indicator_col}' created, special value {special_value} handled.")
    return df

def _encode_target_variable(df: pd.DataFrame, target_col: str, encoding_map: dict) -> pd.DataFrame:
    """Encodes the target variable using the provided mapping."""
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found. Skipping target encoding.")
        return df
    
    # Standardize target column values before mapping (e.g. lowercase, strip)
    # This is important if raw data isn't perfectly clean e.g. "Yes", " no "
    df[target_col] = df[target_col].astype(str).str.lower().str.strip()

    original_values = df[target_col].unique()
    df[target_col] = df[target_col].map(encoding_map)
    
    if df[target_col].isnull().any():
        logger.warning(f"NaNs introduced in target column '{target_col}' after mapping. Original values: {original_values}. Map: {encoding_map}. Check data and mapping.")
        # Decide on a strategy: raise error, fill with a default, or leave as NaN if subsequent steps handle it.
        # For now, logging a warning. Critical if this happens.
    logger.info(f"Target variable '{target_col}' encoded using map: {encoding_map}.")
    return df

def _drop_client_id(df: pd.DataFrame, client_id_col: str) -> pd.DataFrame:
    """Drops the client_id column if it exists."""
    if client_id_col in df.columns:
        df = df.drop(columns=[client_id_col])
        logger.info(f"Dropped '{client_id_col}' column.")
    else:
        logger.info(f"Client ID column '{client_id_col}' not found for dropping, or not specified.")
    return df

# --- Main Preprocessing Function ---

def preprocess_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Performs data cleaning and preprocessing on the raw DataFrame using configurations.

    Args:
        df (pd.DataFrame): The raw input DataFrame.
        config (dict): The pipeline configuration dictionary.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """      
    logger.info("Starting data preprocessing...")
    df_processed = df.copy()

    # Step 1: Standardize column names
    # This is crucial as config keys often assume standardized column names.
    df_processed = _standardize_column_names(df_processed)
    
    # Subsequent steps use config keys that should match standardized column names in the DataFrame.

    # Step 2: Handle 'age'
    if 'age_processing' in config:
        df_processed = _preprocess_age(df_processed, config['age_processing'])
    else:
        logger.warning("Age processing configuration not found. Skipping age preprocessing.")

    # Step 3: Standardize categorical string values
    df_processed = _standardize_categorical_strings(
        df_processed,
        config.get('categorical_cols_to_standardize', []),
        config.get('contact_method_consolidation', {})
    )

    # Step 4: Handle missing and "unknown" in loan columns
    if 'loan_cols_missing_handling' in config:
        df_processed = _handle_loan_columns(df_processed, config['loan_cols_missing_handling'])
    else:
        logger.warning("Loan columns missing handling configuration not found. Skipping.")

    # Step 5: Handle 'campaign_calls'
    if 'campaign_calls_processing' in config:
        df_processed = _preprocess_campaign_calls(df_processed, config['campaign_calls_processing'])
    else:
        logger.warning("Campaign calls processing configuration not found. Skipping.")
    
    # Step 6: Handle 'previous_contact_days'
    if 'previous_contact_days_processing' in config:
        df_processed = _preprocess_previous_contact_days(df_processed, config['previous_contact_days_processing'])
    else:
        logger.warning("Previous contact days processing configuration not found. Skipping.")

    # Step 7: Encode target variable
    target_col = config.get('target_column') # Assumes this key in config matches standardized name
    encoding_map = config.get('target_variable_encoding')
    if target_col and encoding_map: # target_col should now be 'subscription_status'
        df_processed = _encode_target_variable(df_processed, target_col, encoding_map)
    else:
        logger.warning(f"Target column '{target_col}' or encoding map not fully specified in config. Skipping target encoding.")
        if target_col and target_col not in df_processed.columns:
             logger.warning(f"Target column '{target_col}' (expected standardized) not found in DataFrame columns: {df_processed.columns.tolist()}")


    # Step 8: Drop client_id
    client_id_col = config.get('client_id_column') # Assumes this key in config matches standardized name
    if client_id_col: # client_id_col should now be 'client_id'
        df_processed = _drop_client_id(df_processed, client_id_col)
    else:
        logger.info("Client ID column for dropping not specified in config.")

    logger.info(f"Data preprocessing complete. DataFrame shape after preprocessing: {df_processed.shape}")
    if not df_processed.empty:
        logger.debug(f"First 5 rows of preprocessed data:\n{df_processed.head().to_string()}")
    else:
        logger.warning("DataFrame is empty after preprocessing.")
        
    return df_processed