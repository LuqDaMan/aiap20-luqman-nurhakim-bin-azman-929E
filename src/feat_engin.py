# src/feature_engineering.py
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Any, Union # For type hinting

logger = logging.getLogger('pipeline.feat_engin')

def engineer_features_and_split_data(
    df: pd.DataFrame,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Performs feature engineering (encoding, scaling) and splits data into
    training and testing sets.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        config (dict): The pipeline configuration dictionary.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
            X_train_processed_df: Processed training features.
            X_test_processed_df: Processed testing features.
            y_train: Training target variable.
            y_test: Testing target variable.
            preprocessor: The fitted scikit-learn ColumnTransformer object.
    """
    logger.info("Starting feature engineering and data splitting...")

    target_col = config.get('target_column')
    if not target_col or target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in DataFrame or not specified in config.")
        raise ValueError(f"Target column '{target_col}' not found or not specified.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"Separated features (X shape: {X.shape}) and target (y shape: {y.shape}).")

    # Identify feature lists from config (ensure they exist in X's columns)
    numerical_cols = [col for col in config.get('numerical_features_for_scaling', []) if col in X.columns]
    nominal_cols = [col for col in config.get('nominal_features_for_onehot', []) if col in X.columns]

    # Ordinal features (example: education_level)
    ordinal_col_definitions = { # Could be expanded if more ordinal features
        'education_level': config.get('education_level_order', [])
    }
    ordinal_features_pipelines = []
    processed_ordinal_cols = [] # Keep track of ordinal columns handled

    for col_name, order in ordinal_col_definitions.items():
        if col_name in X.columns and order:
            # Ensure the column doesn't have values not in the defined order before OrdinalEncoder if strict
            # For robustness with OrdinalEncoder, handle_unknown='use_encoded_value' is used.
            ordinal_pipeline = Pipeline([
                ('ordinal', OrdinalEncoder(categories=[order], handle_unknown='use_encoded_value', unknown_value=-1)) # -1 for unknown
            ])
            ordinal_features_pipelines.append((f'ordinal_{col_name}', ordinal_pipeline, [col_name]))
            processed_ordinal_cols.append(col_name)
            logger.info(f"Ordinal encoding pipeline defined for: '{col_name}' with order: {order}")
        elif col_name in X.columns and not order:
            logger.warning(f"Order not defined for ordinal column '{col_name}' in config. It will not be ordinally encoded by this logic.")


    # Boolean features (already 0/1 or True/False from preprocessing)
    bool_cols_config_keys = [
        config.get('campaign_calls_processing', {}).get('negative_adjustment_indicator_col'),
        config.get('previous_contact_days_processing', {}).get('previously_contacted_indicator_col')
    ]
    boolean_cols = [col for col in bool_cols_config_keys if col and col in X.columns]

    for col in boolean_cols:
        if col in X.columns: # check
            X[col] = X[col].astype(int)
            logger.debug(f"Boolean column '{col}' ensured as integer type.")

    # Features for log transformation
    log_transform_cols = [col for col in config.get('features_for_log_transform', []) if col in numerical_cols]
    std_scale_only_cols = [col for col in numerical_cols if col not in log_transform_cols]

    transformers_list = []

    if log_transform_cols:
        log_pipeline = Pipeline([
            ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')), # np.log1p handles 0s by calculating log(1+x)
            ('scaler', StandardScaler())
        ])
        transformers_list.append(('log_transform_scale', log_pipeline, log_transform_cols))
        logger.info(f"Log transformation and scaling pipeline defined for: {log_transform_cols}")

    if std_scale_only_cols:
        std_scale_pipeline = Pipeline([('scaler', StandardScaler())])
        transformers_list.append(('std_scale_only', std_scale_pipeline, std_scale_only_cols))
        logger.info(f"Standard scaling pipeline defined for: {std_scale_only_cols}")

    if nominal_cols:
        nominal_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers_list.append(('nominal_onehot', nominal_pipeline, nominal_cols))
        logger.info(f"One-hot encoding pipeline defined for: {nominal_cols}")

    transformers_list.extend(ordinal_features_pipelines) # Add defined ordinal pipelines

    # Explicitly passthrough boolean columns if they are not part of other transformations
    # This ensures they are kept "as is (0/1)", assuming they are not in numerical_cols for scaling.
    all_transformer_input_cols = log_transform_cols + std_scale_only_cols + nominal_cols + processed_ordinal_cols
    passthrough_bool_cols = [b_col for b_col in boolean_cols if b_col not in all_transformer_input_cols]
    if passthrough_bool_cols:
        transformers_list.append(('boolean_passthrough', 'passthrough', passthrough_bool_cols))
        logger.info(f"Boolean features to be passed through: {passthrough_bool_cols}")


    if not transformers_list:
        logger.warning("No transformers were defined for ColumnTransformer. Features might not be processed as expected.")
    
    # Check for duplicate column processing
    temp_col_set = set()
    for name, trans, cols_to_transform in transformers_list:
        if trans == 'passthrough': continue # Passthrough doesn't transform in a way that causes issues
        for c in cols_to_transform:
            if c in temp_col_set:
                logger.warning(f"Column '{c}' is configured for multiple transformations. Check transformer definitions.")
            temp_col_set.add(c)


    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='passthrough',  # Passes through any columns not explicitly handled.
        verbose_feature_names_out=False # Keeps column names cleaner e.g. 'col' instead of 'transformer__col'
    )
    # Setting verbose_feature_names_out=False makes get_feature_names_out() simpler.

    # Split data BEFORE fitting the preprocessor
    split_config = config.get('train_test_split', {})
    test_size = split_config.get('test_size', 0.2)
    stratify_target = split_config.get('stratify_by_target', True)
    random_seed = config.get('random_seed', 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y if stratify_target else None
    )
    logger.info(f"Data split into train and test sets. X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Fit preprocessor on X_train and transform X_train and X_test
    logger.info("Fitting ColumnTransformer on training data (X_train)...")
    preprocessor.fit(X_train)
    
    logger.info("Transforming training data (X_train)...")
    X_train_processed_array = preprocessor.transform(X_train)
    
    logger.info("Transforming test data (X_test)...")
    X_test_processed_array = preprocessor.transform(X_test)
    
    # Get feature names after transformation for DataFrame conversion
    try:
        feature_names_out = preprocessor.get_feature_names_out()
        X_train_processed_df = pd.DataFrame(X_train_processed_array, columns=feature_names_out, index=X_train.index)
        X_test_processed_df = pd.DataFrame(X_test_processed_array, columns=feature_names_out, index=X_test.index)
        logger.info(f"Processed X_train_df shape: {X_train_processed_df.shape}, X_test_df shape: {X_test_processed_df.shape}")
        logger.debug(f"Processed feature names: {feature_names_out.tolist()}")
    except Exception as e:
        logger.error(f"Error getting feature names from ColumnTransformer or creating DataFrames: {e}. Returning arrays in DataFrames without column names.")
        X_train_processed_df = pd.DataFrame(X_train_processed_array, index=X_train.index)
        X_test_processed_df = pd.DataFrame(X_test_processed_array, index=X_test.index)

    logger.info("Feature engineering and data splitting complete.")
    
    return X_train_processed_df, X_test_processed_df, y_train, y_test, preprocessor

def engineer_features_only(
    df: pd.DataFrame,
    config: dict,
    fit_transformers: bool = True,
    preprocessor: ColumnTransformer = None
) -> Union[Tuple[pd.DataFrame, pd.Series, ColumnTransformer], Tuple[pd.DataFrame, pd.Series]]:
    """
    Performs feature engineering (encoding, scaling) without data splitting.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        config (dict): The pipeline configuration dictionary.
        fit_transformers (bool): Whether to fit transformers (True for train, False for test).
        preprocessor (ColumnTransformer, optional): Pre-fitted preprocessor for test data.

    Returns:
        If fit_transformers=True: Tuple[pd.DataFrame, pd.Series, ColumnTransformer]
            X_processed_df: Processed features.
            y: Target variable.
            preprocessor: The fitted scikit-learn ColumnTransformer object.
        If fit_transformers=False: Tuple[pd.DataFrame, pd.Series]
            X_processed_df: Processed features.
            y: Target variable.
    """
    logger.info(f"Starting feature engineering {'(fitting transformers)' if fit_transformers else '(applying pre-fitted transformers)'}...")

    target_col = config.get('target_column')
    if not target_col or target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in DataFrame or not specified in config.")
        raise ValueError(f"Target column '{target_col}' not found or not specified.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"Separated features (X shape: {X.shape}) and target (y shape: {y.shape}).")

    if fit_transformers:
        # Build preprocessor for training data
        # Identify feature lists from config (ensure they exist in X's columns)
        numerical_cols = [col for col in config.get('numerical_features_for_scaling', []) if col in X.columns]
        nominal_cols = [col for col in config.get('nominal_features_for_onehot', []) if col in X.columns]

        # Ordinal features (example: education_level)
        ordinal_col_definitions = {
            'education_level': config.get('education_level_order', [])
        }
        ordinal_features_pipelines = []
        processed_ordinal_cols = []

        for col_name, order in ordinal_col_definitions.items():
            if col_name in X.columns and order:
                ordinal_pipeline = Pipeline([
                    ('ordinal', OrdinalEncoder(categories=[order], handle_unknown='use_encoded_value', unknown_value=-1))
                ])
                ordinal_features_pipelines.append((f'ordinal_{col_name}', ordinal_pipeline, [col_name]))
                processed_ordinal_cols.append(col_name)
                logger.info(f"Ordinal encoding pipeline defined for: '{col_name}' with order: {order}")
            elif col_name in X.columns and not order:
                logger.warning(f"Order not defined for ordinal column '{col_name}' in config. It will not be ordinally encoded by this logic.")

        # Boolean features (already 0/1 or True/False from preprocessing)
        bool_cols_config_keys = [
            config.get('campaign_calls_processing', {}).get('negative_adjustment_indicator_col'),
            config.get('previous_contact_days_processing', {}).get('previously_contacted_indicator_col')
        ]
        boolean_cols = [col for col in bool_cols_config_keys if col and col in X.columns]

        for col in boolean_cols:
            if col in X.columns:
                X[col] = X[col].astype(int)
                logger.debug(f"Boolean column '{col}' ensured as integer type.")

        # Features for log transformation
        log_transform_cols = [col for col in config.get('features_for_log_transform', []) if col in numerical_cols]
        std_scale_only_cols = [col for col in numerical_cols if col not in log_transform_cols]

        transformers_list = []

        if log_transform_cols:
            log_pipeline = Pipeline([
                ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                ('scaler', StandardScaler())
            ])
            transformers_list.append(('log_transform_scale', log_pipeline, log_transform_cols))
            logger.info(f"Log transformation and scaling pipeline defined for: {log_transform_cols}")

        if std_scale_only_cols:
            std_scale_pipeline = Pipeline([('scaler', StandardScaler())])
            transformers_list.append(('std_scale_only', std_scale_pipeline, std_scale_only_cols))
            logger.info(f"Standard scaling pipeline defined for: {std_scale_only_cols}")

        if nominal_cols:
            nominal_pipeline = Pipeline([
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers_list.append(('nominal_onehot', nominal_pipeline, nominal_cols))
            logger.info(f"One-hot encoding pipeline defined for: {nominal_cols}")

        transformers_list.extend(ordinal_features_pipelines)

        # Explicitly passthrough boolean columns if they are not part of other transformations
        all_transformer_input_cols = log_transform_cols + std_scale_only_cols + nominal_cols + processed_ordinal_cols
        passthrough_bool_cols = [b_col for b_col in boolean_cols if b_col not in all_transformer_input_cols]
        if passthrough_bool_cols:
            transformers_list.append(('boolean_passthrough', 'passthrough', passthrough_bool_cols))
            logger.info(f"Boolean features to be passed through: {passthrough_bool_cols}")

        if not transformers_list:
            logger.warning("No transformers were defined for ColumnTransformer. Features might not be processed as expected.")

        preprocessor = ColumnTransformer(
            transformers=transformers_list,
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        # Fit preprocessor on training data
        logger.info("Fitting ColumnTransformer on training data...")
        preprocessor.fit(X)
        
        logger.info("Transforming training data...")
        X_processed_array = preprocessor.transform(X)
        
        # Get feature names after transformation for DataFrame conversion
        try:
            feature_names_out = preprocessor.get_feature_names_out()
            X_processed_df = pd.DataFrame(X_processed_array, columns=feature_names_out, index=X.index)
            logger.info(f"Processed X_df shape: {X_processed_df.shape}")
            logger.debug(f"Processed feature names: {feature_names_out.tolist()}")
        except Exception as e:
            logger.error(f"Error getting feature names from ColumnTransformer: {e}. Creating DataFrame without column names.")
            X_processed_df = pd.DataFrame(X_processed_array, index=X.index)

        logger.info("Feature engineering (with fitting) complete.")
        return X_processed_df, y, preprocessor
    
    else:
        # Apply pre-fitted preprocessor to test data
        if preprocessor is None:
            raise ValueError("preprocessor must be provided when fit_transformers=False")
        
        # Process boolean columns the same way as training
        bool_cols_config_keys = [
            config.get('campaign_calls_processing', {}).get('negative_adjustment_indicator_col'),
            config.get('previous_contact_days_processing', {}).get('previously_contacted_indicator_col')
        ]
        boolean_cols = [col for col in bool_cols_config_keys if col and col in X.columns]

        for col in boolean_cols:
            if col in X.columns:
                X[col] = X[col].astype(int)
                logger.debug(f"Boolean column '{col}' ensured as integer type.")
        
        logger.info("Transforming test data with pre-fitted preprocessor...")
        X_processed_array = preprocessor.transform(X)
        
        try:
            feature_names_out = preprocessor.get_feature_names_out()
            X_processed_df = pd.DataFrame(X_processed_array, columns=feature_names_out, index=X.index)
            logger.info(f"Processed X_df shape: {X_processed_df.shape}")
        except Exception as e:
            logger.error(f"Error getting feature names from ColumnTransformer: {e}. Creating DataFrame without column names.")
            X_processed_df = pd.DataFrame(X_processed_array, index=X.index)

        logger.info("Feature engineering (applying pre-fitted) complete.")
        return X_processed_df, y
