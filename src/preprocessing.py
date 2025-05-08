# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from typing import List, Union, Optional, Dict # Union was used in thought process, changed to Optional where more appropriate

import logging
logger = logging.getLogger(__name__)

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects specified columns from a DataFrame."""
    def __init__(self, columns: List[str]):
        """
        Args:
            columns (List[str]): A list of column names to select.
        """
        if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
            raise ValueError("Columns must be a list of strings.")
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Checks if the columns exist in the DataFrame.
        Args:
            X (pd.DataFrame): The input DataFrame.
            y (Optional[pd.Series]): Ignored.
        Returns:
            self: The fitted transformer.
        """
        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame by selecting the specified columns.
        Args:
            X (pd.DataFrame): The input DataFrame.
        Returns:
            pd.DataFrame: A DataFrame with only the selected columns.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        try:
            return X[self.columns].copy() # Use .copy() to avoid SettingWithCopyWarning on downstream tasks
        except KeyError as e:
            raise ValueError(f"Error selecting columns. Make sure all columns {self.columns} exist in the input DataFrame. Original error: {e}")


class NumericalImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in numerical columns using mean, median, or a constant.
    """
    def __init__(self, strategy: str = 'median', variables: Optional[List[str]] = None, fill_value: Optional[Union[int, float]] = None):
        """
        Args:
            strategy (str): The imputation strategy. One of 'mean', 'median', 'constant'.
                            Defaults to 'median'.
            variables (Optional[List[str]]): List of numerical column names to impute.
                                             If None, will try to apply to all numerical columns.
            fill_value (Optional[Union[int, float]]): Value to use when strategy is 'constant'.
        """
        if strategy not in ['mean', 'median', 'constant']:
            raise ValueError(f"Strategy must be one of 'mean', 'median', 'constant'. Got {strategy}")
        if strategy == 'constant' and fill_value is None:
            raise ValueError("If strategy is 'constant', fill_value must be provided.")

        self.strategy = strategy
        self.variables = variables
        self.fill_value = fill_value
        self.imputer_: Optional[SimpleImputer] = None # Stores the fitted sklearn imputer
        self.fit_params_: Dict = {} # Stores parameters learned during fit (e.g., means or medians)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        if self.variables is None:
            self.variables_ = X.select_dtypes(include=np.number).columns.tolist()
            if not self.variables_:
                logger.warning("No numerical variables found to impute.")
                return self
        else:
            self.variables_ = [var for var in self.variables if var in X.columns]
            if not self.variables_:
                logger.warning(f"None of the specified variables {self.variables} found in the DataFrame.")
                return self
            if not all(pd.api.types.is_numeric_dtype(X[var]) for var in self.variables_ if var in X.columns):
                non_numeric_vars = [var for var in self.variables_ if not pd.api.types.is_numeric_dtype(X[var])]
                raise TypeError(f"All 'variables' for NumericalImputer must be numeric. Non-numeric: {non_numeric_vars}")


        if not self.variables_: # If still no variables after checks
            return self

        self.imputer_ = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        self.imputer_.fit(X[self.variables_])

        # Store learned statistics for potential inspection
        if self.strategy in ['mean', 'median']:
            self.fit_params_ = dict(zip(self.variables_, self.imputer_.statistics_))
        elif self.strategy == 'constant':
             self.fit_params_ = {var: self.fill_value for var in self.variables_}
        logger.info(f"NumericalImputer fitted with strategy '{self.strategy}' for variables: {self.variables_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if self.imputer_ is None or not hasattr(self, 'variables_') or not self.variables_:
            logger.warning("NumericalImputer is not fitted or no variables to transform.")
            return X.copy()

        X_transformed = X.copy()
        # Ensure variables_ exist in X before attempting to transform
        vars_to_transform = [var for var in self.variables_ if var in X_transformed.columns]
        if not vars_to_transform:
            logger.warning(f"None of the fitted variables {self.variables_} are present in the input DataFrame for transform.")
            return X_transformed

        X_transformed[vars_to_transform] = self.imputer_.transform(X_transformed[vars_to_transform])
        logger.debug(f"NumericalImputation applied to variables: {vars_to_transform}")
        return X_transformed


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in categorical columns using most frequent value or a constant.
    """
    def __init__(self, strategy: str = 'most_frequent', variables: Optional[List[str]] = None, fill_value: str = 'Missing'):
        """
        Args:
            strategy (str): The imputation strategy. One of 'most_frequent', 'constant'.
                            Defaults to 'most_frequent'.
            variables (Optional[List[str]]): List of categorical column names to impute.
                                             If None, will apply to all object/category columns.
            fill_value (str): Value to use when strategy is 'constant'. Defaults to 'Missing'.
        """
        if strategy not in ['most_frequent', 'constant']:
            raise ValueError(f"Strategy must be one of 'most_frequent', 'constant'. Got {strategy}")
        self.strategy = strategy
        self.variables = variables
        self.fill_value = fill_value
        self.imputer_: Optional[SimpleImputer] = None
        self.fit_params_: Dict = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        if self.variables is None:
            self.variables_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
            if not self.variables_:
                logger.warning("No categorical variables found to impute.")
                return self
        else:
            self.variables_ = [var for var in self.variables if var in X.columns]
            if not self.variables_:
                logger.warning(f"None of the specified variables {self.variables} found in the DataFrame.")
                return self
            # Consider checking if they are actually categorical/object types
            # for var in self.variables_:
            #     if not pd.api.types.is_object_dtype(X[var]) and not pd.api.types.is_categorical_dtype(X[var]):
            #         raise TypeError(f"Variable '{var}' for CategoricalImputer is not of object or category type.")


        if not self.variables_:
            return self

        self.imputer_ = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        self.imputer_.fit(X[self.variables_])

        if self.strategy == 'most_frequent':
            self.fit_params_ = dict(zip(self.variables_, self.imputer_.statistics_))
        elif self.strategy == 'constant':
             self.fit_params_ = {var: self.fill_value for var in self.variables_}
        logger.info(f"CategoricalImputer fitted with strategy '{self.strategy}' for variables: {self.variables_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if self.imputer_ is None or not hasattr(self, 'variables_') or not self.variables_:
            logger.warning("CategoricalImputer is not fitted or no variables to transform.")
            return X.copy()

        X_transformed = X.copy()
        vars_to_transform = [var for var in self.variables_ if var in X_transformed.columns]
        if not vars_to_transform:
            logger.warning(f"None of the fitted variables {self.variables_} are present in the input DataFrame for transform.")
            return X_transformed

        X_transformed[vars_to_transform] = self.imputer_.transform(X_transformed[vars_to_transform])
        logger.debug(f"CategoricalImputation applied to variables: {vars_to_transform}")
        return X_transformed


class SklearnScalerWrapper(BaseEstimator, TransformerMixin):
    """
    A wrapper for scikit-learn scalers (StandardScaler, MinMaxScaler, RobustScaler).
    Operates on specified numerical variables.
    """
    def __init__(self, scaler_type: str = 'standard', variables: Optional[List[str]] = None):
        """
        Args:
            scaler_type (str): Type of scaler. One of 'standard', 'minmax', 'robust'.
                               Defaults to 'standard'.
            variables (Optional[List[str]]): List of numerical column names to scale.
                                             If None, will apply to all numerical columns.
        """
        if scaler_type not in ['standard', 'minmax', 'robust']:
            raise ValueError("scaler_type must be 'standard', 'minmax', or 'robust'.")
        self.scaler_type = scaler_type
        self.variables = variables
        self.scaler_: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        if self.variables is None:
            self.variables_ = X.select_dtypes(include=np.number).columns.tolist()
            if not self.variables_:
                logger.warning("No numerical variables found to scale.")
                return self
        else:
            self.variables_ = [var for var in self.variables if var in X.columns]
            if not self.variables_:
                logger.warning(f"None of the specified variables {self.variables} found for scaling.")
                return self
            if not all(pd.api.types.is_numeric_dtype(X[var]) for var in self.variables_):
                non_numeric_vars = [var for var in self.variables_ if not pd.api.types.is_numeric_dtype(X[var])]
                raise TypeError(f"All 'variables' for SklearnScalerWrapper must be numeric. Non-numeric: {non_numeric_vars}")


        if not self.variables_:
            return self

        if self.scaler_type == 'standard':
            self.scaler_ = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler_ = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler_ = RobustScaler()

        self.scaler_.fit(X[self.variables_])
        logger.info(f"SklearnScalerWrapper (type: {self.scaler_type}) fitted for variables: {self.variables_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if self.scaler_ is None or not hasattr(self, 'variables_') or not self.variables_:
            logger.warning(f"SklearnScalerWrapper (type: {self.scaler_type}) is not fitted or no variables to transform.")
            return X.copy()

        X_transformed = X.copy()
        vars_to_transform = [var for var in self.variables_ if var in X_transformed.columns]
        if not vars_to_transform:
            logger.warning(f"None of the fitted variables {self.variables_} are present in the input DataFrame for transform.")
            return X_transformed
        
        X_transformed[vars_to_transform] = self.scaler_.transform(X_transformed[vars_to_transform])
        logger.debug(f"Scaling applied to variables: {vars_to_transform}")
        return X_transformed


class SklearnEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    A wrapper for scikit-learn categorical encoders (OneHotEncoder, OrdinalEncoder).
    Operates on specified categorical variables.
    """
    def __init__(self, encoder_type: str = 'onehot', variables: Optional[List[str]] = None,
                 handle_unknown: str = 'ignore', drop_first_if_onehot: bool = False):
        """
        Args:
            encoder_type (str): Type of encoder. One of 'onehot', 'ordinal'. Defaults to 'onehot'.
            variables (Optional[List[str]]): List of categorical column names to encode.
                                             If None, will apply to all object/category columns.
            handle_unknown (str): For OneHotEncoder, how to handle unknown categories.
                                  'error' or 'ignore'. Defaults to 'ignore'.
                                  For OrdinalEncoder, set to 'use_encoded_value' and provide `unknown_value`.
            drop_first_if_onehot (bool): Whether to drop the first category in OneHotEncoder
                                         to avoid multicollinearity. Defaults to False.
        """
        if encoder_type not in ['onehot', 'ordinal']:
            raise ValueError("encoder_type must be 'onehot' or 'ordinal'.")
        self.encoder_type = encoder_type
        self.variables = variables
        self.handle_unknown = handle_unknown
        self.drop_first_if_onehot = drop_first_if_onehot
        self.encoder_: Optional[Union[OneHotEncoder, OrdinalEncoder]] = None
        self.encoded_feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        if self.variables is None:
            self.variables_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
            if not self.variables_:
                logger.warning("No categorical variables found to encode.")
                return self
        else:
            self.variables_ = [var for var in self.variables if var in X.columns]
            if not self.variables_:
                logger.warning(f"None of the specified variables {self.variables} found for encoding.")
                return self
            # for var in self.variables_: # Type check
            #     if not pd.api.types.is_object_dtype(X[var]) and not pd.api.types.is_categorical_dtype(X[var]):
            #         raise TypeError(f"Variable '{var}' for SklearnEncoderWrapper is not of object or category type.")


        if not self.variables_:
            return self

        if self.encoder_type == 'onehot':
            drop_strategy = 'first' if self.drop_first_if_onehot else None
            self.encoder_ = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False, drop=drop_strategy)
        elif self.encoder_type == 'ordinal':
            # For ordinal, handle_unknown requires unknown_value to be set if using 'use_encoded_value'
            if self.handle_unknown == 'use_encoded_value':
                 # Define a conventional unknown value (e.g., -1 or a large number)
                self.encoder_ = OrdinalEncoder(handle_unknown=self.handle_unknown, unknown_value=np.nan) # Will impute later if needed
            else: # default 'error'
                self.encoder_ = OrdinalEncoder()


        self.encoder_.fit(X[self.variables_])

        if self.encoder_type == 'onehot':
            try:
                self.encoded_feature_names_ = self.encoder_.get_feature_names_out(self.variables_).tolist()
            except Exception as e:
                logger.error(f"Could not get feature names from OneHotEncoder. Check sklearn version. Error: {e}")
                # Fallback for older sklearn or simple naming
                new_cols = []
                for i, var_name in enumerate(self.variables_):
                    for cat in self.encoder_.categories_[i]:
                        new_cols.append(f"{var_name}_{cat}")
                # This fallback won't respect drop='first' correctly in naming easily.
                # Modern sklearn's get_feature_names_out is preferred.
                self.encoded_feature_names_ = new_cols


        logger.info(f"SklearnEncoderWrapper (type: {self.encoder_type}) fitted for variables: {self.variables_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if self.encoder_ is None or not hasattr(self, 'variables_') or not self.variables_:
            logger.warning(f"SklearnEncoderWrapper (type: {self.encoder_type}) is not fitted or no variables to transform.")
            return X.copy()

        X_transformed = X.copy()
        vars_to_encode = [var for var in self.variables_ if var in X_transformed.columns]

        if not vars_to_encode:
            logger.warning(f"None of the fitted variables {self.variables_} are present in the input DataFrame for encoding.")
            return X_transformed

        encoded_data = self.encoder_.transform(X_transformed[vars_to_encode])

        if self.encoder_type == 'onehot':
            if not self.encoded_feature_names_: # Fallback if names weren't generated in fit
                 logger.warning("Encoded feature names not available for OneHotEncoder. Original columns will be dropped and replaced by numerically named columns.")
                 # This situation is less ideal; consider raising an error or more robust naming.
                 num_output_features = encoded_data.shape[1]
                 self.encoded_feature_names_ = [f"ohe_feat_{i}" for i in range(num_output_features)]


            encoded_df = pd.DataFrame(encoded_data, columns=self.encoded_feature_names_, index=X_transformed.index)
            X_transformed = X_transformed.drop(columns=vars_to_encode)
            X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
        elif self.encoder_type == 'ordinal':
            X_transformed[vars_to_encode] = encoded_data
            # Handle NaN produced by OrdinalEncoder for unknown values if unknown_value=np.nan was used
            # This would typically be followed by an imputer if NaNs are not desired.
            # For example, if unknown_value=np.nan, you might want to fill these NaNs:
            # for col in vars_to_encode:
            #     if X_transformed[col].isnull().any():
            #         X_transformed[col] = X_transformed[col].fillna(-1) # Or some other placeholder


        logger.debug(f"Encoding applied to variables: {vars_to_encode}")
        return X_transformed


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a logarithmic transformation (log1p) to specified numerical columns
    to handle skewed data.
    """
    def __init__(self, variables: Optional[List[str]] = None):
        """
        Args:
            variables (Optional[List[str]]): List of numerical column names to transform.
                                             If None, will apply to all numerical columns.
        """
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        if self.variables is None:
            self.variables_ = X.select_dtypes(include=np.number).columns.tolist()
        else:
            self.variables_ = [var for var in self.variables if var in X.columns]
            if not self.variables_:
                 logger.warning(f"None of the specified variables {self.variables} found for LogTransformer.")
                 return self
            if not all(pd.api.types.is_numeric_dtype(X[var]) for var in self.variables_):
                non_numeric_vars = [var for var in self.variables_ if not pd.api.types.is_numeric_dtype(X[var])]
                raise TypeError(f"All 'variables' for LogTransformer must be numeric. Non-numeric: {non_numeric_vars}")

        if not self.variables_:
             logger.warning("No variables selected for LogTransformer.")
             return self

        # Check for non-positive values if using np.log, np.log1p handles X=0 by returning 0.
        # For X < 0, np.log1p will produce NaNs or errors.
        # It's assumed data is already non-negative or pre-processed for log transform.
        for var in self.variables_:
            if (X[var] < 0).any():
                logger.warning(f"Variable '{var}' contains negative values. "
                                "LogTransformer (np.log1p) will produce NaNs for values < -1. "
                                "Ensure data is non-negative or appropriately pre-processed.")
        logger.info(f"LogTransformer 'fitted' (initialized) for variables: {self.variables_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if not hasattr(self, 'variables_') or not self.variables_:
            logger.warning("LogTransformer is not fitted or no variables to transform.")
            return X.copy()

        X_transformed = X.copy()
        vars_to_transform = [var for var in self.variables_ if var in X_transformed.columns]
        if not vars_to_transform:
            logger.warning(f"None of the fitted variables {self.variables_} are present in the input DataFrame for LogTransformer transform.")
            return X_transformed

        for var in vars_to_transform:
            if (X_transformed[var] < 0).any(): # Re-check at transform time for safety
                 logger.warning(f"Transforming variable '{var}' which contains negative values with np.log1p. NaNs may result for x < -1.")
            X_transformed[var] = np.log1p(X_transformed[var])
        logger.debug(f"Log transformation (np.log1p) applied to variables: {vars_to_transform}")
        return X_transformed


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drops specified columns from a DataFrame.
    """
    def __init__(self, variables_to_drop: List[str]):
        """
        Args:
            variables_to_drop (List[str]): A list of column names to drop.
        """
        if not isinstance(variables_to_drop, list) or not all(isinstance(col, str) for col in variables_to_drop):
            raise ValueError("variables_to_drop must be a list of strings.")
        self.variables_to_drop = variables_to_drop

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """No fitting required for this transformer."""
        # Can add a check here if all variables_to_drop are in X.columns,
        # though transform will handle it with errors='ignore'.
        self.fitted_variables_to_drop_ = [col for col in self.variables_to_drop if col in X.columns]
        logger.info(f"ColumnDropper 'fitted'. Will attempt to drop: {self.fitted_variables_to_drop_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame by dropping the specified columns.
        Args:
            X (pd.DataFrame): The input DataFrame.
        Returns:
            pd.DataFrame: A DataFrame with the specified columns removed.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        
        # Only drop columns that actually exist to avoid KeyError
        cols_to_drop_present = [col for col in self.variables_to_drop if col in X.columns]
        if not cols_to_drop_present:
            logger.warning(f"None of the specified columns to drop {self.variables_to_drop} exist in the DataFrame.")
            return X.copy()

        X_transformed = X.drop(columns=cols_to_drop_present, errors='ignore')
        logger.debug(f"Columns dropped: {cols_to_drop_present}")
        return X_transformed


if __name__ == '__main__':
    # Example Usage (assumes DataFrame with mixed types and NaNs)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Starting preprocessing transformers example...")

    data = {
        'numeric_col1': [1, 2, np.nan, 4, 5, 1000],
        'numeric_col2': [np.nan, 0.5, 0.1, 0.8, 0.2, 0.9],
        'skewed_col': [1, 10, 100, 1000, 10000, 100000], # For log transform
        'categorical_col1': ['A', 'B', 'A', np.nan, 'C', 'B'],
        'categorical_col2': ['X', 'Y', 'Y', 'X', np.nan, 'Z'],
        'to_drop': [1,2,3,4,5,6]
    }
    sample_df = pd.DataFrame(data)
    logger.info(f"Original DataFrame:\n{sample_df}\nMissing values:\n{sample_df.isnull().sum()}")

    # --- Test ColumnSelector ---
    selector = ColumnSelector(columns=['numeric_col1', 'categorical_col1'])
    selector.fit(sample_df)
    selected_df = selector.transform(sample_df.copy())
    logger.info(f"\n--- ColumnSelector output for ['numeric_col1', 'categorical_col1'] ---:\n{selected_df}")
    try:
        ColumnSelector(columns=['non_existent_col']).fit(sample_df)
    except ValueError as e:
        logger.info(f"Caught expected error for ColumnSelector with non-existent column: {e}")


    # --- Test NumericalImputer ---
    num_imputer = NumericalImputer(strategy='median', variables=['numeric_col1', 'numeric_col2'])
    num_imputer.fit(sample_df)
    df_num_imputed = num_imputer.transform(sample_df.copy())
    logger.info(f"\n--- NumericalImputer (median) output for ['numeric_col1', 'numeric_col2'] ---:\n{df_num_imputed}\nFit params: {num_imputer.fit_params_}")

    # --- Test CategoricalImputer ---
    cat_imputer = CategoricalImputer(strategy='constant', fill_value='UNKNOWN', variables=['categorical_col1', 'categorical_col2'])
    cat_imputer.fit(sample_df) # Use df_num_imputed as it might be part of a sequence
    df_cat_imputed = cat_imputer.transform(df_num_imputed.copy())
    logger.info(f"\n--- CategoricalImputer (constant='UNKNOWN') output ---:\n{df_cat_imputed}\nFit params: {cat_imputer.fit_params_}")

    # --- Test LogTransformer ---
    log_transformer = LogTransformer(variables=['skewed_col']) # numeric_col1 also for testing, includes a large value
    log_transformer.fit(df_cat_imputed)
    df_log_transformed = log_transformer.transform(df_cat_imputed.copy())
    logger.info(f"\n--- LogTransformer output for ['skewed_col'] ---:\n{df_log_transformed[['skewed_col']]}")
    logger.info(f"Value 100000 becomes: {np.log1p(100000)}")


    # --- Test SklearnScalerWrapper ---
    scaler = SklearnScalerWrapper(scaler_type='standard', variables=['numeric_col1', 'numeric_col2', 'skewed_col'])
    scaler.fit(df_log_transformed) # Use already imputed and transformed data
    df_scaled = scaler.transform(df_log_transformed.copy())
    logger.info(f"\n--- SklearnScalerWrapper (standard) output for ['numeric_col1', 'numeric_col2', 'skewed_col_log'] ---:\n{df_scaled[['numeric_col1', 'numeric_col2', 'skewed_col']]}")


    # --- Test SklearnEncoderWrapper (OneHot) ---
    # For encoder, use df_cat_imputed as it has categories filled
    # and before log transform/scaling on original numeric_col1 for clarity
    ohe_encoder = SklearnEncoderWrapper(encoder_type='onehot', variables=['categorical_col1', 'categorical_col2'], drop_first_if_onehot=True)
    ohe_encoder.fit(df_cat_imputed)
    df_onehot_encoded = ohe_encoder.transform(df_cat_imputed.copy())
    logger.info(f"\n--- SklearnEncoderWrapper (onehot, drop_first=True) output ---:\n{df_onehot_encoded.head()}")
    logger.info(f"Encoded feature names: {ohe_encoder.encoded_feature_names_}")


    # --- Test SklearnEncoderWrapper (Ordinal) ---
    ord_encoder = SklearnEncoderWrapper(encoder_type='ordinal', variables=['categorical_col1', 'categorical_col2'])
    ord_encoder.fit(df_cat_imputed)
    df_ordinal_encoded = ord_encoder.transform(df_cat_imputed.copy()) # Use df_cat_imputed again for a clean comparison
    logger.info(f"\n--- SklearnEncoderWrapper (ordinal) output ---:\n{df_ordinal_encoded[['categorical_col1', 'categorical_col2']]}")

    # --- Test ColumnDropper ---
    col_dropper = ColumnDropper(variables_to_drop=['to_drop', 'non_existent_column'])
    col_dropper.fit(df_onehot_encoded) # Use any df for this
    df_dropped = col_dropper.transform(df_onehot_encoded.copy())
    logger.info(f"\n--- ColumnDropper output (dropping 'to_drop') ---:\n{df_dropped.head()}")
    logger.info(f"Columns after dropping: {df_dropped.columns.tolist()}")

    logger.info("\nPreprocessing transformers example finished.")