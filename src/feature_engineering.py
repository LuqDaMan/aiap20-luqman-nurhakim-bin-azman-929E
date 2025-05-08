# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Tuple, Optional, Dict, Union

import logging
logger = logging.getLogger(__name__)

class InteractionFeatures(BaseEstimator, TransformerMixin):
    """
    Creates interaction features by multiplying specified pairs of numerical columns.
    """
    def __init__(self, interaction_pairs: List[Tuple[str, str]], include_original: bool = True):
        """
        Args:
            interaction_pairs (List[Tuple[str, str]]): A list of tuples, where each
                tuple contains two column names to interact.
            include_original (bool): Whether to include original columns in the output.
                                     If False, only new interaction features are returned
                                     (requires careful use in ColumnTransformer).
                                     Defaults to True (appends new features).
        """
        if not isinstance(interaction_pairs, list) or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in interaction_pairs):
            raise ValueError("interaction_pairs must be a list of string tuples, each with two elements.")
        self.interaction_pairs = interaction_pairs
        self.include_original = include_original
        self._feature_names_out: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Validates that the specified columns exist in the DataFrame.
        Args:
            X (pd.DataFrame): The input DataFrame.
            y (Optional[pd.Series]): Ignored.
        Returns:
            self: The fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        
        self.original_columns_ = X.columns.tolist()
        missing_cols = []
        for col1, col2 in self.interaction_pairs:
            if col1 not in X.columns:
                missing_cols.append(col1)
            if col2 not in X.columns:
                missing_cols.append(col2)
        
        if missing_cols:
            raise ValueError(f"The following columns are not in the DataFrame: {list(set(missing_cols))}")
        
        # Check if interacting columns are numeric
        for col1, col2 in self.interaction_pairs:
            if not pd.api.types.is_numeric_dtype(X[col1]):
                raise TypeError(f"Column '{col1}' for interaction is not numeric.")
            if not pd.api.types.is_numeric_dtype(X[col2]):
                raise TypeError(f"Column '{col2}' for interaction is not numeric.")

        # Determine output feature names
        new_feature_names = []
        for col1, col2 in self.interaction_pairs:
            new_feature_names.append(f"{col1}_x_{col2}")
        
        if self.include_original:
            self._feature_names_out = self.original_columns_ + new_feature_names
        else:
            self._feature_names_out = new_feature_names

        logger.info(f"InteractionFeatures fitted for pairs: {self.interaction_pairs}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds interaction features to the DataFrame.
        Args:
            X (pd.DataFrame): The input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with added interaction features.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        
        X_transformed = X.copy() if self.include_original else pd.DataFrame(index=X.index)

        for col1, col2 in self.interaction_pairs:
            if col1 not in X.columns or col2 not in X.columns:
                raise ValueError(f"Columns {col1} or {col2} not in input DataFrame for transform.")
            interaction_col_name = f"{col1}_x_{col2}"
            X_transformed[interaction_col_name] = X[col1] * X[col2]
        
        logger.debug(f"Interaction features created for pairs: {self.interaction_pairs}")
        
        if not self.include_original: # Return only new features
             return X_transformed[[f"{c1}_x_{c2}" for c1,c2 in self.interaction_pairs]]
        return X_transformed

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if self._feature_names_out is None:
            # This might happen if fit was not called, or if include_original was false and original cols were not passed
            if input_features and not self.include_original:
                return [f"{c1}_x_{c2}" for c1,c2 in self.interaction_pairs]
            raise NotFittedError("This InteractionFeatures instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        return self._feature_names_out


class PolynomialFeaturesWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for sklearn's PolynomialFeatures, allowing application to specified
    numerical columns and returning a DataFrame.
    """
    def __init__(self, variables: List[str], degree: int = 2, interaction_only: bool = False, include_bias: bool = False):
        """
        Args:
            variables (List[str]): List of numerical column names to generate polynomial features from.
            degree (int): The degree of the polynomial features. Defaults to 2.
            interaction_only (bool): If True, only interaction features are produced. Defaults to False.
            include_bias (bool): If True, include a bias column (feature of ones). Defaults to False.
        """
        if not isinstance(variables, list) or not variables:
            raise ValueError("Variables must be a non-empty list of column names.")
        self.variables = variables
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.poly_transformer_: Optional[PolynomialFeatures] = None
        self.output_feature_names_: Optional[List[str]] = None
        self.original_columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        
        self.original_columns_ = X.columns.tolist()
        missing_cols = [col for col in self.variables if col not in X.columns]
        if missing_cols:
            raise ValueError(f"The following variables are not in the DataFrame: {missing_cols}")

        for var in self.variables:
            if not pd.api.types.is_numeric_dtype(X[var]):
                raise TypeError(f"Column '{var}' for PolynomialFeatures is not numeric.")

        self.poly_transformer_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        self.poly_transformer_.fit(X[self.variables])
        
        # Get output feature names
        try:
            self.output_feature_names_ = self.poly_transformer_.get_feature_names_out(self.variables).tolist()
        except Exception as e:
            logger.warning(f"Could not get feature names from PolynomialFeatures. Error: {e}. Defaulting to generic names.")
            # Fallback (less informative names)
            num_output_features = self.poly_transformer_.n_output_features_
            self.output_feature_names_ = [f"poly_{i}" for i in range(num_output_features)]

        logger.info(f"PolynomialFeaturesWrapper fitted for variables: {self.variables} with degree {self.degree}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if self.poly_transformer_ is None or self.output_feature_names_ is None or self.original_columns_ is None:
            raise NotFittedError("PolynomialFeaturesWrapper is not fitted yet.")

        X_poly_transformed = self.poly_transformer_.transform(X[self.variables])
        poly_df = pd.DataFrame(X_poly_transformed, columns=self.output_feature_names_, index=X.index)

        # Drop original columns that were used for polynomial transformation to avoid duplication
        # as PolynomialFeatures includes the original features (degree 1 terms) by default unless interaction_only=True and degree=1
        # or if include_bias is false and degree is 1, it only returns originals.
        # A common pattern is to replace original columns with their polynomial expansions.
        X_remaining = X.drop(columns=[col for col in self.variables if col in X.columns])
        
        # Concatenate the polynomial features with the remaining original features
        X_transformed = pd.concat([X_remaining, poly_df], axis=1)
        
        logger.debug(f"Polynomial features created for variables: {self.variables}")
        return X_transformed

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if self.output_feature_names_ is None or self.original_columns_ is None:
            raise NotFittedError("This PolynomialFeaturesWrapper instance is not fitted yet.")
        
        # Features that were not part of the polynomial transformation
        remaining_original_features = [col for col in self.original_columns_ if col not in self.variables]
        return remaining_original_features + self.output_feature_names_


class AgeBinner(BaseEstimator, TransformerMixin):
    """
    Converts a numerical 'age' column into categorical bins.
    The new column will replace the original 'age' column.
    """
    def __init__(self, age_variable: str = 'age', bins: Optional[List[Union[int, float]]] = None,
                 labels: Optional[List[str]] = None, new_col_name: Optional[str] = None):
        """
        Args:
            age_variable (str): The name of the age column. Defaults to 'age'.
            bins (Optional[List[Union[int, float]]]): List of bin edges.
                Example: [0, 30, 40, 50, 60, 100]. If None, uses default bins.
            labels (Optional[List[str]]): Labels for the bins. Must be len(bins) - 1.
                If None, default labels are generated.
            new_col_name (Optional[str]): Name for the new binned column.
                                          If None, defaults to '{age_variable}_binned'.
        """
        self.age_variable = age_variable
        self.bins = bins if bins is not None else [0, 18, 30, 40, 50, 60, 120] # Example default bins
        self.labels = labels
        self.new_col_name = new_col_name if new_col_name is not None else f"{self.age_variable}_binned"

        if self.labels and len(self.labels) != len(self.bins) - 1:
            raise ValueError("Number of labels must be one less than the number of bin edges.")
        if not self.labels:
            self.labels = [f"bin_{i+1}" for i in range(len(self.bins)-1)]


    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if self.age_variable not in X.columns:
            raise ValueError(f"Age variable '{self.age_variable}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(X[self.age_variable]):
            raise TypeError(f"Age variable '{self.age_variable}' must be numeric.")
        logger.info(f"AgeBinner fitted for variable '{self.age_variable}' with bins {self.bins}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if self.age_variable not in X.columns:
            raise ValueError(f"Age variable '{self.age_variable}' not found in DataFrame during transform.")

        X_transformed = X.copy()
        X_transformed[self.new_col_name] = pd.cut(
            X_transformed[self.age_variable],
            bins=self.bins,
            labels=self.labels,
            right=False, # [0, 18)
            include_lowest=True
        )
        # Optionally drop the original age column if the new binned column is preferred
        # X_transformed = X_transformed.drop(columns=[self.age_variable])
        logger.debug(f"Age variable '{self.age_variable}' binned into '{self.new_col_name}'. Original column kept by default.")
        return X_transformed

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out if original columns are not stored during fit.")
        
        output_features = input_features[:]
        if self.new_col_name not in output_features:
            output_features.append(self.new_col_name)
        # If original age_variable is meant to be dropped, remove it here
        # if self.age_variable in output_features and self.age_variable != self.new_col_name:
        #     output_features.remove(self.age_variable)
        return output_features


class PdaysTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms the 'pdays' column (days since last contact).
    - Creates 'was_contacted_previously' (binary).
    - Converts 'pdays' for contacted clients (values other than 999 or -1)
      into a 'days_since_last_contact' column, possibly scaling or capping it.
    Original 'pdays' column is typically dropped.
    """
    def __init__(self, pdays_variable: str = 'pdays', special_value: int = 999, create_days_since_col: bool = True):
        """
        Args:
            pdays_variable (str): Name of the pdays column. Defaults to 'pdays'.
            special_value (int): Value in pdays indicating client was not previously contacted
                                 (commonly 999 in UCI Bank Marketing dataset, sometimes -1).
                                 Defaults to 999.
            create_days_since_col (bool): Whether to create the 'days_since_last_contact' column.
                                          Defaults to True.
        """
        self.pdays_variable = pdays_variable
        self.special_value = special_value
        self.create_days_since_col = create_days_since_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if self.pdays_variable not in X.columns:
            raise ValueError(f"Pdays variable '{self.pdays_variable}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(X[self.pdays_variable]):
            raise TypeError(f"Pdays variable '{self.pdays_variable}' must be numeric.")
        logger.info(f"PdaysTransformer fitted for variable '{self.pdays_variable}' with special value {self.special_value}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if self.pdays_variable not in X.columns:
            raise ValueError(f"Pdays variable '{self.pdays_variable}' not found in DataFrame during transform.")

        X_transformed = X.copy()
        
        # Feature 1: Was contacted previously
        X_transformed['was_contacted_previously'] = (X_transformed[self.pdays_variable] != self.special_value).astype(int)

        if self.create_days_since_col:
            # Feature 2: Days since last contact (actual days, or 0 if not contacted/special_value)
            # Using np.where to handle this: if pdays is special_value, then 0 (or another placeholder), else pdays.
            # Or, more directly, only copy over pdays values where not special_value, and fill others with 0 or NaN.
            days_col_name = 'days_since_last_contact'
            X_transformed[days_col_name] = X_transformed[self.pdays_variable].copy()
            X_transformed.loc[X_transformed[self.pdays_variable] == self.special_value, days_col_name] = 0 # Or np.nan if imputation is desired later
        
        # Optionally drop original pdays column
        # X_transformed = X_transformed.drop(columns=[self.pdays_variable])
        logger.debug(f"Pdays variable '{self.pdays_variable}' transformed. Original column kept by default.")
        return X_transformed

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if input_features is None:
            raise ValueError("input_features must be provided.")
        
        output_features = input_features[:]
        if 'was_contacted_previously' not in output_features:
            output_features.append('was_contacted_previously')
        if self.create_days_since_col and 'days_since_last_contact' not in output_features:
            output_features.append('days_since_last_contact')
        
        # If original pdays_variable is meant to be dropped, remove it here
        # if self.pdays_variable in output_features:
        #    output_features.remove(self.pdays_variable)
        return output_features


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Starting feature_engineering transformers example...")

    data = {
        'age': [25, 45, 30, 50, 35, 60],
        'balance': [1000, 5000, 200, 10000, 1500, 3000],
        'duration': [120, 300, 90, 600, 200, 150],
        'campaign': [1, 2, 1, 3, 2, 1],
        'pdays': [999, 10, 20, 999, 5, 30], # 999 means not contacted
        'job': ['admin', 'manager', 'student', 'retired', 'tech', 'services'] # For passthrough
    }
    sample_df = pd.DataFrame(data)
    logger.info(f"Original DataFrame:\n{sample_df}")

    # --- Test InteractionFeatures ---
    interaction_transformer = InteractionFeatures(interaction_pairs=[('age', 'balance'), ('duration', 'campaign')])
    interaction_transformer.fit(sample_df)
    df_interacted = interaction_transformer.transform(sample_df.copy())
    logger.info(f"\n--- InteractionFeatures output ---:\n{df_interacted.head()}")
    logger.info(f"Feature names out: {interaction_transformer.get_feature_names_out(sample_df.columns.tolist())}")


    # --- Test PolynomialFeaturesWrapper ---
    # Using df_interacted to show chaining, though typically you'd apply to base numeric features
    poly_transformer = PolynomialFeaturesWrapper(variables=['age', 'balance'], degree=2, include_bias=False)
    poly_transformer.fit(sample_df) # Fit on original sample for simplicity here
    df_poly = poly_transformer.transform(sample_df.copy())
    logger.info(f"\n--- PolynomialFeaturesWrapper output for ['age', 'balance'] (degree 2) ---:\n{df_poly.head()}")
    logger.info(f"Feature names out: {poly_transformer.get_feature_names_out(sample_df.columns.tolist())}")


    # --- Test AgeBinner ---
    age_binner = AgeBinner(age_variable='age', bins=[0, 30, 40, 50, 120], labels=['<30', '30-39', '40-49', '50+'])
    age_binner.fit(sample_df)
    df_age_binned = age_binner.transform(sample_df.copy())
    logger.info(f"\n--- AgeBinner output ---:\n{df_age_binned[['age', 'age_binned']].head()}")
    logger.info(f"Feature names out: {age_binner.get_feature_names_out(sample_df.columns.tolist())}")


    # --- Test PdaysTransformer ---
    pdays_transformer = PdaysTransformer(pdays_variable='pdays', special_value=999)
    pdays_transformer.fit(sample_df)
    df_pdays_transformed = pdays_transformer.transform(sample_df.copy())
    logger.info(f"\n--- PdaysTransformer output ---:\n{df_pdays_transformed[['pdays', 'was_contacted_previously', 'days_since_last_contact']].head()}")
    logger.info(f"Feature names out: {pdays_transformer.get_feature_names_out(sample_df.columns.tolist())}")

    logger.info("\nFeature_engineering transformers example finished.")