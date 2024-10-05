import numpy as np
import pandas as pd


class Datapreprocessor:
    """A class with some static methods for general data preprocessing tasks specific to our dataset"""

    CONTINUOUS_REALISTIC_LIMITS: dict[str, tuple[int, int]] = {
        'age': (0, 125),
        'resting blood pressure': (0, 250),
        'chol': (0, 600),
        'max_heart_rate': (50, 250),
        'oldpeak': (0, 10),
    }

    @staticmethod
    def drop_missing_targets(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where we don't have a target
         Parameters:
                    df (pd.DataFrame): Input data
            Returns:
                    df (pd.DataFrame): df with missing target rows removed
        """

        num_missing_targets = df['target'].isna().sum()
        if num_missing_targets > 0:
            print(f"Warning: missing {num_missing_targets} target values")
        return df.dropna(subset=['target'])

    @staticmethod
    def clean_oldpeak(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace -99.99 values in oldpeak feature
         Parameters:
                    df (pd.DataFrame): Input data
            Returns:
                    df (pd.DataFrame): df with oldpeak processed
        """

        df['oldpeak'] = df['oldpeak'].replace(-99.99, np.nan)
        return df

    @staticmethod
    def sanity_check_values(df):
        """
        Check values in the dataframe are realistic given known limits, warn if they are not
          Parameters:
                    df (pd.DataFrame): Input data
        """
        for feature_name, (min_val, max_val) in Datapreprocessor.CONTINUOUS_REALISTIC_LIMITS.items():
            if feature_name in df.columns:
                out_of_bounds = (df[feature_name] < min_val) | (df[feature_name] > max_val)
                if out_of_bounds.any():
                    # Get the indicies of unrealistic data
                    row_indices = df[out_of_bounds].index.tolist()
                    print(
                        f"Warning: Feature '{feature_name}' contains values outside of realstic range {min_val}-{max_val}.")
                    print(f"Row numbers with unlikely values: {row_indices}")

    @staticmethod
    def process(df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all preprocessing steps
         Parameters:
                    df (pd.DataFrame): Input data
            Returns:
                    df (pd.DataFrame): Preprocessed data
        """
        df: pd.DataFrame = Datapreprocessor.drop_missing_targets(df)
        df: pd.DataFrame = Datapreprocessor.clean_oldpeak(df)
        Datapreprocessor.sanity_check_values(df)
        return df
