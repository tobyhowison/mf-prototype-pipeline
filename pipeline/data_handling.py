from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_params import DATA_PATH


class HeartDataLoader:
    """
    Purpose of this class is to provide generalised way of loading heard data...
    e.g. if it is a different format in the future
    """
    @staticmethod
    def from_csv(csv_path: str) -> pd.DataFrame:
        """Load from .csv into DataFrame"""
        return pd.read_csv(csv_path)


class HeartDataPreprocessor:

    CATEGORICAL_FEATURES: list[str] = [
        'age',
        'sex',
        'chest pain type',
        'fasting blood sugar',
        'resting ECG',
        'exang',
        'slope',
        'number vessels flourosopy',
        'thal'
    ]
    CONTINUOUS_FEATURES: list[str] = [
        'resting blood pressure',
        'chol',
        'max heart rate',
        'oldpeak']
    CATEGORICAL_IMPUTATION_STRATEGY: str = 'most_frequent'
    CONTINUOUS_IMPUTATION_STRATEGY: str = 'mean'
    CONTINUOUS_LIMITS: dict[str, tuple[int, int]] = {
        'age': (0, 125),
        'resting blood pressure': (0, 250),
        'chol': (0, 600),
        'max_heart_rate': (50, 250),
        'oldpeak': (0, 10),
    }

    def __init__(self):
        self.preprocessor = self._create_preprocessor()

    def _create_preprocessor(self) -> ColumnTransformer:
        """Create preprocessors for imputation of different feature types"""

        # Pipeline for continuous features (imputation + scaling)
        continuous_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.CONTINUOUS_IMPUTATION_STRATEGY)),
            # ('scaler', MinMaxScaler())  # Apply MinMaxScaler to continuous features
        ])

        # Imputer for categorical features (no scaling)
        categorical_imputer = SimpleImputer(strategy=self.CATEGORICAL_IMPUTATION_STRATEGY)

        # Return single column transformer
        return ColumnTransformer(
            transformers=[
                ('continuous', continuous_pipeline, self.CONTINUOUS_FEATURES),
                ('categorical', categorical_imputer, self.CATEGORICAL_FEATURES)
            ])

    def _drop_missing_targets(self, df: pd.DataFrame):
        """Drop rows where we don't have a target"""
        num_missing_targets = df['target'].isna().sum()
        if num_missing_targets > 0:
            print(f"Warning: missing {num_missing_targets} target values")
        return df.dropna(subset=['target'])

    def _sanity_check_values(self, df):
        """Check values in the dataframe are realistic given known limits"""
        for feature_name, (min_val, max_val) in self.CONTINUOUS_LIMITS.items():
            if feature_name in df.columns:
                out_of_bounds = (df[feature_name] < min_val) | (df[feature_name] > max_val)
                if out_of_bounds.any():
                    # Get the indicies of unrealistic data
                    row_indices = df[out_of_bounds].index.tolist()
                    print(f"Warning: Feature '{feature_name}' contains values outside of realstic range {min_val}-{max_val}.")
                    print(f"Row numbers with unlikely values: {row_indices}")


    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Preprocess raw data"""

        # Remove missing target data, if there is any
        df = self._drop_missing_targets(df)

        # Split features and target
        Y: pd.Series = df['target']
        X: pd.DataFrame = df.drop(columns=['target'])

        # Explicitly remove 99.0 values in 'oldpeak' feature
        X['oldpeak'] = X['oldpeak'].replace(-99.99, np.nan)

        # Warn user if any values seem unlikely
        self._sanity_check_values(X)

        # Impute missing values in X
        X: pd.DataFrame = pd.DataFrame(self.preprocessor.fit_transform(X), columns=X.columns)

        return X, Y


if __name__ == "__main__":

    # Get relative path to data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_data_path: str = os.path.join(script_dir, '..', DATA_PATH)

    # Load data
    heart_data: pd.DataFrame = HeartDataLoader.from_csv(relative_data_path)

    # Create data pre-processor
    heart_data_loader: HeartDataPreprocessor = HeartDataPreprocessor()

    # Load
    X, y = heart_data_loader.preprocess(heart_data)

