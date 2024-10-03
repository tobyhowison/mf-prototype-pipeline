from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_params import MODEL_PATH, DATA_PATH

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

    def __init__(self):
        self.preprocessor = self._create_preprocessor()

    def _create_preprocessor(self) -> ColumnTransformer:
        """Create preprocessors for imputation of differnt feature types"""

        # Pipeline for continuous features (imputation + scaling)
        continuous_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.CONTINUOUS_IMPUTATION_STRATEGY)),
            ('scaler', MinMaxScaler())  # Apply MinMaxScaler to continuous features
        ])

        # Imputer for categorical features (no scaling)
        categorical_imputer = SimpleImputer(strategy=self.CATEGORICAL_IMPUTATION_STRATEGY)

        # Return single column transformer
        return ColumnTransformer(
            transformers=[
                ('continuous', continuous_pipeline, self.CONTINUOUS_FEATURES),
                ('categorical', categorical_imputer, self.CATEGORICAL_FEATURES)
            ])

    def _load(self, csv_path: str) -> pd.DataFrame:
        """Load from .csv into DataFrame"""
        return pd.read_csv(csv_path)

    def _preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Preprocess raw data"""

        # Remove missing target data, if there is any
        df = self._drop_missing_targets(df)

        # Split features and target
        Y: pd.Series = df['target']
        X: pd.DataFrame = df.drop(columns=['target'])

        # Explicitly remove 99.0 values in 'oldpeak' feature
        X['oldpeak'] = X['oldpeak'].replace(-99.0, np.nan)

        # Impute missing values in X
        X: pd.DataFrame = pd.DataFrame(self.preprocessor.fit_transform(X), columns=X.columns)

        return X, Y

    def _drop_missing_targets(self, df: pd.DataFrame):
        """Drop rows where we don't have a target"""
        num_missing_targets = df['target'].isna().sum()
        if num_missing_targets > 0:
            print(f"Warning: missing {num_missing_targets} target values")
        return df.dropna(subset=['target'])

    def load_and_preprocess(self, csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
        """User function to load and preprocess from a csv path"""
        df: pd.DataFrame = self._load(csv_path)
        return self._preprocess(df)


if __name__ == "__main__":

    # Get relative path to data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_data_path: str = os.path.join(script_dir, '..', DATA_PATH)

    # Create data pre-processor
    heart_data_loader: HeartDataPreprocessor = HeartDataPreprocessor()

    # Load
    (X, Y) = heart_data_loader.load_and_preprocess(relative_data_path)

