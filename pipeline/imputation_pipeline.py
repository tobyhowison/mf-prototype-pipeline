from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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


class ImputationPipeline:

    """
    This class generates a ColumnTransformer for dealing with missing values in the data
    """

    @staticmethod
    def create() -> ColumnTransformer:
        """
        Creates a ColumnTransformer that fills in missing values in data
            Returns:
                    column_transformer (ColumnTransformer):
        """

        # Pipeline for continuous features (imputation + scaling)
        continuous_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=CONTINUOUS_IMPUTATION_STRATEGY)),
        ])

        # Imputer for categorical features (no scaling)
        categorical_imputer = SimpleImputer(strategy=CATEGORICAL_IMPUTATION_STRATEGY)

        # Return single column transformer
        return ColumnTransformer(
            transformers=[
                ('continuous', continuous_pipeline, CONTINUOUS_FEATURES),
                ('categorical', categorical_imputer, CATEGORICAL_FEATURES)
            ])
