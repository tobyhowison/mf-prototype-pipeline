from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config_params import CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, CATEGORICAL_IMPUTATION_STRATEGY, \
    CONTINUOUS_IMPUTATION_STRATEGY


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
