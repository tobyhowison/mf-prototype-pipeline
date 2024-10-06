import sys
import os
from sklearn.compose import ColumnTransformer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.imputation_pipeline import ImputationPipeline

def test_imputation_pipeline_create():
    column_transformer = ImputationPipeline.create()

    assert isinstance(column_transformer, ColumnTransformer)
    assert len(column_transformer.transformers) == 2  # Ensure continuous and categorical transformers are included