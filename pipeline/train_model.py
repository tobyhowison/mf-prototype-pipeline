import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_params import DATA_PATH, MODEL_SAVE_PATH
from pipeline.data_preprocessor import Datapreprocessor
from pipeline.random_forest_trainer import RandomForestTrainer
from pipeline.data_loader import DataLoader

if __name__ == "__main__":

    # Get correct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_data_path: str = os.path.join(script_dir, '..', DATA_PATH)
    relative_model_save_path: str = os.path.join(script_dir, '..', MODEL_SAVE_PATH)

    # Load data
    heart_data: pd.DataFrame = DataLoader.from_file(relative_data_path)

    # Preprocess
    heart_data = Datapreprocessor.process(heart_data)

    # Split features and target
    X: pd.DataFrame = heart_data.drop(columns=['target'])
    y: pd.Series = heart_data['target']

    # Create trainer and train
    random_forest_trainer: RandomForestTrainer = RandomForestTrainer(X=X, y=y)
    random_forest_trainer.train()

    # Save model
    random_forest_trainer.save_model(relative_model_save_path)
