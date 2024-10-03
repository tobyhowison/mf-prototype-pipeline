from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_params import MODEL_PATH, DATA_PATH
from pipeline.data_preprocess import HeartDataPreprocessor


class HeartModelTrainer:

    TRAIN_TEST_SPLIT: int = 0.2

    def __init__(self, X: pd.DataFrame = None, y: pd.Series = None):
        self.X: pd.DataFrame = pd.DataFrame()
        self.y: pd.Series = pd.Series()
        self.model: RandomForestClassifier() = RandomForestClassifier()
        self.last_data_size: int = 0
        if X is not None and y is not None:
            self.add_data(X, y)

    def add_data(self, X_new: pd.DataFrame, y_new: pd.Series):
        """Add new data for model training."""
        self.X = pd.concat([self.X, X_new], ignore_index=True)
        self.y = pd.concat([self.y, y_new], ignore_index=True)
        self.last_data_size = len(y_new)

    def remove_last(self):
        """Remove last added data."""
        if self.last_data_size != 0:
            self.X = self.X.iloc[:-self.last_data_size]
            self.y = self.y.iloc[:-self.last_data_size]
        else:
            print(f"Nothing to remove")

    def train(self):

        """Train the heart model on loaded data"""

        if len(self.X) == 0:
            print("Add data to train")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train
        self.model.fit(X_train, y_train)

        # Evaluate the model (optional, you can use various metrics like accuracy, precision, etc.)
        score = self.model.score(X_test, y_test)
        print(f"Model Accuracy: {score}")

    def save_last_model(self, path: str):
        joblib.dump(self.model, path)


if __name__ == "__main__":

    # Get correct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_data_path: str = os.path.join(script_dir, '..', DATA_PATH)
    relative_model_path: str = os.path.join(script_dir, '..', MODEL_PATH)

    # Load data
    heart_data_preprocessor: HeartDataPreprocessor = HeartDataPreprocessor()
    X, y = heart_data_preprocessor.load_and_preprocess(relative_data_path)

    # Create trainer and train
    random_forest_trainer: HeartModelTrainer = HeartModelTrainer(X=X, y=y)
    random_forest_trainer.train()

    # Save model
    random_forest_trainer.save_last_model(relative_model_path)
