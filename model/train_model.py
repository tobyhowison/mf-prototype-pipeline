from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_params import MODEL_SAVE_PATH, DATA_PATH, RANDOM_SEED
from pipeline.imputation_pipeline import ImputationPipeline
from pipeline.data_preprocessor import Datapreprocessor
from pipeline.data_loader import DataLoader


class HeartPredictionTrainer:

    """Class to train a random forest to predict heart disease outcome"""

    TRAIN_TEST_SPLIT: int = 0.2

    def __init__(self, X: pd.DataFrame = None, y: pd.Series = None):

        # Store all added data
        self.X: pd.DataFrame = pd.DataFrame()
        self.y: pd.Series = pd.Series()

        # Generate column transformer to deal with missing data etc
        imputer: ColumnTransformer = ImputationPipeline.create()

        # Create a pipeline including min-max scaler and random forest
        self.pipeline = Pipeline([
            ('impute', imputer),
            ('scale', MinMaxScaler()),
            ('rf', RandomForestClassifier(random_state=RANDOM_SEED))
        ])

        # Keeping track of added data
        self.last_data_size: int = 0
        if X is not None and y is not None:
            self.add_data(X, y)

    def add_data(self, X_new: pd.DataFrame, y_new: pd.Series):
        """
        Add data to be trained on
         Parameters:
                    X (pd.DataFrame): features
                    y (pd.DataFrame): target
        """
        self.X = pd.concat([self.X, X_new], ignore_index=True)
        self.y = pd.concat([self.y, y_new], ignore_index=True)
        self.last_data_size = len(y_new)

    def remove_last(self):
        """
        Remove last added data.
        """
        if self.last_data_size != 0:
            self.X = self.X.iloc[:-self.last_data_size]
            self.y = self.y.iloc[:-self.last_data_size]
        else:
            print(f"Nothing to remove")

    def train(self):
        """
        Train the heart model on loaded data
        """

        if len(self.X) == 0:
            print("Add data to train")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

        # Train
        self.pipeline.fit(X_train, y_train)

        # Evaluate the model (optional, you can use various metrics like accuracy, precision, etc.)
        score = self.pipeline.score(X_test, y_test)
        print(f"Model Accuracy: {score}")

    def save_last_model(self, path: str):
        pickle.dump(self.pipeline, open(path, 'wb'))


if __name__ == "__main__":

    # Get correct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_data_path: str = os.path.join(script_dir, '..', DATA_PATH)
    relative_model_path: str = os.path.join(script_dir, '..', MODEL_SAVE_PATH)

    # Load data
    heart_data: pd.DataFrame = DataLoader.from_file(relative_data_path)

    # Preprocess
    heart_data = Datapreprocessor.process(heart_data)

    # Split features and target
    X: pd.DataFrame = heart_data.drop(columns=['target'])
    y: pd.Series = heart_data['target']

    # Create trainer and train
    random_forest_trainer: HeartPredictionTrainer = HeartPredictionTrainer(X=X, y=y)
    random_forest_trainer.train()

    # Save model
    random_forest_trainer.save_last_model(relative_model_path)


