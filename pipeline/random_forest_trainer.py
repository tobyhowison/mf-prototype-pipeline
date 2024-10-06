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
from config_params import RANDOM_SEED
from pipeline.imputation_pipeline import ImputationPipeline


class RandomForestTrainer:

    """
    A class to train a Random Forest model to predict heart disease outcomes.

    Class allows  user to incrementally add data for training, handle missing values,
    scale features, and train a Random Forest classifier using a pipeline. It supports data
    addition, removal of the last added batch, model training, and saving the trained model.

    Attributes:
    -----------
    X (pd.DataFrame): features
    y (pd.Series): targets
    pipeline (Pipeline): A scikit-learn pipeline for preprocessing and model training.
    last_data_size (int): Tracks the size of the last batch of added data to help removal.
    random_seed (int): a random seed
    """

    TRAIN_TEST_SPLIT: float = 0.2

    def __init__(self, X: pd.DataFrame = None, y: pd.Series = None, random_seed: int = RANDOM_SEED):

        # Store all added data
        self.X: pd.DataFrame = pd.DataFrame()
        self.y: pd.Series = pd.Series()

        # Generate column transformer to deal with missing data values, we add this to the pipeline before training
        imputer: ColumnTransformer = ImputationPipeline.create()

        # Random seed
        self.random_seed = random_seed

        # Create a pipeline including imputation, min-max scaling and random forest
        self.pipeline = Pipeline([
            ('impute', imputer),
            ('scale', MinMaxScaler()),
            ('rf', RandomForestClassifier(random_state=self.random_seed))
        ])

        # Keeping track of added data
        self.last_data_size: int = 0
        if X is not None and y is not None:
            self.add_data(X, y)

    def add_data(self, X_new: pd.DataFrame, y_new: pd.Series):
        """
        Add data to the trainer
         Parameters:
                    X_new (pd.DataFrame): features
                    y_new (pd.DataFrame): target
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
            print(f"Removed last {self.last_data_size} entries.")
            self.last_data_size = 0
        else:
            print(f"Nothing to remove")

    def train(self):
        """
        Train the heart model on loaded data
        """

        if len(self.X) == 0 or len(self.y) == 0:
            print("Add data to train")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_seed)

        # Train and evaluate
        self.pipeline.fit(X_train, y_train)
        self.evaluate_model(X_test, y_test)

    def evaluate_model(self, X_test, y_test):
        """
            Evaluate the model
        """

        score = self.pipeline.score(X_test, y_test)
        print(f"Model Accuracy: {score}")

    def save_model(self, path: str):
        """
           Save the model pipeline.
            Parameters:
                    path (str): Save to here
        """

        pickle.dump(self.pipeline, open(path, 'wb'))





