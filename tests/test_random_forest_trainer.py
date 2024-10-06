import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.random_forest_trainer import RandomForestTrainer

X = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
y = pd.Series([0, 1])
X_double = pd.DataFrame({'feature1': [1, 2, 1, 2], 'feature2': [3, 4, 3, 4]})
y_double = pd.Series([0, 1, 0, 1])

def test_add_data():
    trainer = RandomForestTrainer()
    trainer.add_data(X, y)
    assert trainer.X.equals(X)
    assert trainer.y.equals(y)


def test_remove_last():
    trainer = RandomForestTrainer()
    trainer.add_data(X, y)
    trainer.add_data(X, y)
    assert trainer.X.equals(X_double)
    assert trainer.y.equals(y_double)
    trainer.remove_last()
    assert trainer.X.equals(X)
    assert trainer.y.equals(y)
