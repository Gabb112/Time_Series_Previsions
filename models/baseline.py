import pandas as pd
from models.model_trainer import ModelTrainer


class PersistenceModel(ModelTrainer):
    """Persistence (Naive) model for forecasting time series."""

    def __init__(self):
        super().__init__()

    def train(self, train_data, val_data=None, target_column=None):
        """Persistence model doesn't actually 'train'. It simply learns the last value."""
        print("Persistence model doesn't train, using last known values.")
        return self

    def predict(self, test_data, target_column="Adj Close"):
        """Makes predictions based on the previous value."""
        if test_data is None or len(test_data) == 0:
            print("Warning: Test data is empty, cannot make predictions")
            return None

        predictions = test_data[target_column].shift(1)
        # The first value is not known, hence fillna using the first value
        predictions = predictions.fillna(test_data[target_column].iloc[0])
        return predictions
