import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from models.model_trainer import ModelTrainer


class ARIMAModel(ModelTrainer):
    """ARIMA model for forecasting time series."""

    def __init__(self, order=(5, 1, 0)):
        super().__init__()
        self.order = order
        self.model = None
        self.trained = False

    def train(self, train_data, val_data=None, target_column="Adj Close"):
        """Trains the ARIMA model."""
        if train_data is None or target_column not in train_data.columns:
            print("Error: Train data is invalid, can't train.")
            return None
        try:
            self.model = ARIMA(train_data[target_column], order=self.order)
            self.model_fit = self.model.fit()
            self.trained = True
            print("ARIMA model has been trained.")
            return self
        except Exception as e:
            print(f"Error during ARIMA model training: {e}")
            return None

    def predict(self, test_data, target_column="Adj Close"):
        """Makes predictions using the trained ARIMA model."""
        if not self.trained:
            print("Error: Model is not trained. Please train before using predict.")
            return None
        if test_data is None or target_column not in test_data.columns:
            print("Error: Test data is invalid, cannot make predictions")
            return None

        try:
            start_index = test_data.index[0]
            end_index = test_data.index[-1]
            predictions = self.model_fit.predict(start=start_index, end=end_index)

            return predictions
        except Exception as e:
            print(f"Error during ARIMA model predictions: {e}")
            return None
