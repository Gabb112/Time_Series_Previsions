import pandas as pd
from prophet import Prophet
from models.model_trainer import ModelTrainer


class ProphetModel(ModelTrainer):
    """Prophet model for forecasting time series."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.trained = False

    def train(self, train_data, val_data=None, target_column="Adj Close"):
        """Trains the Prophet model."""
        if train_data is None or target_column not in train_data.columns:
            print("Error: Train data is invalid, can't train.")
            return None
        try:
            # Prophet requires a column called 'ds' and 'y', hence rename
            df_prophet = train_data.rename(columns={"Date": "ds", target_column: "y"})
            self.model = Prophet()
            self.model.fit(df_prophet)
            self.trained = True
            print("Prophet model has been trained.")
            return self
        except Exception as e:
            print(f"Error during prophet model training: {e}")
            return None

    def predict(self, test_data, target_column="Adj Close"):
        """Makes predictions using the trained Prophet model."""
        if not self.trained:
            print("Error: Model is not trained. Please train before using predict.")
            return None
        if test_data is None or target_column not in test_data.columns:
            print("Error: Test data is invalid, cannot make predictions")
            return None

        try:
            df_prophet = test_data.rename(columns={"Date": "ds"})
            forecast = self.model.predict(df_prophet)
            predictions = forecast["yhat"].values
            return predictions
        except Exception as e:
            print(f"Error during Prophet model predictions: {e}")
            return None
