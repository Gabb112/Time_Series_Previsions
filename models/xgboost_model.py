import pandas as pd
import xgboost as xgb
from models.model_trainer import ModelTrainer


class XGBoostModel(ModelTrainer):
    """XGBoost model for forecasting time series."""

    def __init__(self, params=None):
        super().__init__()
        self.params = (
            params
            if params
            else {"objective": "reg:squarederror", "eval_metric": "rmse", "seed": 42}
        )
        self.model = None
        self.trained = False

    def train(self, train_data, val_data, target_column="Adj Close"):
        """Trains the XGBoost model."""
        if train_data is None or target_column not in train_data.columns:
            print("Error: Train data is invalid, can't train.")
            return None
        if val_data is None or target_column not in val_data.columns:
            print("Error: Val data is invalid, can't train.")
            return None

        try:
            features = [
                col
                for col in train_data.columns
                if col != target_column and col != "Date"
            ]
            xgb_train = xgb.DMatrix(
                train_data[features], label=train_data[target_column]
            )
            xgb_val = xgb.DMatrix(val_data[features], label=val_data[target_column])
            evals = [(xgb_train, "train"), (xgb_val, "eval")]
            self.model = xgb.train(
                self.params,
                xgb_train,
                num_boost_round=100,
                evals=evals,
                early_stopping_rounds=10,
                verbose_eval=False,
            )
            self.trained = True
            print("XGBoost model has been trained.")
            return self
        except Exception as e:
            print(f"Error during XGBoost model training: {e}")
            return None

    def predict(self, test_data, target_column="Adj Close"):
        """Makes predictions using the trained XGBoost model."""
        if not self.trained:
            print("Error: Model is not trained. Please train before using predict.")
            return None

        if test_data is None or target_column not in test_data.columns:
            print("Error: Test data is invalid, cannot make predictions")
            return None

        try:
            features = [
                col
                for col in test_data.columns
                if col != target_column and col != "Date"
            ]
            xgb_test = xgb.DMatrix(test_data[features])
            predictions = self.model.predict(xgb_test)
            return predictions
        except Exception as e:
            print(f"Error during XGBoost model predictions: {e}")
            return None
