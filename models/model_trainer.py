import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
from models.model_utils import evaluate_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ModelTrainer(ABC):
    """Abstract class for training time series models."""

    def __init__(self):
        pass

    @abstractmethod
    def train(self, train_data, val_data, target_column):
        """Abstract method for training the model."""
        pass

    @abstractmethod
    def predict(self, test_data):
        """Abstract method for making predictions with the model."""
        pass

    def evaluate(self, y_true, y_pred):
        """Evaluates the model with MAE, RMSE, and MAPE."""
        return evaluate_model(y_true, y_pred)
