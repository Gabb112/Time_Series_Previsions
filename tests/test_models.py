# tests/test_models.py
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from models.baseline import PersistenceModel
from models.arima import ARIMAModel
from models.prophet import ProphetModel
from models.lstm import LSTMModel
from models.xgboost_model import XGBoostModel


class TestModels(unittest.TestCase):
    def setUp(self):
        # Sample data for testing (adjust to fit your needs)
        self.train_data = pd.DataFrame({
            'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D')),
            'Adj Close': np.linspace(100, 200, 100),
            'feature1': np.linspace(1, 100, 100),
            'feature2': np.linspace(100, 1, 100)
        })
        self.val_data = pd.DataFrame({
            'Date': pd.to_datetime(pd.date_range(start='2023-04-10', periods=30, freq='D')),
            'Adj Close': np.linspace(200, 230, 30),
            'feature1': np.linspace(100, 130, 30),
            'feature2': np.linspace(1, 30, 30)
        })
        self.test_data = pd.DataFrame({
            'Date': pd.to_datetime(pd.date_range(start='2023-05-10', periods=30, freq='D')),
            'Adj Close': np.linspace(230, 260, 30),
            'feature1': np.linspace(130, 160, 30),
            'feature2': np.linspace(30, 60, 30)
        })

        self.lstm_test_data = pd.DataFrame({
            'Date': pd.to_datetime(pd.date_range(start='2023-06-10', periods=50, freq='D')),
            'Adj Close': np.linspace(260, 310, 50),
            'feature1': np.linspace(160, 210, 50),
            'feature2': np.linspace(60, 110, 50)
        })

    def test_persistence_model(self):
        model = PersistenceModel()
        model.train(self.train_data)
        predictions = model.predict(self.test_data)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.test_data))

    def test_arima_model(self):
        model = ARIMAModel(order=(1, 1, 1))
        model.train(self.train_data, target_column='Adj Close')
        self.assertTrue(model.trained)
        predictions = model.predict(self.test_data, target_column='Adj Close')
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.test_data))

    def test_prophet_model(self):
        model = ProphetModel()
        model.train(self.train_data, target_column='Adj Close')
        self.assertTrue(model.trained)
        predictions = model.predict(self.test_data, target_column='Adj Close')
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.test_data))

    def test_lstm_model(self):
        model = LSTMModel()
        model.train(self.train_data, self.val_data, target_column='Adj Close')
        self.assertTrue(model.trained)
        predictions = model.predict(self.lstm_test_data, target_column='Adj Close')
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.lstm_test_data)-10)

    def test_xgboost_model(self):
        model = XGBoostModel()
        features = [col for col in self.train_data.columns if col != 'Adj Close' and col != 'Date']
        model.train(self.train_data, self.val_data, target_column='Adj Close')
        self.assertTrue(model.trained)
        predictions = model.predict(self.test_data, target_column='Adj Close')
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.test_data))


if __name__ == '__main__':
    unittest.main()