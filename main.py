import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from features.build_features import FeatureEngineering
from models.baseline import PersistenceModel
from models.arima import ARIMAModel
from models.prophet import ProphetModel
from models.lstm import LSTMModel
from models.xgboost_model import XGBoostModel
from models.model_utils import evaluate_model, plot_predictions


# Project Directory Setup
project_dir = os.getcwd()
raw_data_dir = os.path.join(project_dir, "data", "raw", "1023")


# Data Loading and Exploration
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from: {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred loading data: {e}")
        return None


def explore_dataframe(df, df_name):
    if df is None:
        return

    print(f"\n--- Exploring DataFrame: {df_name} ---")
    print("\nFirst 5 rows:\n", df.head())
    print("\nDataframe Information:\n", df.info())
    print("\nDescriptive Statistics:\n", df.describe(include="all"))
    print("\nMissing Values (sum per column):\n", df.isnull().sum())


def convert_to_datetime(df, date_columns):
    if df is None:
        return None
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="raise")
                print(f"Column '{col}' converted to datetime.")
            except ValueError as e:
                print(f"Error converting column '{col}' to datetime: {e}")
        else:
            print(f"Warning: column '{col}' not found in dataframe.")
    return df


def plot_sp500_index(sp500_index_df):
    if sp500_index_df is not None:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x="Date", y="S&P500", data=sp500_index_df)
        plt.title("S&P 500 Index over time")
        plt.xlabel("Date")
        plt.ylabel("S&P 500 Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def preprocess_stocks_data(sp500_stocks_df):
    if sp500_stocks_df is None:
        return None
    missing_data = sp500_stocks_df.iloc[:, 2:].isnull().all(axis=1)
    print("\nNumber of rows with missing data:", missing_data.sum())
    print(sp500_stocks_df[missing_data].head())
    sp500_stocks_df = sp500_stocks_df.dropna(subset=["Adj Close"])
    sp500_stocks_df = sp500_stocks_df.sort_values(by="Date").reset_index(drop=True)
    return sp500_stocks_df


def create_and_merge_features(sp500_stocks_df, sp500_companies_df):
    if sp500_stocks_df is None:
        return None
    feature_eng = FeatureEngineering()
    lags_to_use = [1, 7, 30]
    sp500_stocks_df = feature_eng.create_lag_features(
        sp500_stocks_df, "Adj Close", lags_to_use
    )
    windows_to_use = [7, 30, 90]
    sp500_stocks_df = feature_eng.create_rolling_features(
        sp500_stocks_df, "Adj Close", windows_to_use
    )
    sp500_stocks_df = feature_eng.create_calendar_features(sp500_stocks_df, "Date")
    sp500_stocks_df = feature_eng.merge_company_data(
        sp500_stocks_df, sp500_companies_df
    )
    print("\nFirst 5 rows with engineered features: \n", sp500_stocks_df.head())
    print("\nColumns in the dataframe: \n", sp500_stocks_df.columns.tolist())
    return sp500_stocks_df


def split_data(sp500_stocks_df):
    if sp500_stocks_df is None:
        return None, None, None
    feature_eng = FeatureEngineering()
    train_data, val_data, test_data = feature_eng.time_series_split(
        sp500_stocks_df, "Date"
    )
    if train_data is not None:
        print(f"\nTraining set size: {len(train_data)}")
        print(f"Validation set size: {len(val_data)}")
        print(f"Test set size: {len(test_data)}")
        print("\nFirst 5 rows of the train set\n", train_data.head())
    return train_data, val_data, test_data


def train_and_evaluate_model(model, model_name, train_data, test_data, val_data=None):
    print(f"\n--- {model_name} ---")
    if model_name == "Persistence":
        model.train(train_data)  # Persistence model doesn't train
        predictions = model.predict(test_data)
        if predictions is not None:
            print(f"\n{model_name} Model Evaluation:")
            evaluate_model(
                test_data["Adj Close"].iloc[1:], predictions.iloc[1:]
            )
            plot_predictions(
                test_data["Adj Close"].iloc[1:], predictions.iloc[1:], model_name=f"{model_name} Model"
            )
    elif model_name == "LSTM":
        model.train(train_data, val_data, target_column="Adj Close")
        if model.trained:
            predictions = model.predict(test_data, target_column="Adj Close")
            if predictions is not None:
                print(f"\n{model_name} Model Evaluation:")
                evaluate_model(test_data["Adj Close"].iloc[10:], predictions)
                plot_predictions(test_data["Adj Close"].iloc[10:], predictions, model_name=f"{model_name} Model")
    else:
        model.train(train_data, target_column="Adj Close", val_data=val_data)
        if model.trained:
            predictions = model.predict(test_data, target_column="Adj Close")
            if predictions is not None:
                print(f"\n{model_name} Model Evaluation:")
                evaluate_model(test_data["Adj Close"], predictions)
                plot_predictions(test_data["Adj Close"], predictions, model_name=f"{model_name} Model")


# --- Main Script ---
if __name__ == "__main__":
    # 1. Load Data
    sp500_companies_path = os.path.join(raw_data_dir, "sp500_companies.csv")
    sp500_index_path = os.path.join(raw_data_dir, "sp500_index.csv")
    sp500_stocks_path = os.path.join(raw_data_dir, "sp500_stocks.csv")

    sp500_companies_df = load_data(sp500_companies_path)
    sp500_index_df = load_data(sp500_index_path)
    sp500_stocks_df = load_data(sp500_stocks_path)

    # 2. Explore Data
    explore_dataframe(sp500_companies_df, "sp500_companies")
    explore_dataframe(sp500_index_df, "sp500_index")
    explore_dataframe(sp500_stocks_df, "sp500_stocks")

    # 3. Convert Date Columns
    sp500_index_df = convert_to_datetime(sp500_index_df, ["Date"])
    sp500_stocks_df = convert_to_datetime(sp500_stocks_df, ["Date"])

    # 4. Plot S&P 500 Index
    plot_sp500_index(sp500_index_df)

    # 5. Preprocess Stock Data
    sp500_stocks_df = preprocess_stocks_data(sp500_stocks_df)

    # 6. Create and Merge Features
    sp500_stocks_df = create_and_merge_features(sp500_stocks_df, sp500_companies_df)

    # 7. Split Data
    train_data, val_data, test_data = split_data(sp500_stocks_df)

    # 8. Model Training and Evaluation
    if train_data is not None:
        # --- Model Training and Evaluation ---
        train_and_evaluate_model(PersistenceModel(), "Persistence", train_data, test_data)
        train_and_evaluate_model(ARIMAModel(order=(5, 1, 0)), "ARIMA", train_data, test_data)
        train_and_evaluate_model(ProphetModel(), "Prophet", train_data, test_data)
        train_and_evaluate_model(LSTMModel(), "LSTM", train_data, test_data, val_data)
        train_and_evaluate_model(XGBoostModel(), "XGBoost", train_data, test_data, val_data)