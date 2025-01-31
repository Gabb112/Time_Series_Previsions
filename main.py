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
from models.model_utils import evaluate_model


# Project Directory Setup
project_dir = os.getcwd()
raw_data_dir = os.path.join(project_dir, "data", "raw", "1023")


# Function to load data with error handling
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


# Function to explore a dataframe
def explore_dataframe(df, df_name):
    if df is None:
        return

    print(f"\n--- Exploring DataFrame: {df_name} ---")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataframe Information:")
    print(df.info())

    print("\nDescriptive Statistics:")
    print(
        df.describe(include="all")
    )  # Include all to get info for both numeric and categorical

    print("\nMissing Values (sum per column):")
    print(df.isnull().sum())


# Function to convert date columns to datetime
def convert_to_datetime(df, date_columns):
    if df is None:
        return None

    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(
                    df[col], errors="raise"
                )  # errors='raise' throws error if conversion fails
                print(f"Column '{col}' converted to datetime.")
            except ValueError as e:
                print(f"Error converting column '{col}' to datetime: {e}")

        else:
            print(f"Warning: column '{col}' not found in dataframe.")
    return df


# --- Main Script ---
if __name__ == "__main__":
    # Construct file paths
    sp500_companies_path = os.path.join(raw_data_dir, "sp500_companies.csv")
    sp500_index_path = os.path.join(raw_data_dir, "sp500_index.csv")
    sp500_stocks_path = os.path.join(raw_data_dir, "sp500_stocks.csv")

    # Load dataframes
    sp500_companies_df = load_data(sp500_companies_path)
    sp500_index_df = load_data(sp500_index_path)
    sp500_stocks_df = load_data(sp500_stocks_path)

    # Explore each dataframe
    explore_dataframe(sp500_companies_df, "sp500_companies")
    explore_dataframe(sp500_index_df, "sp500_index")
    explore_dataframe(sp500_stocks_df, "sp500_stocks")

    # Convert date columns to datetime format
    sp500_index_df = convert_to_datetime(sp500_index_df, ["Date"])
    sp500_stocks_df = convert_to_datetime(sp500_stocks_df, ["Date"])

    # Example of additional exploration
    if sp500_index_df is not None:
        # Plotting S&P 500 index
        plt.figure(figsize=(10, 6))
        sns.lineplot(x="Date", y="S&P500", data=sp500_index_df)
        plt.title("S&P 500 Index over time")
        plt.xlabel("Date")
        plt.ylabel("S&P 500 Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    if sp500_stocks_df is not None:
        # Example: Check for rows with mostly missing data in sp500_stocks_df
        missing_data = sp500_stocks_df.iloc[:, 2:].isnull().all(axis=1)
        print("\nNumber of rows with missing data:", missing_data.sum())

        # Display rows with mainly missing data
        print(sp500_stocks_df[missing_data].head())

        # Example - Feature engineering, assuming we will use Adj Close as the target feature and we will be using all stocks data
        sp500_stocks_df = sp500_stocks_df.dropna(subset=["Adj Close"])
        sp500_stocks_df = sp500_stocks_df.sort_values(by="Date")

        # Create an instance of FeatureEngineering
        feature_eng = FeatureEngineering()

        # Lag features for adj close price
        lags_to_use = [1, 7, 30]
        sp500_stocks_df = feature_eng.create_lag_features(
            sp500_stocks_df, "Adj Close", lags_to_use
        )

        # Rolling window features
        windows_to_use = [7, 30, 90]
        sp500_stocks_df = feature_eng.create_rolling_features(
            sp500_stocks_df, "Adj Close", windows_to_use
        )

        # Calendar features
        sp500_stocks_df = feature_eng.create_calendar_features(sp500_stocks_df, "Date")

        # Merge company data
        sp500_stocks_df = feature_eng.merge_company_data(
            sp500_stocks_df, sp500_companies_df
        )

        print("\nFirst 5 rows with engineered features: ")
        print(sp500_stocks_df.head())
        print("\n Columns in the dataframe: ")
        print(sp500_stocks_df.columns.tolist())

        # Example: Time-series data splitting
        train_data, val_data, test_data = feature_eng.time_series_split(
            sp500_stocks_df, "Date"
        )

        if train_data is not None:
            print(f"\nTraining set size: {len(train_data)}")
            print(f"Validation set size: {len(val_data)}")
            print(f"Test set size: {len(test_data)}")
            print("\nFirst 5 rows of the train set")
            print(train_data.head())

            # Baseline Model
            baseline_model = PersistenceModel()
            baseline_model.train(
                train_data
            )  # Not needed for persistance model but kept for consistancy
            baseline_predictions = baseline_model.predict(test_data)

            if baseline_predictions is not None:
                print("\nBaseline (Persistence) Model Evaluation:")
                evaluate_model(
                    test_data["Adj Close"].iloc[1:], baseline_predictions.iloc[1:]
                )

            # ARIMA Model
            arima_model = ARIMAModel(order=(5, 1, 0))  # Example order
            arima_model.train(train_data, target_column="Adj Close")
            if arima_model.trained:
                arima_predictions = arima_model.predict(
                    test_data, target_column="Adj Close"
                )
                if arima_predictions is not None:
                    print("\nARIMA Model Evaluation:")
                    evaluate_model(test_data["Adj Close"], arima_predictions)

            # Prophet Model
            prophet_model = ProphetModel()
            prophet_model.train(train_data, target_column="Adj Close")
            if prophet_model.trained:
                prophet_predictions = prophet_model.predict(
                    test_data, target_column="Adj Close"
                )
                if prophet_predictions is not None:
                    print("\nProphet Model Evaluation:")
                    evaluate_model(test_data["Adj Close"], prophet_predictions)

            # LSTM Model
            lstm_model = LSTMModel()
            lstm_model.train(train_data, val_data, target_column="Adj Close")
            if lstm_model.trained:
                lstm_predictions = lstm_model.predict(
                    test_data, target_column="Adj Close"
                )
                if lstm_predictions is not None:
                    print("\nLSTM Model Evaluation:")
                    # Since LSTM is predicting after 'seq_length' of the data, hence using appropriate indeces
                    evaluate_model(test_data["Adj Close"].iloc[10:], lstm_predictions)

            # XGBoost Model
            xgboost_model = XGBoostModel()
            xgboost_model.train(train_data, val_data, target_column="Adj Close")
            if xgboost_model.trained:
                xgboost_predictions = xgboost_model.predict(
                    test_data, target_column="Adj Close"
                )
                if xgboost_predictions is not None:
                    print("\nXGBoost Model Evaluation:")
                    evaluate_model(test_data["Adj Close"], xgboost_predictions)
