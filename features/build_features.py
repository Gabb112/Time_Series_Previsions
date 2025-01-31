# src/features/build_features.py (modified)
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import numpy as np


class FeatureEngineering:
    def __init__(self):
        pass

    def create_lag_features(self, df, target_column, lags):
        if df is None or target_column not in df.columns:
            return df

        for lag in lags:
            df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)
        return df

    def create_rolling_features(self, df, target_column, windows):
        if df is None or target_column not in df.columns:
            return df

        for window in windows:
            df[f"{target_column}_rolling_mean_{window}"] = (
                df[target_column].rolling(window=window).mean()
            )
            df[f"{target_column}_rolling_std_{window}"] = (
                df[target_column].rolling(window=window).std()
            )
        return df

    def create_calendar_features(self, df, date_column):
        if df is None or date_column not in df.columns:
            return df

        df["day_of_week"] = df[date_column].dt.dayofweek
        df["month"] = df[date_column].dt.month
        df["year"] = df[date_column].dt.year
        return df

    def merge_company_data(self, stock_df, company_df):
        if stock_df is None or company_df is None:
            return stock_df

        # Ensure 'Symbol' is a column in both DataFrames before the merge
        if "Symbol" not in stock_df.columns or "Symbol" not in company_df.columns:
            print(
                "Error: 'Symbol' column not found in one or both dataframes. Cannot merge"
            )
            return stock_df
        # Merging on symbol
        merged_df = pd.merge(stock_df, company_df, on="Symbol", how="left")
        print(f"Merged the stock data with company data based on Symbol column.")
        return merged_df

    def time_series_split(self, df, date_column, test_size=0.2, val_size=0.2):
        if df is None or date_column not in df.columns:
            return None, None, None

        df = df.sort_values(by=date_column)
        df = df.dropna(subset=[date_column])  # Remove any rows with empty date values
        df = df.reset_index(drop=True)

        # Calculate sizes of different sets
        total_size = len(df)
        test_len = int(total_size * test_size)
        val_len = int(total_size * val_size)
        train_len = total_size - test_len - val_len

        if train_len <= 0:
            print(
                "Error: Not enough data for train/val/test split. Ensure you have enough data"
            )
            return None, None, None

        # Split data
        train_data = df.iloc[:train_len]
        val_data = df.iloc[train_len : train_len + val_len]
        test_data = df.iloc[train_len + val_len :]
        print(f"Data has been split in a time-based fashion.")
        return train_data, val_data, test_data
