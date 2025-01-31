import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project Directory Setup (Same as your previous script)
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
    # You can add your EDA code here
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
        missing_data = (
            sp500_stocks_df.iloc[:, 2:].isnull().all(axis=1)
        )  # Check all columns from index 2 onwards
        print("\nNumber of rows with missing data:", missing_data.sum())

        # Display rows with mainly missing data
        print(sp500_stocks_df[missing_data].head())
