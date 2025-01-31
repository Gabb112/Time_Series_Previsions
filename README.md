# S&P 500 Stock Price Time Series Forecasting

## Project Overview

This project aims to build and showcase time series forecasting models for predicting S&P 500 stock prices. It utilizes historical stock data, S&P 500 index data, and company information to train and evaluate various forecasting models. The project demonstrates key data science skills, including data preprocessing, feature engineering, time series analysis, and model building.

## Dataset

The project utilizes three datasets sourced from Kaggle:

1.  **`sp500_companies.csv`**: Contains information about S&P 500 companies, including sector, industry, market cap, etc.
2.  **`sp500_index.csv`**: Contains historical data for the S&P 500 index.
3.  **`sp500_stocks.csv`**: Contains historical daily stock price data for all S&P 500 companies.

These datasets are downloaded and stored in the `data/raw` directory.

## Project Structure

*   **`data/raw/1023/`**: Contains the downloaded raw datasets.
*   **`notebooks/`**: Contains any Jupyter notebooks used for EDA, the initial exploration will be performed inside a `EDA.ipynb`.
*   **`src/data/`**: Contains scripts for data loading and data cleaning.
*   **`src/features/`**: Contains the `build_features.py` file, which includes the class responsible for engineering time series and external features from our raw data.
*   **`src/models/`**:  Will contain scripts related to training and evaluating models (to be implemented).
*   **`src/reports/`**: Will store generated reports, visualizations, and model evaluation metrics (to be implemented).
*   **`main.py`**: The main script to orchestrate and execute the different project components, including feature engineering and data splitting.
*   **.gitignore**: Specifies intentionally untracked files that Git should ignore.
*   **`requirements.txt`**: Lists the project's Python dependencies.

## Getting Started

### Prerequisites

*   Python 3.7+
*   `pip`
*   An active internet connection to download the data through kaggle.

### Installation

1.  Clone the repository:

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  Create a virtual environment (optional, but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate # On macOS and Linux
    # venv\Scripts\activate # On Windows
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

1.  To download the data and run the data loading, cleaning, feature engineering, and data splitting:

    ```bash
    python main.py
    ```
    This will download the data (from Kaggle if it is the first time you are doing this step), and will output the engineered features and train/val/test datasets.

## Methodology

The project follows these steps:

1.  **Data Loading and Exploration:**
    *   Loads the three CSV datasets using `pandas`.
    *   Performs initial exploration with descriptive statistics, data type checks, missing value identification, and visualization.

2.  **Feature Engineering:**
    *   Creates lag features for stock prices (`Adj Close`).
    *   Generates rolling statistics (mean and standard deviation) using different window sizes.
    *   Extracts calendar-based features (day of week, month, year).
    *   Merges company-specific features from the company dataset into the stock dataset.
    *   Uses a feature engineering class in `src/features` for maintainability.

3.  **Data Splitting:**
    *   Splits the data into training, validation, and test sets, respecting the temporal order.
    *   Uses a custom `time_series_split` method inside the feature engineering class.

4.  **Time Series Modeling:**
     *   To be implemented: models that will include ARIMA, Prophet, LSTM, XGBoost

5.  **Model Evaluation:**
      *   To be implemented: evaluating the performance of models based on proper time-series metrics, including a model comparison.

## Results

(This section will be filled with plots and metrics when you start modelling)

## Future Work

*   Implement different time series forecasting models (ARIMA, Prophet, LSTM, XGBoost).
*   Fine-tune model hyperparameters to improve accuracy.
*   Explore more advanced feature engineering strategies.
*   Incorporate external data sources (e.g., news sentiment, economic indicators).
*   Develop a more sophisticated model evaluation pipeline.
*   Create a model selection function to choose the best model.

## Lessons Learned

(This section can include details of the challenges you faced and how you overcame them).

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
