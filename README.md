# S&P 500 Stock Price Forecasting Project

## Overview

This project is a comprehensive exploration of time series forecasting, focusing on predicting the adjusted closing prices of stocks within the S&P 500 index.  We leverage historical stock data, S&P 500 index values, and company information to build and evaluate a range of forecasting models. This project serves as a demonstration of time-series data processing, feature engineering, model training, and evaluation skills.

## Problem Statement

Accurately forecasting stock prices is a challenging problem, due to many factors that affect stock prices, including market sentiment, economic indicators, and company-specific events. Time series analysis offers a powerful toolkit to tackle this problem. This project focuses on exploring and implementing different time series forecasting techniques on the S&P 500 dataset to demonstrate the model building process from end-to-end.

## Dataset

The dataset used for this project is sourced from Kaggle: [S&P 500 Stocks Dataset](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks). It consists of three main CSV files:

*   `sp500_companies.csv`: Contains information about the companies that constitute the S&P 500 index, including their sectors, industries, market capitalization, and business summaries.
*   `sp500_index.csv`: Contains the historical daily closing values of the S&P 500 index.
*   `sp500_stocks.csv`: Contains historical daily stock data for individual companies in the S&P 500, including Open, High, Low, Close, and Adj Close prices, along with Volume.

## Data Exploration

Before diving into model building, we perform a thorough exploration of the data. This includes:

*   **Loading and Inspecting DataFrames:** Loading the data from the CSV files and getting basic information about each dataframe using `head()`, `info()`, `describe()`, and `isnull().sum()`.
*   **Handling Date Columns:** Converting date columns to datetime format using the pandas `to_datetime` function.
*   **Visualizing Time Series:** Plotting the S&P 500 index over time to observe its trends.
*   **Identifying Missing Values:** Exploring and understanding missing data patterns, especially within `sp500_stocks.csv`, and dropping rows that do not contain adjusted close information

## Feature Engineering

To improve model accuracy and capture relevant patterns in the data, we engineered several features:

*   **Lag Features:** Created lagged features using `Adj Close` prices from 1, 7, and 30 days ago to capture the short-term and medium-term stock behavior.
*   **Rolling Window Features:** Computed rolling mean and standard deviation for the `Adj Close` prices using windows of 7, 30, and 90 days to smooth out daily fluctuations.
*   **Calendar Features:** Extracted calendar-based features (day of the week, month, and year) from the date column, to capture any seasonal patterns.
*   **Merging Company Data:** Incorporated additional information about each stock by merging the `sp500_stocks.csv` DataFrame with `sp500_companies.csv` based on the company symbol.

## Models

We implemented a variety of time series forecasting models, each with its unique approach to capturing the patterns in the data:

1.  **Persistence Model:** A baseline model that predicts the next day's `Adj Close` price as the previous day's value.
2.  **ARIMA Model:** An Autoregressive Integrated Moving Average model, trained and tuned to capture the time series dynamics.
3.  **Prophet Model:** A forecasting model from Facebook, used to capture seasonality and trend.
4.  **LSTM Model:** A Long Short-Term Memory neural network, trained to capture long-range time dependencies.
5. **XGBoost Model:** An XGBoost gradient boosting model, trained with engineered features for prediction.

## Training and Evaluation

*   **Time-Based Split:** Time-series data was split into training, validation, and test sets to prevent data leakage, preserving chronological order.
*   **Model Training:** Each model was trained using the appropriate training functions.
*   **Evaluation Metrics:** Models are evaluated using:
    *   **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values.
    *   **Root Mean Squared Error (RMSE):** The square root of the average squared difference between predicted and actual values.
    *   **Mean Absolute Percentage Error (MAPE):** Average percentage difference between predicted and actual values.
*   **Visual Analysis:**
     * Time Series plots of the S&P 500 index.
     * Plots to see real vs predicted values of each model.
     * Residual analysis plots.
*    **Rolling Forecast Evaluation:** A rolling forecast strategy is employed to evaluate the model’s performance on rolling test sets.
*    **Statistical Analysis:** t-tests are performed to understand statistical differences in the outputs of different models.

**Model Evaluation Results**
<details>
    <summary>Click to see the training and evaluation results</summary>
      <pre>
Number of rows with missing data: 1273705
        Date Symbol  Adj Close  Close  High  Low  Open  Volume
0 2010-01-04    MMM        NaN    NaN   NaN  NaN   NaN     NaN
1 2010-01-05    MMM        NaN    NaN   NaN  NaN   NaN     NaN
2 2010-01-06    MMM        NaN    NaN   NaN  NaN   NaN     NaN
3 2010-01-07    MMM        NaN    NaN   NaN  NaN   NaN     NaN
4 2010-01-08    MMM        NaN    NaN   NaN  NaN   NaN     NaN
Merged the stock data with company data based on Symbol column.

First 5 rows with engineered features: 
        Date Symbol  Adj Close     Close     High      Low      Open    Volume  Adj Close_lag_1  Adj Close_lag_7  Adj Close_lag_30  Adj Close_rolling_mean_7  Adj Close_rolling_std_7  Adj Close_rolling_mean_30  Adj Close_rolling_std_30  Adj Close_rolling_mean_90  Adj Close_rolling_std_90  day_of_week  month  year  Exchange Shortname    Longname              Sector    Industry  Currentprice  Marketcap      Ebitda  Revenuegrowth City     State Country  Fulltimeemployees                 Longbusinesssummary    Weight
0 2010-01-04    AOS  12.613409  14.31000  14.42000  14.28000  14.38000   718500.0      NaN           NaN           NaN                         NaN                      NaN                       NaN                      NaN                        NaN                      NaN             0      1    2010       NYQ      A. O. Smith  A. O. Smith Corporation   Industrials  Building Products   94.33     13340000000.0 1660000000.0            0.02  Milwaukee  Wisconsin    USA            12300.0  A. O. Smith Corporation manufactures and marke...  0.000179
1 2010-01-04    GPN   25.956106  27.36000  27.61000  27.25000  27.57000  3055100.0      NaN           NaN           NaN                         NaN                      NaN                       NaN                      NaN                        NaN                      NaN             0      1    2010       NYQ  Global Payments   Global Payments Inc.     Financials  Financial Services   159.71     55440000000.0  4276000000.0             0.11     Atlanta    Georgia    USA            28000.0  Global Payments Inc. provides payment technolo...  0.000513
2 2010-01-04    CRM   15.866663  16.04000  16.17000  15.88000  16.03000  6096400.0      NaN           NaN           NaN                         NaN                      NaN                       NaN                      NaN                        NaN                      NaN             0      1    2010       NYQ  Salesforce.com    Salesforce, Inc.            Technology     Software   289.36    275120000000.0 1277000000.0             0.22  San Francisco   California    USA            73541.0  Salesforce, Inc. provides Customer Relationshi...  0.005917
3 2010-01-04    HIG  22.943678  24.87000  24.92000  24.69000  24.86000  1132000.0      NaN           NaN           NaN                         NaN                      NaN                       NaN                      NaN                        NaN                      NaN             0      1    2010       NYQ    Hartford  The Hartford Financial Services Group, Inc., t...     Financials  Insurance   84.46     24690000000.0  2677000000.0             0.03    Hartford  Connecticut    USA            30500.0   The Hartford Financial Services Group, Inc., t...  0.000571
4 2010-01-04   HSIC   37.454155  37.82000  37.97000  37.56000  37.83000   1001100.0      NaN           NaN           NaN                         NaN                      NaN                       NaN                      NaN                        NaN                      NaN             0      1    2010       NYQ  Henry Schein  Henry Schein, Inc.          Healthcare  Healthcare Equipment   88.83     12170000000.0 1149000000.0             0.09    Melville    New York    USA            22000.0  Henry Schein, Inc. provides health care produc...  0.000157

[5 rows x 35 columns]

 Columns in the dataframe: 
['Date', 'Symbol', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close_lag_1', 'Adj Close_lag_7', 'Adj Close_lag_30', 'Adj Close_rolling_mean_7', 'Adj Close_rolling_std_7', 'Adj Close_rolling_mean_30', 'Adj Close_rolling_std_30', 'Adj Close_rolling_mean_90', 'Adj Close_rolling_std_90', 'day_of_week', 'month', 'year', 'Exchange', 'Shortname', 'Longname', 'Sector', 'Industry', 'Currentprice', 'Marketcap', 'Ebitda', 'Revenuegrowth', 'City', 'State', 'Country', 'Fulltimeemployees', 'Longbusinesssummary', 'Weight']
Data has been split in a time-based fashion.

Training set size: 370699
Validation set size: 123566
Test set size: 123566

First 5 rows of the train set
        Date Symbol  ...                                Longbusinesssummary    Weight
0 2010-01-04    AOS  ...  A. O. Smith Corporation manufactures and marke...  0.000179
1 2010-01-04   GOOG  ...  Alphabet Inc. offers various products and plat...  0.042309
2 2010-01-04    WMB  ...  The Williams Companies, Inc., together with it...  0.001173
3 2010-01-04     MO  ...  Altria Group, Inc., through its subsidiaries, ...  0.001642
4 2010-01-04   AMZN  ...  Amazon.com, Inc. engages in the retail sale of...  0.042550

[5 rows x 35 columns]
--- Persistence Model ---
Persistence model doesn't train, using last known values.

Persistence Model Evaluation:
Mean Absolute Error (MAE):   129.33
Root Mean Squared Error (RMSE): 217.55
Mean Absolute Percentage Error (MAPE): 162.68%
--- ARIMA ---
ARIMA model has been trained.

ARIMA Model Evaluation:
Mean Absolute Error (MAE):   91.07
Root Mean Squared Error (RMSE): 170.77
Mean Absolute Percentage Error (MAPE): 68.74%
--- Prophet ---
20:23:42 - cmdstanpy - INFO - Chain [1] start processing
20:24:03 - cmdstanpy - INFO - Chain [1] done processing
Prophet model has been trained.

Prophet Model Evaluation:
Mean Absolute Error (MAE):   87.34
Root Mean Squared Error (RMSE): 156.11
Mean Absolute Percentage Error (MAPE): 106.99%
--- LSTM ---
Epoch: 1/100, Training Loss:0.0227, Validation Loss: 0.0311
Epoch: 2/100, Training Loss:0.0243, Validation Loss: 0.0318
Epoch: 3/100, Training Loss:0.0219, Validation Loss: 0.0305
Epoch: 4/100, Training Loss:0.0226, Validation Loss: 0.0310
Epoch: 5/100, Training Loss:0.0225, Validation Loss: 0.0310
Epoch: 6/100, Training Loss:0.0229, Validation Loss: 0.0309
Epoch: 7/100, Training Loss:0.0219, Validation Loss: 0.0304
Epoch: 8/100, Training Loss:0.0222, Validation Loss: 0.0309
Epoch: 9/100, Training Loss:0.0224, Validation Loss: 0.0310
Epoch: 10/100, Training Loss:0.0223, Validation Loss: 0.0307
Epoch: 11/100, Training Loss:0.0223, Validation Loss: 0.0308
Epoch: 12/100, Training Loss:0.0217, Validation Loss: 0.0305
Epoch: 13/100, Training Loss:0.0222, Validation Loss: 0.0307
Epoch: 14/100, Training Loss:0.0220, Validation Loss: 0.0307
Epoch: 15/100, Training Loss:0.0224, Validation Loss: 0.0308
Epoch: 16/100, Training Loss:0.0220, Validation Loss: 0.0309
Epoch: 17/100, Training Loss:0.0224, Validation Loss: 0.0311
Epoch: 18/100, Training Loss:0.0217, Validation Loss: 0.0305
Epoch: 19/100, Training Loss:0.0218, Validation Loss: 0.0307
Epoch: 20/100, Training Loss:0.0225, Validation Loss: 0.0310
Epoch: 21/100, Training Loss:0.0220, Validation Loss: 0.0308
Epoch: 22/100, Training Loss:0.0232, Validation Loss: 0.0313
Epoch: 23/100, Training Loss:0.0230, Validation Loss: 0.0314
Epoch: 24/100, Training Loss:0.0217, Validation Loss: 0.0304
Epoch: 25/100, Training Loss:0.0218, Validation Loss: 0.0306
Epoch: 26/100, Training Loss:0.0224, Validation Loss: 0.0310
Epoch: 27/100, Training Loss:0.0222, Validation Loss: 0.0308
Epoch: 28/100, Training Loss:0.0220, Validation Loss: 0.0307
Epoch: 29/100, Training Loss:0.0221, Validation Loss: 0.0309
Epoch: 30/100, Training Loss:0.0222, Validation Loss: 0.0311
Epoch: 31/100, Training Loss:0.0215, Validation Loss: 0.0307
Epoch: 32/100, Training Loss:0.0214, Validation Loss: 0.0307
Epoch: 33/100, Training Loss:0.0222, Validation Loss: 0.0308
Epoch: 34/100, Training Loss:0.0220, Validation Loss: 0.0310
Epoch: 35/100, Training Loss:0.0221, Validation Loss: 0.0309
Epoch: 36/100, Training Loss:0.0225, Validation Loss: 0.0313
Epoch: 37/100, Training Loss:0.0218, Validation Loss: 0.0309
Epoch: 38/100, Training Loss:0.0225, Validation Loss: 0.0312
Epoch: 39/100, Training Loss:0.0213, Validation Loss: 0.0307
Epoch: 40/100, Training Loss:0.0224, Validation Loss: 0.0311
Epoch: 41/100, Training Loss:0.0219, Validation Loss: 0.0309
Epoch: 42/100, Training Loss:0.0218, Validation Loss: 0.0309
Epoch: 43/100, Training Loss:0.0218, Validation Loss: 0.0309
Epoch: 44/100, Training Loss:0.0216, Validation Loss: 0.0308
Epoch: 45/100, Training Loss:0.0216, Validation Loss: 0.0306
Epoch: 46/100, Training Loss:0.0217, Validation Loss: 0.0310
Epoch: 47/100, Training Loss:0.0217, Validation Loss: 0.0308
Epoch: 48/100, Training Loss:0.0211, Validation Loss: 0.0306
Epoch: 49/100, Training Loss:0.0224, Validation Loss: 0.0312
Epoch: 50/100, Training Loss:0.0218, Validation Loss: 0.0308
Epoch: 51/100, Training Loss:0.0225, Validation Loss: 0.0313
Epoch: 52/100, Training Loss:0.0220, Validation Loss: 0.0310
Epoch: 53/100, Training Loss:0.0219, Validation Loss: 0.0312
Epoch: 54/100, Training Loss:0.0218, Validation Loss: 0.0308
Epoch: 55/100, Training Loss:0.0221, Validation Loss: 0.0310
Epoch: 56/100, Training Loss:0.0212, Validation Loss: 0.0305
Epoch: 57/100, Training Loss:0.0215, Validation Loss: 0.0307
Epoch: 58/100, Training Loss:0.0215, Validation Loss: 0.0308
Epoch: 59/100, Training Loss:0.0214, Validation Loss: 0.0307
Epoch: 60/100, Training Loss:0.0223, Validation Loss: 0.0312
Epoch: 61/100, Training Loss:0.0213, Validation Loss: 0.0307
Epoch: 62/100, Training Loss:0.0216, Validation Loss: 0.0310
Epoch: 63/100, Training Loss:0.0219, Validation Loss: 0.0311
Epoch: 64/100, Training Loss:0.0216, Validation Loss: 0.0308
Epoch: 65/100, Training Loss:0.0211, Validation Loss: 0.0305
Epoch: 66/100, Training Loss:0.0216, Validation Loss: 0.0310
Epoch: 67/100, Training Loss:0.0220, Validation Loss: 0.0307
Epoch: 68/100, Training Loss:0.0216, Validation Loss: 0.0307
Epoch: 69/100, Training Loss:0.0223, Validation Loss: 0.0311
Epoch: 70/100, Training Loss:0.0209, Validation Loss: 0.0304
Epoch: 71/100, Training Loss:0.0214, Validation Loss: 0.0308
Epoch: 72/100, Training Loss:0.0211, Validation Loss: 0.0307
Epoch: 73/100, Training Loss:0.0211, Validation Loss: 0.0306
Epoch: 74/100, Training Loss:0.0211, Validation Loss: 0.0307
Epoch: 75/100, Training Loss:0.0225, Validation Loss: 0.0314
Epoch: 76/100, Training Loss:0.0214, Validation Loss: 0.0309
Epoch: 77/100, Training Loss:0.0210, Validation Loss: 0.0306
Epoch: 78/100, Training Loss:0.0214, Validation Loss: 0.0309
Epoch: 79/100, Training Loss:0.0211, Validation Loss: 0.0305
Epoch: 80/100, Training Loss:0.0217, Validation Loss: 0.0310
Epoch: 81/100, Training Loss:0.0215, Validation Loss: 0.0308
Epoch: 82/100, Training Loss:0.0215, Validation Loss: 0.0309
Epoch: 83/100, Training Loss:0.0208, Validation Loss: 0.0305
Epoch: 84/100, Training Loss:0.0219, Validation Loss: 0.0312
Epoch: 85/100, Training Loss:0.0211, Validation Loss: 0.0306
Epoch: 86/100, Training Loss:0.0210, Validation Loss: 0.0305
Epoch: 87/100, Training Loss:0.0216, Validation Loss: 0.0309
Epoch: 88/100, Training Loss:0.0211, Validation Loss: 0.0307
Epoch: 89/100, Training Loss:0.0212, Validation Loss: 0.0306
Epoch: 90/100, Training Loss:0.0216, Validation Loss: 0.0312
Epoch: 91/100, Training Loss:0.0222, Validation Loss: 0.0308
Epoch: 92/100, Training Loss:0.0214, Validation Loss: 0.0307
Epoch: 93/100, Training Loss:0.0219, Validation Loss: 0.0311
Epoch: 94/100, Training Loss:0.0220, Validation Loss: 0.0310
Epoch: 95/100, Training Loss:0.0220, Validation Loss: 0.0309
Epoch: 96/100, Training Loss:0.0204, Validation Loss: 0.0302
Epoch: 97/100, Training Loss:0.0219, Validation Loss: 0.0311
Epoch: 98/100, Training Loss:0.0219, Validation Loss: 0.0313
Epoch: 99/100, Training Loss:0.0215, Validation Loss: 0.0307
Epoch: 100/100, Training Loss:0.0214, Validation Loss: 0.0306
LSTM model has been trained.
      </pre>
    </details>
## XGBoost model failed to train and give feature importances:

The XGBoost model failed to train due to an issue with the datatypes of the input dataframe, which was using object types.

**Training and Evaluation Results**
*   **Persistence Model:**
    *   MAE: 129.33
    *   RMSE: 217.55
    *   MAPE: 162.68%
*   **ARIMA Model:**
    *   MAE: 91.07
    *   RMSE: 170.77
    *   MAPE: 68.74%
*  **Prophet Model**
    *   MAE: 87.34
    *   RMSE: 156.11
    *   MAPE: 106.99%
*   **LSTM Model:**
    *   MAE: 87.96
    *   RMSE: 165.78
    *   MAPE: 76.69%

## Conclusion

The ARIMA, Prophet, and LSTM models performed better than the baseline Persistence Model and gave promising results. However, there is still room to improve the performance of the models by implementing hyperparameter tuning, cross-validation and other feature engineering techniques. XGBoost failed to train, this is to be explored further.

## Next Steps

As part of further work, I will be implementing the following:
- Detailed exploratory data analysis to see data and features' interactions and potential outliers
- Hyperparameter tuning and cross-validation for all the models to improve the performance
- Try implementing more advanced models
- Feature selection to remove unnecessary features that are not useful for the model
- Implement model explainability to better understand the decision process of the model.

## How to Run the Code

1.  Clone the repository.
2.  Install the requirements using: `pip install -r requirements.txt`.
3.  Run the data download script: `python src/download_data.py`.
4.  Run the main script: `python src/main.py`.
5.  To see detailed results and exploration, open the notebooks in the `notebooks` folder.

## File Structure

*   `your_project/`
    *   `data/`
        *   `raw/`
            *   `1023/`
                *   `sp500_companies.csv`
                *   `sp500_index.csv`
                *   `sp500_stocks.csv`
    *   `models/`
        *   `__init__.py`
        *   `arima.py`
        *   `baseline.py`
        *   `lstm.py`
        *   `model_trainer.py`
        *   `model_utils.py`
        *   `prophet.py`
        *   `xgboost_model.py`
    *   `features/`
        *   `__init__.py`
        *   `build_features.py`
    *   `notebooks/`
        *    `eda.ipynb`
        *    `feature_engineering_exploration.ipynb`
        *    `model_tuning.ipynb`
        *   `model_evaluation.ipynb`
     *   `src/`
        *   `download_data.py`
        *   `main.py`
    *   `tests/`
        *   `test_models.py`
    *   `requirements.txt`
    *   `README.md`
    *   `LICENSE`
