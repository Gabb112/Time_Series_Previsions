# S&P 500 Stock Price Forecasting

## Project Overview

This project aims to forecast stock prices for companies within the S&P 500 index using various time series forecasting techniques. We use historical stock data, S&P 500 index data, and company information to predict future 'Adj Close' prices.

## Dataset

The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks), and consists of three main components:

*   `sp500_companies.csv`: Contains information about each company within the S&P 500.
*   `sp500_index.csv`: Contains the historical S&P 500 index values.
*   `sp500_stocks.csv`: Contains the historical stock prices for each company.

## Data Exploration

We begin by exploring the dataset.

### S&P 500 Companies Data (`sp500_companies.csv`)

*   **Shape:** 502 rows x 16 columns.
*   **Features:** Includes company information such as sector, industry, market cap, and a short business summary.
*   **Missing Values:** There are missing values in columns like `Ebitda`, `Revenuegrowth`, `State`, and `Fulltimeemployees`.

<details>
<summary>Click to see the data exploration results</summary>
<pre>
--- Exploring DataFrame: sp500_companies ---
First 5 rows:
Exchange Symbol ... Longbusinesssummary Weight
0 NMS AAPL ... Apple Inc. designs, manufactures, and markets ... 0.069209
1 NMS NVDA ... NVIDIA Corporation provides graphics and compu... 0.059350
2 NMS MSFT ... Microsoft Corporation develops and supports so... 0.058401
3 NMS AMZN ... Amazon.com, Inc. engages in the retail sale of... 0.042550
4 NMS GOOGL ... Alphabet Inc. offers various products and plat... 0.042309

[5 rows x 16 columns]

Dataframe Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 502 entries, 0 to 501
Data columns (total 16 columns):

Column Non-Null Count Dtype
0 Exchange 502 non-null object
1 Symbol 502 non-null object
2 Shortname 502 non-null object
3 Longname 502 non-null object
4 Sector 502 non-null object
5 Industry 502 non-null object
6 Currentprice 502 non-null float64
7 Marketcap 502 non-null int64
8 Ebitda 473 non-null float64
9 Revenuegrowth 499 non-null float64
10 City 502 non-null object
11 State 482 non-null object
12 Country 502 non-null object
13 Fulltimeemployees 493 non-null float64
14 Longbusinesssummary 502 non-null object
15 Weight 502 non-null float64
dtypes: float64(5), int64(1), object(10)
memory usage: 62.9+ KB
None

Descriptive Statistics:
Exchange Symbol ... Longbusinesssummary Weight
count 502 502 ... 502 502.000000
unique 4 502 ... 499 NaN
top NYQ AAPL ... Fox Corporation operates as a news, sports, an... NaN
freq 348 1 ... 2 NaN
mean NaN NaN ... NaN 0.001992
std NaN NaN ... NaN 0.006189
min NaN NaN ... NaN 0.000084
25% NaN NaN ... NaN 0.000348
50% NaN NaN ... NaN 0.000667
75% NaN NaN ... NaN 0.001409
max NaN NaN ... NaN 0.069209

[11 rows x 16 columns]

Missing Values (sum per column):
Exchange 0
Symbol 0
Shortname 0
Longname 0
Sector 0
Industry 0
Currentprice 0
Marketcap 0
Ebitda 29
Revenuegrowth 3
City 0
State 20
Country 0
Fulltimeemployees 9
Longbusinesssummary 0
Weight 0
dtype: int64
</pre>

</details>

### S&P 500 Index Data (`sp500_index.csv`)
Shape: 2517 rows x 2 columns.

Features: Contains the S&P 500 index value over time.

Missing Values: No missing values.

<details>
  <summary>Click to see the data exploration results</summary>
    <pre>
--- Exploring DataFrame: sp500_index ---

First 5 rows:
         Date   S&P500
0  2014-12-22  2078.54
1  2014-12-23  2082.17
2  2014-12-24  2081.88
3  2014-12-26  2088.77
4  2014-12-29  2090.57

Dataframe Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2517 entries, 0 to 2516
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Date    2517 non-null   object 
 1   S&P500  2517 non-null   float64
dtypes: float64(1), object(1)
memory usage: 39.5+ KB
None

Descriptive Statistics:
              Date       S&P500
count         2517  2517.000000
unique        2517          NaN
top     2014-12-22          NaN
freq             1          NaN
mean           NaN  3346.351605
std            NaN  1078.204274
min            NaN  1829.080000
25%            NaN  2428.370000
50%            NaN  2999.910000
75%            NaN  4199.120000
max            NaN  6090.270000

Missing Values (sum per column):
Date      0
S&P500    0
dtype: int64
  </pre>
  </details>
S&P 500 Stocks Data (sp500_stocks.csv)
Shape: 1891536 rows x 8 columns.

Features: Contains daily stock information for each company, including Adj Close, Open, High, Low and Volume.

Missing Values: There are many missing values in the price-related columns, especially in the earlier data.

<details>
<summary>Click to see the data exploration results</summary>
  <pre>
--- Exploring DataFrame: sp500_stocks ---

First 5 rows:
Date Symbol Adj Close Close High Low Open Volume
0 2010-01-04 MMM NaN NaN NaN NaN NaN NaN
1 2010-01-05 MMM NaN NaN NaN NaN NaN NaN
2 2010-01-06 MMM NaN NaN NaN NaN NaN NaN
3 2010-01-07 MMM NaN NaN NaN NaN NaN NaN
4 2010-01-08 MMM NaN NaN NaN NaN NaN NaN

Dataframe Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1891536 entries, 0 to 1891535
Data columns (total 8 columns):

Column Dtype
0 Date object
1 Symbol object
2 Adj Close float64
3 Close float64
4 High float64
5 Low float64
6 Open float64
7 Volume float64
dtypes: float64(6), object(2)
memory usage: 115.5+ MB
None

Descriptive Statistics:
Date Symbol Adj Close ... Low Open Volume
count 1891536 1891536 617831.000000 ... 617831.000000 617831.000000 6.178310e+05
unique 3768 502 NaN ... NaN NaN NaN
top 2010-01-04 MMM NaN ... NaN NaN NaN
freq 502 3768 NaN ... NaN NaN NaN
mean NaN NaN 79.672357 ... 86.480997 87.460302 9.347125e+06
std NaN NaN 102.742931 ... 103.300770 104.519845 4.771669e+07
min NaN NaN 0.203593 ... 0.216250 0.218000 0.000000e+00
25% NaN NaN 26.572459 ... 32.299999 32.689999 1.144000e+06
50% NaN NaN 49.821613 ... 58.500000 59.119999 2.453400e+06
75% NaN NaN 94.831036 ... 103.889999 105.000000 5.657850e+06
max NaN NaN 1702.530029 ... 1696.900024 1706.400024 3.692928e+09

[11 rows x 8 columns]

Missing Values (sum per column):
Date 0
Symbol 0
Adj Close 1273705
Close 1273705
High 1273705
Low 1273705
Open 1273705
Volume 1273705
dtype: int64
</pre>
</details>

## Feature Engineering

The following features were added to the dataset to enhance the model's performance:

*   **Lag Features:** Past values of 'Adj Close' at lags of 1, 7, and 30 days.
*   **Rolling Window Features:** Rolling mean and standard deviation of 'Adj Close' over 7, 30 and 90 days.
*   **Calendar Features:** Day of the week, month and year.
*   **Company Data:** Merging the stock data with company information from `sp500_companies.csv`.

## Models

We evaluated several time series forecasting models:

1.  **Persistence Model:** A baseline model that predicts the next value to be the previous value.
2.  **ARIMA:** An autoregressive integrated moving average model.
3.  **Prophet:** A forecasting model developed by Facebook.
4.  **LSTM:** A Long Short-Term Memory recurrent neural network.
5.  **XGBoost:** A gradient boosting algorithm.

## Results

The models were evaluated on a time-based split of the data into training, validation, and test sets. We use Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) as evaluation metrics. Here are the results of each model on the test set:

**Training and Evaluation Results**
<details>
<summary>Click to see the training and evaluation results</summary>
<pre>
Number of rows with missing data: 1273705
Date Symbol Adj Close Close High Low Open Volume
0 2010-01-04 MMM NaN NaN NaN NaN NaN NaN
1 2010-01-05 MMM NaN NaN NaN NaN NaN NaN
2 2010-01-06 MMM NaN NaN NaN NaN NaN NaN
3 2010-01-07 MMM NaN NaN NaN NaN NaN NaN
4 2010-01-08 MMM NaN NaN NaN NaN NaN NaN
Merged the stock data with company data based on Symbol column.
First 5 rows with engineered features:
Date Symbol ... Longbusinesssummary Weight
0 2010-01-04 AOS ... A. O. Smith Corporation manufactures and marke... 0.000179
1 2010-01-04 GPN ... Global Payments Inc. provides payment technolo... 0.000513
2 2010-01-04 CRM ... Salesforce, Inc. provides Customer Relationshi... 0.005917
3 2010-01-04 HIG ... The Hartford Financial Services Group, Inc., t... 0.000571
4 2010-01-04 HSIC ... Henry Schein, Inc. provides health care produc... 0.000157

[5 rows x 35 columns]

Columns in the dataframe:
['Date', 'Symbol', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close_lag_1', 'Adj Close_lag_7', 'Adj Close_lag_30', 'Adj Close_rolling_mean_7', 'Adj Close_rolling_std_7', 'Adj Close_rolling_mean_30', 'Adj Close_rolling_std_30', 'Adj Close_rolling_mean_90', 'Adj Close_rolling_std_90', 'day_of_week', 'month', 'year', 'Exchange', 'Shortname', 'Longname', 'Sector', 'Industry', 'Currentprice', 'Marketcap', 'Ebitda', 'Revenuegrowth', 'City', 'State', 'Country', 'Fulltimeemployees', 'Longbusinesssummary', 'Weight']
Data has been split in a time-based fashion.

Training set size: 370699
Validation set size: 123566
Test set size: 123566

First 5 rows of the train set
Date Symbol ... Longbusinesssummary Weight
0 2010-01-04 AOS ... A. O. Smith Corporation manufactures and marke... 0.000179
1 2010-01-04 GOOG ... Alphabet Inc. offers various products and plat... 0.042309
2 2010-01-04 WMB ... The Williams Companies, Inc., together with it... 0.001173
3 2010-01-04 MO ... Altria Group, Inc., through its subsidiaries, ... 0.001642
4 2010-01-04 AMZN ... Amazon.com, Inc. engages in the retail sale of... 0.042550

[5 rows x 35 columns]
Persistence model doesn't train, using last known values.

Baseline (Persistence) Model Evaluation:
Mean Absolute Error (MAE): 129.33
Root Mean Squared Error (RMSE): 217.55
Mean Absolute Percentage Error (MAPE): 162.68%
ARIMA model has been trained.

ARIMA Model Evaluation:
Mean Absolute Error (MAE): 91.07
Root Mean Squared Error (RMSE): 170.77
Mean Absolute Percentage Error (MAPE): 68.74%
13:42:41 - cmdstanpy - INFO - Chain [1] start processing
13:42:59 - cmdstanpy - INFO - Chain [1] done processing
Prophet model has been trained.

Prophet Model Evaluation:
Mean Absolute Error (MAE): 87.34
Root Mean Squared Error (RMSE): 156.11
Mean Absolute Percentage Error (MAPE): 106.99%
Epoch: 1/100, Training Loss:0.0232, Validation Loss: 0.0311
Epoch: 2/100, Training Loss:0.0244, Validation Loss: 0.0317
Epoch: 3/100, Training Loss:0.0246, Validation Loss: 0.0319
Epoch: 4/100, Training Loss:0.0219, Validation Loss: 0.0304
Epoch: 5/100, Training Loss:0.0250, Validation Loss: 0.0318
Epoch: 6/100, Training Loss:0.0229, Validation Loss: 0.0310
Epoch: 7/100, Training Loss:0.0220, Validation Loss: 0.0307
Epoch: 8/100, Training Loss:0.0219, Validation Loss: 0.0306
Epoch: 9/100, Training Loss:0.0226, Validation Loss: 0.0311
Epoch: 10/100, Training Loss:0.0226, Validation Loss: 0.0308
Epoch: 11/100, Training Loss:0.0220, Validation Loss: 0.0306
Epoch: 12/100, Training Loss:0.0235, Validation Loss: 0.0314
Epoch: 13/100, Training Loss:0.0213, Validation Loss: 0.0304
Epoch: 14/100, Training Loss:0.0211, Validation Loss: 0.0302
Epoch: 15/100, Training Loss:0.0219, Validation Loss: 0.0306
Epoch: 16/100, Training Loss:0.0232, Validation Loss: 0.0313
Epoch: 17/100, Training Loss:0.0229, Validation Loss: 0.0310
Epoch: 18/100, Training Loss:0.0224, Validation Loss: 0.0310
Epoch: 19/100, Training Loss:0.0225, Validation Loss: 0.0310
Epoch: 20/100, Training Loss:0.0235, Validation Loss: 0.0312
Epoch: 21/100, Training Loss:0.0224, Validation Loss: 0.0307
Epoch: 22/100, Training Loss:0.0230, Validation Loss: 0.0310
Epoch: 23/100, Training Loss:0.0220, Validation Loss: 0.0307
Epoch: 24/100, Training Loss:0.0233, Validation Loss: 0.0313
Epoch: 25/100, Training Loss:0.0221, Validation Loss: 0.0308
Epoch: 26/100, Training Loss:0.0220, Validation Loss: 0.0305
Epoch: 27/100, Training Loss:0.0221, Validation Loss: 0.0308
Epoch: 28/100, Training Loss:0.0209, Validation Loss: 0.0302
Epoch: 29/100, Training Loss:0.0222, Validation Loss: 0.0308
Epoch: 30/100, Training Loss:0.0223, Validation Loss: 0.0309
Epoch: 31/100, Training Loss:0.0223, Validation Loss: 0.0310
Epoch: 32/100, Training Loss:0.0223, Validation Loss: 0.0307
Epoch: 33/100, Training Loss:0.0222, Validation Loss: 0.0308
Epoch: 34/100, Training Loss:0.0215, Validation Loss: 0.0305
Epoch: 35/100, Training Loss:0.0226, Validation Loss: 0.0312
Epoch: 36/100, Training Loss:0.0227, Validation Loss: 0.0312
Epoch: 37/100, Training Loss:0.0218, Validation Loss: 0.0306
Epoch: 38/100, Training Loss:0.0224, Validation Loss: 0.0309
Epoch: 39/100, Training Loss:0.0219, Validation Loss: 0.0307
Epoch: 40/100, Training Loss:0.0224, Validation Loss: 0.0310
Epoch: 41/100, Training Loss:0.0223, Validation Loss: 0.0309
Epoch: 42/100, Training Loss:0.0222, Validation Loss: 0.0310
Epoch: 43/100, Training Loss:0.0222, Validation Loss: 0.0310
Epoch: 44/100, Training Loss:0.0211, Validation Loss: 0.0304
Epoch: 45/100, Training Loss:0.0229, Validation Loss: 0.0313
Epoch: 46/100, Training Loss:0.0224, Validation Loss: 0.0310
Epoch: 47/100, Training Loss:0.0225, Validation Loss: 0.0311
Epoch: 48/100, Training Loss:0.0214, Validation Loss: 0.0305
Epoch: 49/100, Training Loss:0.0223, Validation Loss: 0.0313
Epoch: 50/100, Training Loss:0.0220, Validation Loss: 0.0309
Epoch: 51/100, Training Loss:0.0216, Validation Loss: 0.0309
Epoch: 52/100, Training Loss:0.0226, Validation Loss: 0.0315
Epoch: 53/100, Training Loss:0.0216, Validation Loss: 0.0309
Epoch: 54/100, Training Loss:0.0227, Validation Loss: 0.0316
Epoch: 55/100, Training Loss:0.0218, Validation Loss: 0.0309
Epoch: 56/100, Training Loss:0.0214, Validation Loss: 0.0308
Epoch: 57/100, Training Loss:0.0216, Validation Loss: 0.0313
Epoch: 58/100, Training Loss:0.0221, Validation Loss: 0.0314
Epoch: 59/100, Training Loss:0.0225, Validation Loss: 0.0316
Epoch: 60/100, Training Loss:0.0219, Validation Loss: 0.0314
Epoch: 61/100, Training Loss:0.0215, Validation Loss: 0.0311
Epoch: 62/100, Training Loss:0.0219, Validation Loss: 0.0313
Epoch: 63/100, Training Loss:0.0224, Validation Loss: 0.0320
Epoch: 64/100, Training Loss:0.0229, Validation Loss: 0.0320
Epoch: 65/100, Training Loss:0.0218, Validation Loss: 0.0316
Epoch: 66/100, Training Loss:0.0210, Validation Loss: 0.0309
Epoch: 67/100, Training Loss:0.0220, Validation Loss: 0.0315
Epoch: 68/100, Training Loss:0.0231, Validation Loss: 0.0325
Epoch: 69/100, Training Loss:0.0220, Validation Loss: 0.0315
Epoch: 70/100, Training Loss:0.0215, Validation Loss: 0.0314
Epoch: 71/100, Training Loss:0.0221, Validation Loss: 0.0315
Epoch: 72/100, Training Loss:0.0201, Validation Loss: 0.0305
Epoch: 73/100, Training Loss:0.0213, Validation Loss: 0.0310
Epoch: 74/100, Training Loss:0.0211, Validation Loss: 0.0309
Epoch: 75/100, Training Loss:0.0211, Validation Loss: 0.0309
Epoch: 76/100, Training Loss:0.0205, Validation Loss: 0.0305
Epoch: 77/100, Training Loss:0.0219, Validation Loss: 0.0312
Epoch: 78/100, Training Loss:0.0227, Validation Loss: 0.0316
Epoch: 79/100, Training Loss:0.0216, Validation Loss: 0.0314
Epoch: 80/100, Training Loss:0.0238, Validation Loss: 0.0321
Epoch: 81/100, Training Loss:0.0213, Validation Loss: 0.0306
Epoch: 82/100, Training Loss:0.0206, Validation Loss: 0.0303
Epoch: 83/100, Training Loss:0.0217, Validation Loss: 0.0309
Epoch: 84/100, Training Loss:0.0215, Validation Loss: 0.0311
Epoch: 85/100, Training Loss:0.0218, Validation Loss: 0.0307
Epoch: 86/100, Training Loss:0.0214, Validation Loss: 0.0307
Epoch: 87/100, Training Loss:0.0211, Validation Loss: 0.0306
Epoch: 88/100, Training Loss:0.0227, Validation Loss: 0.0312
Epoch: 89/100, Training Loss:0.0204, Validation Loss: 0.0302
Epoch: 90/100, Training Loss:0.0215, Validation Loss: 0.0307
Epoch: 91/100, Training Loss:0.0225, Validation Loss: 0.0311
Epoch: 92/100, Training Loss:0.0209, Validation Loss: 0.0304
Epoch: 93/100, Training Loss:0.0221, Validation Loss: 0.0311
Epoch: 94/100, Training Loss:0.0224, Validation Loss: 0.0310
Epoch: 95/100, Training Loss:0.0228, Validation Loss: 0.0315
Epoch: 96/100, Training Loss:0.0224, Validation Loss: 0.0310
Epoch: 97/100, Training Loss:0.0218, Validation Loss: 0.0308
Epoch: 98/100, Training Loss:0.0218, Validation Loss: 0.0309
Epoch: 99/100, Training Loss:0.0217, Validation Loss: 0.0308
Epoch: 100/100, Training Loss:0.0229, Validation Loss: 0.0313
LSTM model has been trained.
</pre>

</details>

* **Persistence Model:**
* MAE: 129.33
* RMSE: 217.55
* MAPE: 162.68%

* **ARIMA Model:**
* MAE: 91.07
* RMSE: 170.77
* MAPE: 68.74%

* **Prophet Model**
* MAE: 87.34
* RMSE: 156.11
* MAPE: 106.99%

* **LSTM Model:**
* MAE: 87.96
* RMSE: 165.78
* MAPE: 76.69%


Conclusion
The ARIMA, Prophet and LSTM models performed better than the baseline Persistence Model and gave promising results. However, there is still room to improve the performance of the models by implementing hyperparameter tuning, cross validation and other feature engineering techniques.
