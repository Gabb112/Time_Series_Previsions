import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def calculate_rmse(y_true, y_pred):
    """Calculates the Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mape(y_true, y_pred):
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return np.nan  # Handle cases where all true values are zero
    return (
        np.mean(
            np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]
            )
        )
        * 100
    )


def evaluate_model(y_true, y_pred):
    """Calculates and prints common evaluation metrics for time series models."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    print(f"Mean Absolute Error (MAE):   {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    return mae, rmse, mape

def plot_predictions(y_true, y_pred, model_name="Model"):
    """Plots the true vs. predicted values."""
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.values, label="True Values", color='blue')
    plt.plot(y_pred, label="Predictions", color='red', linestyle='--')
    plt.title(f"{model_name} Predictions vs True Values")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()