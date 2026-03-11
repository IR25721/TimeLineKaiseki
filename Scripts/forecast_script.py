import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def forecast_sarima():
    print("Loading preprocessed data...")
    # Read the data. We only need the Log_Passengers because SARIMAX will handle differencing.
    df = pd.read_csv('Results/air_passengers_preprocessed.csv')
    df['Month'] = pd.date_range(start='1949-01-01', periods=len(df), freq='MS')
    df.set_index('Month', inplace=True)
    ts_log = df['Log_Passengers']
    
    # Train-Test Split (Let's keep the last 24 months for testing)
    train = ts_log[:-24]
    test = ts_log[-24:]

    print("\nTraining SARIMAX model...")
    # Model parameters:
    # order=(p, d, q) = (0, 1, 1) -> MA(1) with 1 regular difference
    # seasonal_order=(P, D, Q, s) = (0, 1, 0, 12) -> 1 seasonal difference of period 12
    # These parameters were identified throughout our stationarity and ACF/PACF testing.
    model = sm.tsa.statespace.SARIMAX(
        train, 
        order=(0, 1, 1), 
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    
    # Save the full model summary
    os.makedirs('Results', exist_ok=True)
    with open('Results/sarimax_model_summary.txt', 'w') as f:
        f.write("--- SARIMAX(0,1,1)(0,1,1,12) Model Summary ---\n\n")
        f.write(results.summary().as_text())
    print("Saved SARIMAX model summary to Results/sarimax_model_summary.txt")

    # Forecasting
    print("Generating forecasts...")
    forecast_log = results.forecast(steps=len(test))
    
    # Convert predictions back to original scale (exp(log(y)))
    train_original = np.exp(train)
    test_original = np.exp(test)
    forecast_original = np.exp(forecast_log)
    
    # Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(test_original, forecast_original))
    mae = mean_absolute_error(test_original, forecast_original)
    
    with open('Results/evaluation_metrics.txt', 'w') as f:
        f.write(f"--- Evaluation Metrics (Test Set Data) ---\n")
        f.write(f"RMSE (Root Mean Squared Error): {rmse:.2f}\n")
        f.write(f"MAE (Mean Absolute Error):      {mae:.2f}\n")
    print("\nEvaluation Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(train_original.index, train_original, label='Train Data')
    plt.plot(test_original.index, test_original, label='Actual Test Data')
    plt.plot(test_original.index, forecast_original, color='red', label='SARIMA Forecast')
    
    plt.title('Air Passengers Forecast with SARIMA(0,1,1)(0,1,1,12)')
    plt.xlabel('Date')
    plt.ylabel('Passengers')
    plt.legend()
    
    os.makedirs('DataFig', exist_ok=True)
    plot_path = 'DataFig/sarima_forecast.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved forecast plot to {plot_path}")

if __name__ == "__main__":
    forecast_sarima()
