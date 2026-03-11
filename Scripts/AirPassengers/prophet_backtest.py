import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def run_backtest():
    print("Loading data for Backtesting...")
    df = pd.read_csv('Datas/air_passengers.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    df.rename(columns={'Month': 'ds', 'Passengers': 'y'}, inplace=True)
    
    # Split point: Training up to the end of 1954
    split_date = '1954-12-31'
    train_df = df[df['ds'] <= split_date]
    test_df = df[df['ds'] > split_date]
    
    print(f"Training on data up to {split_date} ({len(train_df)} months).")
    print(f"Testing on data from 1955-01-01 to end ({len(test_df)} months).")
    
    # Initialize the advanced model (Bayesian MCMC + Multiplicative)
    print("\nTraining Advanced Prophet model (this may take a moment due to MCMC 300)...")
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        mcmc_samples=300
    )
    model.add_seasonality(name='yearly', period=365.25, fourier_order=20)
    
    model.fit(train_df)
    
    # Generate forecast for the entire period (including future test period)
    # The total number of points to forecast is the length of test_df
    future = model.make_future_dataframe(periods=len(test_df), freq='MS')
    
    print("\nSimulating forecast for 1955-1960...")
    forecast = model.predict(future)
    
    # Extract predictions for the test period
    forecast_test = forecast[forecast['ds'] > split_date]
    
    # Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(test_df['y'], forecast_test['yhat']))
    mae = mean_absolute_error(test_df['y'], forecast_test['yhat'])
    
    print(f"\nBacktest Evaluation Metrics (1955-1960):")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    
    # Plotting
    plt.figure(figsize=(15, 8))
    
    # Actual Data
    plt.plot(train_df['ds'], train_df['y'], 'k.', label='Actual (Train: -1954)')
    plt.plot(test_df['ds'], test_df['y'], 'g.', label='Actual (Test: 1955-1960)')
    
    # Forecast
    plt.plot(forecast['ds'], forecast['yhat'], color='red', label='Prophet Forecast')
    
    # Uncertainty Interval
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2, label='95% Uncertainty Interval')
    
    # Separation Line
    plt.axvline(x=pd.to_datetime(split_date), color='blue', linestyle='--', alpha=0.7)
    plt.text(pd.to_datetime(split_date), plt.ylim()[1]*0.9, '  Forecast Start (1955)', color='blue', fontweight='bold')
    
    plt.title('Air Passengers: Long-term Backtest (Training until 1954, Predicting 1955-1960)')
    plt.xlabel('Date')
    plt.ylabel('Passengers')
    plt.legend()
    
    os.makedirs('DataFig', exist_ok=True)
    plot_path = 'DataFig/prophet_backtest_1955.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nBacktest complete. Plot saved to: {plot_path}")
    
    os.makedirs('Results', exist_ok=True)
    with open('Results/prophet_backtest_summary.txt', 'w') as f:
        f.write("--- Prophet Backtest (Train up to 1954) Summary ---\n")
        f.write(f"Split Date: {split_date}\n")
        f.write(f"RMSE (Jan 1955 - Dec 1960): {rmse:.2f}\n")
        f.write(f"MAE  (Jan 1955 - Dec 1960): {mae:.2f}\n")

if __name__ == "__main__":
    run_backtest()
