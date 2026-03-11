import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def run_prophet_advanced():
    print("Loading data...")
    df = pd.read_csv('Datas/air_passengers.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    df.rename(columns={'Month': 'ds', 'Passengers': 'y'}, inplace=True)
    
    # Train-test split (last 24 months for testing)
    train_df = df.iloc[:-24]
    test_df = df.iloc[-24:]
    
    print("\nTraining Prophet model with custom periodicities...")
    # Initialize Prophet with multiplicative mode and MCMC for Bayesian inference
    model = Prophet(
        yearly_seasonality=False, # Disable default yearly to add custom one with more Fourier terms
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        mcmc_samples=300
    )
    
    # 1. Add yearly seasonality with higher flexibility (Fourier order = 20)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=20)
    
    # 2. Add special periodicity (if any, e.g., quarterly)
    # model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
    
    # Fit the model
    model.fit(train_df)
    
    # Create future dataframe for 5 years (60 months) to see the long term simulation
    future = model.make_future_dataframe(periods=60, freq='MS')
    
    print("\nSimulating future (5 years forward)...")
    forecast = model.predict(future)
    
    # Plotting
    plt.figure(figsize=(15, 8))
    
    # Training and Test data
    plt.plot(train_df['ds'], train_df['y'], 'k.', label='Actual (Train)')
    plt.plot(test_df['ds'], test_df['y'], 'g.', label='Actual (Test)')
    
    # Forecast
    plt.plot(forecast['ds'], forecast['yhat'], color='red', label='Prophet Trend + Custom Seasonality')
    
    # Uncertainty Interval
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2, label='95% Uncertainty Interval')
    
    # Mark the start of future simulation
    plt.axvline(x=test_df['ds'].iloc[-1], color='blue', linestyle='--', alpha=0.5)
    plt.text(test_df['ds'].iloc[-1], plt.ylim()[1]*0.9, '  Future Simulation Start', color='blue')
    
    plt.title('Air Passengers: 5-Year Bayesian Simulation (Custom Seasonality)')
    plt.xlabel('Date')
    plt.ylabel('Passengers')
    plt.legend()
    
    os.makedirs('DataFig', exist_ok=True)
    plot_path = 'DataFig/prophet_future_simulation.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Components plot
    fig = model.plot_components(forecast)
    fig.savefig('DataFig/prophet_advanced_components.png')
    plt.close(fig)
    
    # Evaluation (on the 24-month test set)
    forecast_test = forecast[(forecast['ds'] >= test_df['ds'].min()) & (forecast['ds'] <= test_df['ds'].max())]
    rmse = np.sqrt(mean_squared_error(test_df['y'], forecast_test['yhat']))
    
    print(f"\nSimulation Complete.")
    print(f"Test RMSE with custom seasonality: {rmse:.2f}")
    print(f"Forecast plot saved to: {plot_path}")

if __name__ == "__main__":
    run_prophet_advanced()
