import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def run_prophet():
    print("Loading data for Prophet...")
    # Load original data
    df = pd.read_csv('Datas/air_passengers.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Prophet requires columns to be named 'ds' and 'y'
    df.rename(columns={'Month': 'ds', 'Passengers': 'y'}, inplace=True)
    
    # We will pass the original raw data to Prophet, since it can handle multiplicative seasonality internally
    prophet_df = df[['ds', 'y']]
    
    # Train-test split (last 24 months for testing)
    train_df = prophet_df.iloc[:-24]
    test_df = prophet_df.iloc[-24:]
    
    # The actual original values for evaluation later (same as test_df in this case)
    train_actuals = train_df.copy()
    test_actuals = test_df.copy()
    
    print("\nTraining Prophet model...")
    # Initialize Prophet. We enable MCMC sampling to get full Bayesian inference
    # and set seasonality_mode to 'multiplicative' to let Prophet handle the increasing variance
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        mcmc_samples=300             # Increased to 300 as suggested
    )
    
    # Fit the model
    model.fit(train_df)
    
    # Create future dataframe for the next 24 months
    future = model.make_future_dataframe(periods=24, freq='MS')
    
    print("\nGenerating forecasts (with MCMC sampling, this may take a moment)...")
    forecast = model.predict(future)
    
    # Extract just the corresponding test period predictions
    forecast_test = forecast.iloc[-24:]
    
    # Since we used seasonality_mode='multiplicative', Prophet's output 'yhat' is already in the original scale!
    # No need to np.exp() it.
    forecast_original = forecast_test['yhat']
    forecast_lower = forecast_test['yhat_lower']
    forecast_upper = forecast_test['yhat_upper']
    
    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(test_actuals['y'], forecast_original))
    mae = mean_absolute_error(test_actuals['y'], forecast_original)
    
    print("\nProphet Evaluation Metrics (Multiplicative Mode):")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    
    os.makedirs('Results', exist_ok=True)
    with open('Results/prophet_evaluation_metrics.txt', 'w') as f:
        f.write(f"--- Prophet Evaluation Metrics (Multiplicative Mode) ---\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE:  {mae:.2f}\n")
        f.write(f"Note: Model used {model.mcmc_samples} MCMC samples for Bayesian inference.\n")
        
    print("Generating plots...")

    os.makedirs('DataFig', exist_ok=True)
    
    # Full dataset plot
    full_ds = df['ds']
    full_y = df['y']
    
    plt.figure(figsize=(14, 7))
    plt.plot(train_actuals['ds'], train_actuals['y'], label='Train Data')
    plt.plot(test_actuals['ds'], test_actuals['y'], label='Actual Test Data')
    plt.plot(forecast_test['ds'], forecast_original, color='red', label='Prophet Forecast')
    
    # Add uncertainty intervals
    plt.fill_between(forecast_test['ds'], forecast_lower, forecast_upper, color='red', alpha=0.2, label='95% Uncertainty Interval')
    
    plt.title('Air Passengers Forecast with Prophet (Bayesian MCMC)')
    plt.xlabel('Date')
    plt.ylabel('Passengers')
    plt.legend()
    
    plot_path = 'DataFig/prophet_forecast.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Also save the Prophet built-in components plot
    # Reverting scale to original for component plot visualization is complex, 
    # so we plot components based on the log scale model output.
    fig = model.plot_components(forecast)
    comp_path = 'DataFig/prophet_components.png'
    fig.savefig(comp_path)
    plt.close(fig)
    
    print(f"Saved forecast plots to DataFig/")
    
if __name__ == "__main__":
    run_prophet()
