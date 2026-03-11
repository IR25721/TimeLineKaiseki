import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson
import os

def residual_analysis():
    print("Loading data...")
    df = pd.read_csv('Datas/air_passengers.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    df.rename(columns={'Month': 'ds', 'Passengers': 'y'}, inplace=True)
    
    # Using the advanced settings with custom seasonality
    print("Fitting Advanced Prophet model (multiplicative + 20 Fourier order)...")
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model.add_seasonality(name='yearly', period=365.25, fourier_order=20)
    model.fit(df) # Fit on full data for residual analysis
    
    print("Calculating residuals...")
    forecast = model.predict(df)
    df['yhat'] = forecast['yhat']
    
    # Calculate Log Residuals for multiplicative model assessment
    # e_t = log(y_t) - log(yhat_t)
    df['residuals'] = np.log(df['y']) - np.log(df['yhat'])
    
    os.makedirs('DataFig', exist_ok=True)
    os.makedirs('Results', exist_ok=True)
    
    # 1. Log Residuals over time
    plt.figure(figsize=(12, 5))
    plt.plot(df['ds'], df['residuals'], marker='o', linestyle='-', color='purple', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Prophet Model Log Residuals over Time')
    plt.xlabel('Date')
    plt.ylabel('Log Residual (log(y) - log(yhat))')
    plt.savefig('DataFig/prophet_residuals_time.png')
    plt.close()
    
    # 2. Histogram of Log Residuals (Check for Normality)
    plt.figure(figsize=(10, 5))
    sns.histplot(df['residuals'], kde=True, color='purple')
    plt.title('Distribution of Log Residuals')
    plt.savefig('DataFig/prophet_residuals_dist.png')
    plt.close()
    
    # 3. ACF of Log Residuals (Check for Autocorrelation)
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(df['residuals'], lags=24, ax=ax)
    plt.title('ACF of Log Residuals')
    plt.savefig('DataFig/prophet_residuals_acf.png')
    plt.close()
    
    # 4. Statistics
    dw = durbin_watson(df['residuals'])
    mean_res = df['residuals'].mean()
    std_res = df['residuals'].std()
    
    print(f"\nResidual Statistics (Log Scale):")
    print(f"Mean Log Residual: {mean_res:.4f}")
    print(f"Std Dev of Log Residuals: {std_res:.4f}")
    print(f"Durbin-Watson Statistic (on Log Res): {dw:.4f}")
    
    with open('Results/prophet_residual_analysis.txt', 'w') as f:
        f.write("--- Prophet Residual Analysis Summary (Log Scale) ---\n")
        f.write(f"Mean Log Residual: {mean_res:.4f}\n")
        f.write(f"Std Deviation:    {std_res:.4f}\n")
        f.write(f"Durbin-Watson:    {dw:.4f}\n")
        f.write("\nNote: Calculated using log(y) - log(yhat) for multiplicative model consistency.\n")
        f.write("\nInterpretation Hint:\n")
        f.write("- Mean close to 0: Unbiased predictions.\n")
        f.write("- Durbin-Watson near 2: No significant autocorrelation.\n")
        f.write("- Histogram bell-curved: Normally distributed errors.\n")

    print("\nResidual analysis complete. Plots saved to DataFig/.")

if __name__ == "__main__":
    residual_analysis()
