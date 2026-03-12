import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

def check_stationarity(timeseries, name):
    print(f"\n--- ADF Test for: {name} ---")
    dftest = adfuller(timeseries.dropna(), maxlag=30)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
    return dftest[1] <= 0.05

def run_transformation():
    data_path = "Datas/EnergyConsumption/PJME_hourly.csv"
    results_dir = "Results/EnergyConsumption"
    fig_dir = "DataFig/EnergyConsumption/Transformation"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    # 1. Resample to Daily
    print("Resampling to Daily frequency...")
    df_daily = df.resample('D').mean()
    
    # 2. Transformations focusing on Weekly Seasonality
    df_daily['Log_MW'] = np.log(df_daily['PJME_MW'])
    df_daily['Log_Diff'] = df_daily['Log_MW'].diff()
    # Weekly Seasonal Difference (7 days)
    df_daily['Log_Diff_Weekly'] = df_daily['Log_Diff'].diff(7)
    
    # 3. ADF Tests
    stats_path = os.path.join(results_dir, "stationarity_tests_v2_daily.txt")
    with open(stats_path, 'w') as f:
        f.write("--- Stationarity Test Results (Daily, v2: Weekly Fix) ---\n")
        
        def log_adf(name, ts):
            dftest = adfuller(ts.dropna(), maxlag=30)
            f.write(f"\n{name}:\n")
            f.write(f"ADF Statistic: {dftest[0]}\n")
            f.write(f"p-value: {dftest[1]}\n")
            f.write(f"Stationary: {dftest[1] <= 0.05}\n")
            
        log_adf("Log + 1st Diff", df_daily['Log_Diff'])
        log_adf("Log + 1st Diff + Weekly Diff", df_daily['Log_Diff_Weekly'])
    
    # 4. Visualization
    def plot_diagnostics(ts, name, filename):
        ts_clean = ts.dropna()
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))
        ts_clean.plot(ax=axes[0], title=f'{name} - Time Series', color='teal', linewidth=1)
        axes[0].grid(True, alpha=0.3)
        plot_acf(ts_clean, lags=60, ax=axes[1], title=f'{name} - ACF')
        plot_pacf(ts_clean, lags=60, ax=axes[2], title=f'{name} - PACF', method='ywm')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, filename))
        plt.close()

    print("Generating diagnostic plots...")
    plot_diagnostics(df_daily['Log_Diff'], "Daily Log + 1st Diff", "v2_diagnostics_diff.png")
    plot_diagnostics(df_daily['Log_Diff_Weekly'], "Daily Log + 1st Diff + Weekly Diff", "v2_diagnostics_weekly.png")
    
    # Save transformed data
    processed_path = os.path.join(results_dir, "energy_stationary_v2.csv")
    df_daily.to_csv(processed_path)
    print(f"Saved stationary data (v2) to {processed_path}")

if __name__ == "__main__":
    run_transformation()
