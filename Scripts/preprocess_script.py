import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os
import seaborn as sns

sns.set(style="whitegrid")

def test_stationarity(timeseries, window=12, title="Time Series"):
    # Calculate rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

    # Plot rolling statistics
    plt.figure(figsize=(12, 6))
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title(f'Rolling Mean & Standard Deviation - {title}')
    
    # Save the plot
    filename = f"DataFig/{title.replace(' ', '_').lower()}_stationarity.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

    # Perform Dickey-Fuller test
    print(f'Results of Dickey-Fuller Test ({title}):')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    
    # Print to console
    print(dfoutput)
    print("-" * 30)
    
    # Save to text file
    os.makedirs('Results', exist_ok=True)
    with open('Results/adf_test_results.txt', 'a') as f:
        f.write(f'--- Results of Dickey-Fuller Test ({title}) ---\n')
        f.write(dfoutput.to_string() + '\n')
        f.write("-" * 30 + '\n\n')

def preprocess_data():
    # Make sure to clear the previous results file if it exists
    if os.path.exists('Results/adf_test_results.txt'):
        os.remove('Results/adf_test_results.txt')

    print("Loading data...")
    df = pd.read_csv('Datas/air_passengers.csv', parse_dates=['Month'])
    df.set_index('Month', inplace=True)
    ts = df['Passengers']
    
    # Ensure DataFig directory exists
    os.makedirs('DataFig', exist_ok=True)
    
    # 1. Test original data
    test_stationarity(ts, title="Original Data")
    
    # 2. Log Transformation (to stabilize variance)
    ts_log = np.log(ts)
    test_stationarity(ts_log, title="Log Transformed Data")
    
    # 3. Differencing (to remove trend)
    ts_log_diff = ts_log - ts_log.shift()
    # Drop first NaN value before testing
    ts_log_diff.dropna(inplace=True) 
    test_stationarity(ts_log_diff, title="Log Diff Data")
    
    # 4. Seasonal Differencing (to remove seasonality, period=12 for monthly)
    ts_log_seasonal_diff = ts_log - ts_log.shift(12)
    ts_log_seasonal_diff.dropna(inplace=True)
    test_stationarity(ts_log_seasonal_diff, title="Log Seasonal Diff Data")
    
    # 5. Combined Differencing (Log + Diff + Seasonal Diff)
    # Taking the seasonal difference of the already differenced data
    ts_log_diff_seasonal_diff = ts_log_diff - ts_log_diff.shift(12)
    ts_log_diff_seasonal_diff.dropna(inplace=True)
    test_stationarity(ts_log_diff_seasonal_diff, title="Log Diff Seasonal Diff Data")
    
    # 6. Save preprocessed data for modeling
    os.makedirs('Results', exist_ok=True)
    df_preprocessed = pd.DataFrame({
        'Log_Passengers': ts_log,
        'Log_Diff': ts_log - ts_log.shift(),
        'Log_Seasonal_Diff': ts_log - ts_log.shift(12),
        'Log_Diff_Seasonal_Diff': (ts_log - ts_log.shift()) - (ts_log - ts_log.shift()).shift(12)
    })
    df_preprocessed.to_csv('Results/air_passengers_preprocessed.csv')
    print("Saved preprocessed data to Results/air_passengers_preprocessed.csv")

if __name__ == "__main__":
    preprocess_data()
