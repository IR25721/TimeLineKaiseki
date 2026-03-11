import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# Set style
sns.set(style="whitegrid")

def perform_eda():
    print("Loading data...")
    df = pd.read_csv('Datas/air_passengers.csv', parse_dates=['Month'])
    df.set_index('Month', inplace=True)
    
    # Ensure DataFig directory exists
    os.makedirs('DataFig', exist_ok=True)
    
    # 1. Time Series Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Passengers'])
    plt.title('Air Passengers over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Passengers')
    plt.savefig('DataFig/time_series_plot.png')
    plt.close()
    print("Saved time_series_plot.png")
    
    # 2. Seasonal Decomposition
    # Multiplicative because seasonality seems to increase with the trend
    decomposition = seasonal_decompose(df['Passengers'], model='multiplicative', period=12)
    
    fig = decomposition.plot()
    fig.set_size_inches(12, 10)
    plt.tight_layout()
    plt.savefig('DataFig/decomposition_plot.png')
    plt.close()
    print("Saved decomposition_plot.png")
    
    # 3. Monthly distribution (Boxplot)
    df['Month_num'] = df.index.month
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Month_num', y='Passengers', data=df)
    plt.title('Monthly Distribution of Passengers')
    plt.xlabel('Month')
    plt.ylabel('Passengers')
    plt.savefig('DataFig/monthly_boxplot.png')
    plt.close()
    print("Saved monthly_boxplot.png")

if __name__ == "__main__":
    perform_eda()
