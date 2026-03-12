import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    data_path = "Datas/EnergyConsumption/PJME_hourly.csv"
    fig_dir = "DataFig/EnergyConsumption"
    results_dir = "Results/EnergyConsumption"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocessing for EDA
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    # 1. Overall Time Series Plot
    print("Plotting full time series...")
    plt.figure(figsize=(15, 6))
    df['PJME_MW'].plot(color='teal', linewidth=0.5)
    plt.title('PJM East (PJME) Hourly Energy Consumption (2002-2018)')
    plt.ylabel('MW')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, "full_timeseries.png"))
    plt.close()
    
    # 2. Extract time features for seasonal analysis
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # 3. Monthly Seasonality (Boxplot)
    print("Plotting monthly seasonality...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='month', y='PJME_MW', palette='viridis')
    plt.title('Energy Consumption by Month (Monthly Seasonality)')
    plt.ylabel('MW')
    plt.savefig(os.path.join(fig_dir, "monthly_boxplot.png"))
    plt.close()
    
    # 4. Hourly Seasonality (Boxplot)
    print("Plotting hourly seasonality...")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='hour', y='PJME_MW', palette='magma')
    plt.title('Energy Consumption by Hour (Daily Seasonality)')
    plt.ylabel('MW')
    plt.savefig(os.path.join(fig_dir, "hourly_boxplot.png"))
    plt.close()
    
    # 5. Daily Seasonality (Avg by Day of Week)
    print("Plotting daily seasonality...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='dayofweek', y='PJME_MW', palette='coolwarm')
    plt.title('Energy Consumption by Day of Week (Weekly Seasonality)')
    plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.ylabel('MW')
    plt.savefig(os.path.join(fig_dir, "dayofweek_boxplot.png"))
    plt.close()

    # Save summary stats
    summary = df['PJME_MW'].describe()
    with open(os.path.join(results_dir, "eda_stats.txt"), 'w') as f:
        f.write("--- PJME Hourly Energy Consumption Stats ---\n\n")
        f.write(summary.to_string())
        f.write(f"\n\nTotal rows: {len(df)}")
        f.write(f"\nDate range: {df.index.min()} to {df.index.max()}")
    
    print(f"EDA completed. Figures saved to {fig_dir}")

if __name__ == "__main__":
    run_eda()
