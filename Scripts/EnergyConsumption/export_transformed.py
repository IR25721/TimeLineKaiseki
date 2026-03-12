import pandas as pd
import numpy as np
import os

def export_transformed_subset():
    results_dir = "Results/EnergyConsumption"
    raw_path = "Datas/EnergyConsumption/PJME_hourly.csv"
    
    print(f"Loading data from {raw_path}...")
    df = pd.read_csv(raw_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    # Resample to Daily
    df_daily = df['PJME_MW'].resample('D').mean().to_frame()
    
    # Apply Transformations (v2 flow)
    df_daily['Log_MW'] = np.log(df_daily['PJME_MW'])
    df_daily['Log_Diff'] = df_daily['Log_MW'].diff()
    df_daily['Log_Diff_Weekly'] = df_daily['Log_Diff'].diff(7)
    
    # Select a subset (approx. 5 years = 1825 days)
    subset = df_daily.tail(1825).copy()
    
    output_path = os.path.join(results_dir, "transformed_5years_weekly.csv")
    subset.to_csv(output_path)
    print(f"Saved transformed subset to {output_path}")
    print("\n--- Sample Output (Last 5 days) ---")
    print(subset.tail())

if __name__ == "__main__":
    export_transformed_subset()
