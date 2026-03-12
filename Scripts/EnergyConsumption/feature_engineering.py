import pandas as pd
import numpy as np
import holidays
import os

def create_features():
    results_dir = "Results/EnergyConsumption"
    input_path = os.path.join(results_dir, "energy_stationary_daily.csv")
    
    if not os.path.exists(input_path):
        # Fallback if the previous step named it differently or it's missing
        input_path = os.path.join(results_dir, "energy_stationary.csv")

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Handling different possible column names for index
    if 'Unnamed: 0' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Unnamed: 0'])
    elif 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    else:
        # Assume first column is date
        df['Datetime'] = pd.to_datetime(df.iloc[:, 0])
        
    df = df.set_index('Datetime').sort_index()
    
    # 1. Calendar Features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 2. Holiday Flags (US)
    us_holidays = holidays.US()
    df['is_holiday'] = df.index.map(lambda x: 1 if x in us_holidays else 0)
    
    # 3. Lag Features (on PJME_MW)
    # Using original values for lags as they are intuitive "exogenous" predictors
    df['lag_1'] = df['PJME_MW'].shift(1)
    df['lag_7'] = df['PJME_MW'].shift(7)
    df['lag_30'] = df['PJME_MW'].shift(30)
    
    # 4. Rolling Statistics
    df['roll_mean_7'] = df['PJME_MW'].rolling(window=7).mean()
    df['roll_std_7'] = df['PJME_MW'].rolling(window=7).std()
    
    # 5. Target for Modeling (Optional: can use Log_MW or PJME_MW)
    # We'll keep both for flexibility
    
    # Drop rows with NaN from shifts/rolling
    df_clean = df.dropna().copy()
    
    # Save
    output_path = os.path.join(results_dir, "energy_features_daily.csv")
    df_clean.to_csv(output_path)
    print(f"Saved featured data to {output_path}")
    
    # Feature Importance (Simple correlation)
    corr_cols = ['PJME_MW', 'day_of_week', 'month', 'is_weekend', 'is_holiday', 'lag_1', 'lag_7', 'roll_mean_7']
    correlation = df_clean[corr_cols].corr()
    
    corr_summary_path = os.path.join(results_dir, "feature_correlation.txt")
    with open(corr_summary_path, 'w') as f:
        f.write("--- Feature Correlation with Target (PJME_MW) ---\n")
        f.write(correlation['PJME_MW'].sort_values(ascending=False).to_string())
    
    print("\n--- Correlation Top Results ---")
    print(correlation['PJME_MW'].sort_values(ascending=False))

if __name__ == "__main__":
    create_features()
