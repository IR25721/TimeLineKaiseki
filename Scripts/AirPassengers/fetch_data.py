import pandas as pd
import statsmodels.api as sm
import os
import numpy as np

def fetch_air_passengers():
    print("Fetching Air Passengers dataset...")
    # Using statsmodels to get the dataset
    dataset = sm.datasets.get_rdataset("AirPassengers")
    data = dataset.data
    
    # Rename columns for clarity
    data.columns = ['time', 'Passengers']
    
    # Convert decimal year to datetime
    # 1949.0 -> 1949-01-01, 1949.0833 -> 1949-02-01
    years = data['time'].astype(int)
    months = np.round((data['time'] - years) * 12).astype(int) + 1
    data['Month'] = pd.to_datetime({'year': years, 'month': months, 'day': 1})
    
    # Keep only Month and Passengers
    data = data[['Month', 'Passengers']]
    
    # Ensure Datas directory exists
    os.makedirs('Datas', exist_ok=True)
    
    # Save to CSV
    output_path = 'Datas/air_passengers.csv'
    data.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(data.head())

if __name__ == "__main__":
    fetch_air_passengers()
