import pandas as pd
import os
import requests

def download_data():
    url = "https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv"
    data_dir = "Datas/EnergyConsumption"
    file_path = os.path.join(data_dir, "PJME_hourly.csv")
    
    os.makedirs(data_dir, exist_ok=True)
    
    if os.path.exists(file_path):
        print(f"File already exists at {file_path}")
        # Even if it exists, we can load a sample to show
        df = pd.read_csv(file_path)
        print("\n--- Data Sample ---")
        print(df.head())
        return
    
    print(f"Downloading data from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded to {file_path}")
        
        df = pd.read_csv(file_path)
        print("\n--- Data Sample ---")
        print(df.head())
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_data()
