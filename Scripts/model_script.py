import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

def model_selection():
    print("Loading preprocessed data...")
    # Load the preprocessed data: deeply stationary (Log + Diff + Seasonal Diff)
    df = pd.read_csv('Results/air_passengers_preprocessed.csv')
    ts_stationary = df['Log_Diff_Seasonal_Diff'].dropna()
    
    os.makedirs('DataFig', exist_ok=True)
    os.makedirs('Results', exist_ok=True)
    
    # 1. Plot ACF and PACF for Fully Differenced Data
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(ts_stationary, lags=40, ax=axes[0])
    axes[0].set_title('ACF - Fully Differenced Data (Log+Diff+Seasonal)')
    
    plot_pacf(ts_stationary, lags=40, ax=axes[1], method='ywm')
    axes[1].set_title('PACF - Fully Differenced Data (Log+Diff+Seasonal)')
    
    plt.tight_layout()
    acf_pacf_path = 'DataFig/acf_pacf_stationary.png'
    plt.savefig(acf_pacf_path)
    plt.close()
    print(f"Saved ACF/PACF plots to {acf_pacf_path}")
    
    # 2. Train an ARIMA Model (now effectively modeling SARIMA without the d and D terms because we did difference manually)
    # The user requested to set ar.L1 to 0, so we use an MA(1) model: ARIMA(0, 0, 1)
    print("\nTraining ARIMA(0, 0, 1) model on Fully Differenced data...")
    try:
        model = sm.tsa.ARIMA(ts_stationary, order=(0, 0, 1))
        results = model.fit()
        
        # Save summary to text file
        summary_path = 'Results/arima_model_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("--- ARIMA(0,0,1) Model Summary (Trained on Fully Differenced Data) ---\n\n")
            f.write(results.summary().as_text())
        print(f"Saved model summary to {summary_path}")
        
        # Print summary to console
        print(results.summary())
        
    except Exception as e:
        print(f"Error training model: {e}")

if __name__ == "__main__":
    model_selection()
