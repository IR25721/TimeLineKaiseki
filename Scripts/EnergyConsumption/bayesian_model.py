import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import os

from sklearn.metrics import r2_score, mean_absolute_error

def evaluate_window(df, test_start_date, window_name, test_size=30, train_years=3):
    """
    指定された開始日から一定期間をテストデータとし、その直前を訓練データとして評価
    """
    test_start_date = pd.to_datetime(test_start_date)
    test_end_date = test_start_date + pd.Timedelta(days=test_size - 1)
    
    # テストデータ
    test_df = df[(df['ds'] >= test_start_date) & (df['ds'] <= test_end_date)].copy()
    
    # 訓練データ
    train_end_date = test_start_date - pd.Timedelta(days=1)
    train_start_date = train_end_date - pd.Timedelta(days=train_years*365)
    train_df = df[(df['ds'] >= train_start_date) & (df['ds'] <= train_end_date)].copy()
    
    if len(test_df) < test_size or len(train_df) < train_years*360:
        print(f"Skipping {window_name}: Insufficient data.")
        return None

    # Model Config (Conservative)
    model = Prophet(
        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
        changepoint_prior_scale=0.01, changepoint_range=0.8,
        seasonality_mode='additive', seasonality_prior_scale=0.05, holidays_prior_scale=0.05
    )
    model.add_seasonality(name='yearly', period=365.25, fourier_order=5) 
    model.add_seasonality(name='weekly', period=7, fourier_order=5)
    model.add_regressor('is_january', prior_scale=0.1)
    model.add_country_holidays(country_name='US')
    
    model.fit(train_df)
    
    future = model.make_future_dataframe(periods=test_size, freq='D')
    future['is_january'] = (future['ds'].dt.month == 1).astype(int)
    forecast = model.predict(future)
    
    # Evaluation
    forecast_test = forecast.tail(test_size)
    actual = test_df['y'].values
    pred = forecast_test['yhat'].values
    
    metrics = {
        'Window': window_name,
        'Test_Start': test_start_date.date(),
        'R2': r2_score(actual, pred),
        'MAPE': np.mean(np.abs((actual - pred) / actual)) * 100,
        'Bias': np.mean(actual - pred)
    }
    
    return metrics, forecast_test, train_df, test_df

def run_bayesian_modeling():
    results_dir = "Results/EnergyConsumption"
    fig_dir = "DataFig/EnergyConsumption/Modeling"
    os.makedirs(fig_dir, exist_ok=True)
    
    df = pd.read_csv(os.path.join(results_dir, "energy_features_daily.csv"))
    df['ds'] = pd.to_datetime(df['Datetime'])
    df['is_january'] = (df['ds'].dt.month == 1).astype(int)
    df['y'] = df['PJME_MW']
    
    windows = [
        ('Winter (Jan)', '2018-01-01'),
        ('Spring (Apr)', '2018-04-01'),
        ('Summer (Jul)', '2018-07-05')
    ]
    
    all_results = []
    
    for name, start_date in windows:
        print(f"\n--- Evaluating Window: {name} ---")
        res = evaluate_window(df, start_date, name)
        if res:
            metrics, forecast_test, train_df, test_df = res
            all_results.append(metrics)
            
            # 簡易可視化（最後のSummerだけ詳細保存するが、各履歴も見れるようにしても良い）
            plt.figure(figsize=(10, 4))
            train_plot = train_df.tail(120)
            plt.plot(train_plot['ds'], train_plot['y'], label='Train')
            plt.plot(test_df['ds'], test_df['y'], label='Actual', color='black')
            plt.plot(forecast_test['ds'], forecast_test['yhat'], label='Pred', color='red')
            plt.title(f"Window: {name}")
            plt.legend()
            save_path = os.path.join(fig_dir, f"prophet_window_{name.split()[0].lower()}.png")
            plt.savefig(save_path)
            plt.close()

    # Results Summary
    results_df = pd.DataFrame(all_results)
    print("\n" + "="*50)
    print("      MULTI-PERIOD VALIDATION SUMMARY")
    print("="*50)
    print(results_df.to_string(index=False))
    print("="*50)
    
    # Save Summary
    metrics_path = os.path.join(results_dir, "multi_period_validation.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"Summary saved to {metrics_path}")

if __name__ == "__main__":
    run_bayesian_modeling()
