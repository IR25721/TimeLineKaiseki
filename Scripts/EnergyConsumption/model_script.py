import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

def run_sarimax_modeling():
    # ディレクトリ準備
    results_dir = "Results/EnergyConsumption"
    fig_dir = "DataFig/EnergyConsumption/Modeling"
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. データの読み込みと基本設定
    input_path = os.path.join(results_dir, "transformed_5years_weekly.csv")
    df = pd.read_csv(input_path, index_col='Datetime', parse_dates=True)
    df = df.asfreq('D').ffill()
    
    # 2. 外部変数の再構築 (年間の動きを教える)
    # 周期性（年間の波: Fourier Series）
    day_of_year = df.index.dayofyear
    df['sin_year'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['cos_year'] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # 祝日フラグ (需要の急落を説明する)
    import holidays
    us_holidays = holidays.US()
    df['is_holiday'] = df.index.map(lambda x: 1 if x in us_holidays else 0)
    
    # 外部変数のまとめ
    exog_cols = ['sin_year', 'cos_year', 'is_holiday']
    exog = df[exog_cols].astype(float)
    
    # 3. ターゲット設定 (ln(yt))
    target_log = df['Log_MW']
    
    # 4. 分割
    test_size = 365
    train_log = target_log[:-test_size]
    test_log = target_log[-test_size:]
    train_exog = exog[:-test_size]
    test_exog = exog[-test_size:]
    
    # 5. モデル構築: 1次階差(d=1) + Fourier(年間) + SARIMA(s=7)
    # d=0では波形が平滑化されすぎたため、d=1に戻して「動き」を出しやすくします。
    # ただし、trend='c'（定数項）を入れると右肩上がりに爆発するため、
    # 定数項を抜き、年間の波は外部変数(Fourier)のみに任せる構成にします。
    print("Fitting model: d=1 (No Trend) + Fourier(Annual) + SARIMA(s=7)...")
    model = SARIMAX(train_log, 
                    exog=train_exog,
                    order=(1, 1, 1), 
                    seasonal_order=(1, 0, 1, 7),
                    trend='n', # 定数項を除去して右肩上がりを抑制
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    
    results = model.fit(disp=False)
    
    # 6. 予測と逆変換
    forecast = results.get_forecast(steps=test_size, exog=test_exog)
    forecast_mw = np.exp(forecast.predicted_mean)
    actual_mw = np.exp(test_log)
    train_mw = np.exp(train_log)
    
    # 7. 可視化
    plt.figure(figsize=(15, 7))
    plt.plot(train_mw[-730:].index, train_mw[-730:], label='Train (Actual)', color='tab:blue', alpha=0.6)
    plt.plot(actual_mw.index, actual_mw, label='Test (Actual)', color='gray', alpha=0.4)
    plt.plot(forecast_mw.index, forecast_mw, label='SARIMAX Forecast (d=1, No Trend)', color='red', linewidth=1.5)
    
    plt.title('Final Energy Consumption Forecast: d=1 without Drift (Fourier Seasonal)')
    plt.ylabel('PJME_MW')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(fig_dir, "sarimax_forecast_final_refined.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    run_sarimax_modeling()