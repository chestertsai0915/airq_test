# alphas/visualize_feature.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# 確保能讀取到根目錄的模組 (backtesting, features 等)
sys.path.append(os.getcwd())

from backtesting.data_factory import BacktestDataFactory

def main():
    # ==========================================
    # 參數設定區 
    target_feature  = "vol_adj_mom_20_close_v1"  # 你要畫的特徵 ID
    target_symbol   = "BTCUSDT"                  # 交易對 (如果是 bybit 請改 "BYBIT_BTCUSDT")
    target_interval = "1m"                       # K線級別
    
    start_date      = None              # 起始時間 (例如 "2024-01-01"，若不限制請設為 None)
    end_date        = None              # 結束時間 (例如 "2024-02-01"，若不限制請設為 None)
    
    output_file     = "feature_plot.png"         # 畫出來的圖片要存成什麼檔名
    # ==========================================

    print(f"[*] 準備擷取特徵: {target_feature}")
    print(f"[*] 標的: {target_symbol} ({target_interval})")
    print(f"[*] 期間: {start_date} ~ {end_date}")

    try:
        factory = BacktestDataFactory()
        # 透過資料工廠準備基礎 K 線與指定的特徵
        df = factory.prepare_features(
            symbol=target_symbol,
            interval=target_interval,
            feature_ids=[target_feature],
            start_time=start_date,
            end_time=end_date
        )
        
        if df.empty:
            print("\n[Error] 找不到任何資料，請檢查時間區間或資料庫。")
            return
            
        print(f"[*] 成功載入 {len(df)} 筆資料，開始繪圖...")

        # 確保有適當的時間索引作為 X 軸
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'])
            df.set_index('open_time', inplace=True)
            
        # 開始使用 matplotlib 繪圖
        fig, ax1 = plt.subplots(figsize=(15, 7))

        # 若 DataFrame 中有 close 欄位，畫雙 Y 軸對比圖
        if 'close' in df.columns:
            color1 = 'tab:blue'
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Close Price', color=color1)
            # 畫收盤價 (左邊 Y 軸)
            ax1.plot(df.index, df['close'], color=color1, alpha=0.5, label='Close')
            ax1.tick_params(axis='y', labelcolor=color1)
            
            # 建立雙 Y 軸 (右邊 Y 軸給特徵用)
            ax2 = ax1.twinx()
            color2 = 'tab:red'
            ax2.set_ylabel(target_feature, color=color2)
            # 畫特徵值 (右邊 Y 軸)
            ax2.plot(df.index, df[target_feature], color=color2, linewidth=1.5, label=target_feature)
            ax2.tick_params(axis='y', labelcolor=color2)
            
            # 合併圖例
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
            
            plt.title(f"{target_symbol} {target_interval} - Price vs {target_feature}")
            
        else:
            # 如果沒有收盤價，只畫單一特徵
            plt.plot(df.index, df[target_feature], color='tab:red', label=target_feature)
            plt.title(f"Feature: {target_feature}")
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend(loc='upper left')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 儲存圖片
        plt.savefig(output_file, dpi=300)
        print(f"\n[+] 繪圖完成！圖片已儲存為: {output_file}")

    except Exception as e:
        print(f"\n[Error] 發生錯誤: {e}")

if __name__ == "__main__":
    main()