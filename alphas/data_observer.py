import os
import sys
import pandas as pd
import importlib.util

# 確保可以引用專案根目錄的模組
sys.path.append(os.getcwd())
try:
    from backtesting.data_factory import BacktestDataFactory
except ImportError as e:
    print(f"[Error] 模組引用失敗: {e}")
    sys.exit(1)

def load_strategy_from_file(filepath):
    """ 動態載入策略檔案 (與 brain.py 邏輯相同) """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到檔案: {filepath}")
        
    module_name = os.path.basename(filepath).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, 'Strategy'):
        StrategyClass = getattr(module, 'Strategy')
        reqs = getattr(StrategyClass, 'requirements', [])
        return StrategyClass, reqs, module_name
    else:
        raise ValueError(f"[Error] 策略檔案 '{filepath}' 必須包含 'class Strategy(BaseAlpha):'")

def main():
    alphas_dir = "alphas"
    
    if not os.path.exists(alphas_dir):
        print(f"錯誤：找不到 '{alphas_dir}' 資料夾。")
        return

    # 1. 產生選單
    files = [f for f in os.listdir(alphas_dir) 
             if f.endswith(".py") 
             and f not in ["__init__.py", "brain.py", "alpha_tools.py", "base.py"]] 
    
    if not files:
        print(f"警告：找不到任何策略檔案。")
        return

    print("\n" + "="*40)
    print("   QUANT BRAIN 數據特徵觀察器")
    print("="*40)
    for i, f in enumerate(files):
        print(f"[{i+1}] {f}")
    print("-" * 40)

    selected_file = None
    while True:
        user_input = input("請輸入要觀察的策略編號 (或是按 q 離開): ").strip()
        if user_input.lower() == 'q': return
        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(files):
                selected_file = files[idx]
                break
            else:
                print("編號無效，請重新輸入。")
        except ValueError:
            print("請輸入有效的數字。")

    # ==========================================
    # 2. 設定擷取資料的時間範圍
    # (建議不要抓太長，否則 1 分鐘 K 線的 CSV 會非常龐大)
    # ==========================================
    start_date = "2025-03-29"
    end_date   = "2025-04-05" # 建議先抓一週或一個月的資料觀察即可

    strategy_path = os.path.join(alphas_dir, selected_file)
    print(f"\n>> 正在載入策略: {selected_file} ...")
    
    # 3. 載入策略與需求特徵
    try:
        StrategyClass, requirements, strategy_name = load_strategy_from_file(strategy_path)
    except Exception as e:
        print(f"[Error] {e}")
        return

    # 4. 透過 DataFactory 準備基礎資料
    print(f">> 正在從資料庫讀取基礎 K 線與 {requirements} ...")
    try:
        factory = BacktestDataFactory()
        df = factory.prepare_features(
            symbol="BTCUSDT", 
            interval="1m", 
            feature_ids=requirements,
            start_time=start_date,
            end_time=end_date
        )
        if df.empty:
            print("[Error] 在指定的時間範圍內找不到任何數據。")
            return
    except Exception as e:
        print(f"[Error] 基礎數據準備失敗: {e}")
        return

    # 5. 執行策略的 prepare_features
    print(f">> 正在執行策略專屬的特徵加工 (prepare_features) ...")
    strategy_instance = StrategyClass()
    processed_df = strategy_instance.prepare_features(df)

    # 6. 將結果輸出為 CSV
    output_filename = f"observe_{strategy_name}_features.csv"
    
    # 將 NaN 值明確輸出為 'NaN'，方便你在 Excel / 編輯器中搜尋
    processed_df.to_csv(output_filename, index=False, na_rep='NaN')
    
    print("\n" + "="*40)
    print(f"數據輸出完成！")
    print(f"檔案名稱: {output_filename}")
    print(f"資料筆數: {len(processed_df)} 筆")
    print(f"包含欄位: {', '.join(processed_df.columns.tolist())}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()