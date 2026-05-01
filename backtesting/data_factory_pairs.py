import pandas as pd
import os
import sys

# 確保可以引用專案根目錄的模組
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.database import DatabaseHandler
from features.feature_store import FeatureStore
from managers.data_manager import DataBoard

class PairsDataFactory:
    """
    專為雙資產對齊設計的資料工廠，完美相容舊有的 FeatureStore。
    """
    def __init__(self, db_path="trading_data.db"):
        self.db = DatabaseHandler(db_path, skip_backup=True)
        self.feature_store = FeatureStore()
        
        self.cache_dir = os.path.join(current_dir, ".data_cache_pairs")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def load_and_align_data(self, sym1='YF_GLD', sym2='YF_SLV', feature_ids=None, start_date=None, end_date=None):
        print(f" [Factory] 準備配對數據 {sym1} & {sym2} ...")
        
        # 1. 讀取雙資產基礎 K 線
        df1 = self.db.load_market_data(sym1, '1d', limit=None)
        df2 = self.db.load_market_data(sym2, '1d', limit=None)
        
        if df1.empty or df2.empty:
            raise ValueError(f"[Error] 資料庫缺少 {sym1} 或 {sym2} 的 K 線數據，請先執行抓取")

        # 整理時間軸
        df1['datetime'] = pd.to_datetime(df1['open_time'], unit='ms')
        df2['datetime'] = pd.to_datetime(df2['open_time'], unit='ms')
        
        # 2. 呼叫 FeatureStore 計算指標 (掛載於 df1)
        if feature_ids:
            print(f" [Factory] 正在計算 {len(feature_ids)} 個技術特徵 (綁定於 {sym1})...")
            dummy_board = DataBoard(main_kline=df1, external_data={})
            try:
                features_df = self.feature_store.load_features(feature_ids, dummy_board)
                if not features_df.empty:
                    if 'open_time' in features_df.columns:
                        features_df['open_time'] = features_df['open_time'].astype('int64')
                    df1 = pd.merge(df1, features_df.drop(columns=['close', 'open', 'high', 'low', 'volume'], errors='ignore'), on='open_time', how='left')
                    df1 = df1.ffill().fillna(0)
            except Exception as e:
                print(f" [Error] FeatureStore 計算失敗: {e}")

        # 3. 完美對齊 (Inner Join)
        aligned_df = pd.merge(
            df1, 
            df2[['open_time', 'datetime', 'open', 'close']], 
            on=['open_time', 'datetime'], 
            how='inner', 
            suffixes=('_1', '_2')
        )

        # 時間篩選
        if start_date:
            aligned_df = aligned_df[aligned_df['datetime'] >= pd.to_datetime(start_date)]
        if end_date:
            aligned_df = aligned_df[aligned_df['datetime'] <= pd.to_datetime(end_date)]

        # 4. 拆回 df1 與 df2 (保持獨立但時間已完美對齊)
        cols_1 = [c for c in aligned_df.columns if not c.endswith('_2')]
        cols_2 = [c for c in aligned_df.columns if not c.endswith('_1')]

        final_df1 = aligned_df[cols_1].rename(columns=lambda x: x.replace('_1', ''))
        final_df2 = aligned_df[cols_2].rename(columns=lambda x: x.replace('_2', ''))

        print(f" [Factory] 雙資產對齊並拆分完成! 共 {len(final_df1)} 個交易日。")
        return final_df1, final_df2