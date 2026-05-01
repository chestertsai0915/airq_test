import numpy as np
import pandas as pd
from base import BaseAlpha

class Strategy(BaseAlpha):
    requirements = []  # 因為特徵我們直接在 prepare_features 算，這裡可以留空
    
    default_params = {
        "z_window": 60,
        "z_threshold": 2.0
    }

    def prepare_features(self, df):
        # 取得參數
        window = self.params["z_window"]
        
        # 1 = GLD, 2 = SLV (對應 DataFactory 產生的欄位)
        df['ratio'] = df['close_1'] / df['close_2']
        
        # 計算 Z-Score
        df['mean'] = df['ratio'].rolling(window=window).mean()
        df['std'] = df['ratio'].rolling(window=window).std()
        df['z_score'] = (df['ratio'] - df['mean']) / df['std']
        
        return df

    def generate_target_position(self, row, account):
        z = row.get('z_score', np.nan)
        if pd.isna(z): 
            return None # 資料不足，保持不動

        threshold = self.params["z_threshold"]
        
        # 回傳字典設定目標權重
        if z >= threshold:
            # 賣出價差：做空黃金(-50%)，做多白銀(+50%)
            return {'YF_GLD': -0.5, 'YF_SLV': 0.5}
            
        elif z <= -threshold:
            # 買入價差：做多黃金(+50%)，做空白銀(-50%)
            return {'YF_GLD': 0.5, 'YF_SLV': -0.5}
            
        elif abs(z) <= 0.0:
            # 回歸均值：平倉 (權重設為 0)
            return {'YF_GLD': 0.0, 'YF_SLV': 0.0}

        return None # 訊號未觸發，保持現有倉位