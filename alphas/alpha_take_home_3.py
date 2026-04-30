import numpy as np
import pandas as pd
from alphas.base import BaseAlpha
import alphas.alpha_tools as tls

class Strategy(BaseAlpha):

    requirements = BaseAlpha.requirements + [
        "close",
        "oim_lvl1_v1"
    ]
    
    default_params = {
        "oim_smooth": 120,     # 輕微平滑
        "threshold": 0.1    # 門檻直接設為 0！
    }

    def prepare_features(self, df):
        # 步驟一：只做最簡單的平滑，連 Z-Score 都不算了！
        df = tls.add_ewm_sma(df, column='oim_lvl1_v1', window=self.params["oim_smooth"], out_name='oim_smoothed')
        return df

    def generate_target_position(self, row, account):
        # 直接拿平滑後的原始 OIM 數值
        oim_val = row.get('oim_smoothed', 0)
        
        # 濾除一開始在計算平滑時產生的 NaN
        if np.isnan(oim_val):
            return 0.0 

        threshold = self.params["threshold"]

        # ==========================================
        # 最原始的本能：買盤大於賣盤 (>0) 就做多，賣盤大於買盤 (<0) 就做空
        # ==========================================
        if oim_val < -threshold:
            return 0.99  
        elif oim_val > threshold:
            return -0.99 
        else:
            return None