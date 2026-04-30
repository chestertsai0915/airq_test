import numpy as np
import pandas as pd
from alphas.base import BaseAlpha
import alphas.alpha_tools as tls

class Strategy(BaseAlpha):

    
    # 1. 定義需求特徵
    requirements = BaseAlpha.requirements + [
        "close",
        "high",
        "low",
        "vol_adj_mom_20_close_v1",  # 基礎因子
        "custom_atr_14_v1"           # 用於波動率濾網
    ]
    
    default_params = {
        "mom_smooth_window": 75,    # 訊號平滑窗口
        "quantile_window": 500,      # 分位數計算窗口
        "zscore_window": 50    ,       # z-score 計算窗口
        "second_factor_window": 75     # 第二因子平滑窗口
    }

    def prepare_features(self, df):
    
        df = tls.add_zscore(df, column='vol_adj_mom_20_close_v1', window=self.params["zscore_window"], out_name='mom_z')
        
     
        
        df = tls.add_ewm_sma(df, column='mom_z', window=self.params["mom_smooth_window"], out_name='mom_smooth')
       
        
        
        df = tls.add_quantile(df, column='mom_smooth', window=self.params["quantile_window"], quantile=0.5, out_name='mom_q50')
        # 註：這裡建議在實作中增加一個 get_rolling_rank 的函數，將當前值轉為 0-1 的排名
        df = tls.add_ewm_sma(df, column='mom_q50', window=self.params["second_factor_window"], out_name='mom_q50smooth')
        
        return df

    def generate_target_position(self, row, account):
        mom_q50 = row.get('mom_q50smooth', 0)
        mom=row.get('mom_smooth', 0)




        return np.tanh(mom_q50)