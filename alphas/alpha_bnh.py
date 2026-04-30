import numpy as np
import pandas as pd
from alphas.base import BaseAlpha

class Strategy(BaseAlpha):
    
    # 只需要最基本的 close，不計算任何複雜特徵
    requirements = BaseAlpha.requirements + ["close"]
    
    def prepare_features(self, df):
        # 什麼都不做，直接回傳原始資料
        return df

    def generate_target_position(self, row, account):
        # 使用一個內部屬性來記錄是不是第一筆
        if not hasattr(self, 'has_traded'):
            self.has_traded = True
            return 0.99  # 第一筆訊號：目標倉位 99%
        
        # 第二筆之後，全部回傳 None (不產生新訊號，維持現有倉位)
        return None