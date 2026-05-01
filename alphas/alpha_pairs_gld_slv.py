"""
alphas/alpha_pairs_gld_slv.py

GLD / SLV 配對交易範例策略（統計套利）。

邏輯：
1. 計算 spread = log(GLD) - beta * log(SLV)
2. 對 spread 做 z-score 標準化
3. z-score > +threshold → 做空 spread（空 GLD、買 SLV）
   z-score < -threshold → 做多 spread（買 GLD、空 SLV）
   |z-score| < exit_threshold → 平倉
"""
import numpy as np
import pandas as pd
from alphas.base_pair import BasePairAlpha


class Strategy(BasePairAlpha):

    # 兩個資產各自需要的特徵（使用 data_factory 的 prepare_pairs_features，無需特別宣告）
    requirements_1 = []
    requirements_2 = []

    default_params = {
        "zscore_window":     60,    # z-score 滾動計算窗口（日數）
        "beta":              1.0,   # spread hedge ratio（可用 OLS 估計，先設 1）
        "entry_threshold":   2.0,   # 進場閾值
        "exit_threshold":    0.5,   # 出場閾值（回歸中值）
        "position_size":     0.8,   # 每個資產的目標倉位比例（0~1）
    }

    def __init__(self, params=None):
        super().__init__(params)
        self._spread_history = []  # 記錄 spread 序列

    def prepare_features(self, df1, df2):
        """
        預先計算整個時間序列的 spread z-score。
        這樣在 generate_target_positions 可以直接讀取，不用每根 K 線重算。
        """
        p = self.params
        w = p["zscore_window"]
        beta = p["beta"]

        # 計算 log spread
        df1 = df1.copy()
        df2 = df2.copy()
        log_gld = np.log(df1['close'].clip(lower=1e-9))
        log_slv = np.log(df2['close'].clip(lower=1e-9))
        spread = log_gld - beta * log_slv

        # 滾動 z-score
        roll_mean = spread.rolling(w).mean()
        roll_std  = spread.rolling(w).std()
        zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)

        df1['spread']   = spread.values
        df1['zscore']   = zscore.values
        df2['spread']   = spread.values
        df2['zscore']   = zscore.values

        return df1, df2

    def generate_target_positions(self, row1, row2, account1, account2):
        """
        根據當前 z-score 決定兩個資產的目標倉位。
        回傳 (pos_gld, pos_slv)。
        """
        z = row1.get('zscore', np.nan)
        p = self.params
        entry   = p["entry_threshold"]
        exit_th = p["exit_threshold"]
        size    = p["position_size"]

        # 資料不足（warmup 期）
        if np.isnan(z):
            return None, None

        # 做空 spread：GLD 高估 → 空 GLD、買 SLV
        if z > entry:
            return -size, +size

        # 做多 spread：GLD 低估 → 買 GLD、空 SLV
        if z < -entry:
            return +size, -size

        # 平倉：回歸均值
        if abs(z) < exit_th:
            if account1.position != 0 or account2.position != 0:
                return 0.0, 0.0

        # 其他情況不調倉
        return None, None