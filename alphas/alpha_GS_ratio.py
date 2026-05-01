import numpy as np
import pandas as pd
from alphas.base_pair import BasePairAlpha


class Strategy(BasePairAlpha):
    """
    金銀比 (Gold/Silver Ratio) 統計套利策略

    數據處理：
        Ratio_t = Price(Gold)_t / Price(Silver)_t

    訊號生成：
        rolling_mean = Ratio 的 60 日滾動平均
        rolling_std  = Ratio 的 60 日滾動標準差
        Z_t = (Ratio_t - rolling_mean) / rolling_std

    交易邏輯：
        Z > +2  → 金銀比偏高（黃金相對高估）→ 賣黃金、買白銀
        Z < -2  → 金銀比偏低（黃金相對低估）→ 買黃金、賣白銀
        |Z| < 0.5 或持倉超過 max_hold_days → 平倉
    """

    requirements_1 = ['close']   # GLD：只需要 close，data_factory 已內建
    requirements_2 = ['close']   # SLV：同上

    default_params = {
        "ratio_window":      60,    # 滾動均值/標準差計算窗口（交易日）
        "entry_z":           2.0,   # 進場 Z-score 閾值
        "exit_z":            0.0,   # 出場 Z-score 閾值（回歸中性）
        "max_hold_days":     30,    # 最長持倉天數（時間出場）
    }

    def __init__(self, params=None):
        super().__init__(params)
        self._hold_days = 0          # 當前持倉天數計數器
        self._in_position = False    # 是否持倉中

    # ------------------------------------------------------------------
    # prepare_features：預先計算整列的 Ratio 與 Z-score
    # ------------------------------------------------------------------

    def prepare_features(self, df1, df2):
        """
        df1 = GLD，df2 = SLV
        計算金銀比、滾動均值、滾動標準差、Z-score，
        全部寫入 df1（df2 帶一份副本，方便 row2 也能取用）
        """
        p  = self.params
        w  = int(p["ratio_window"])

        df1 = df1.copy()
        df2 = df2.copy()

        price_gld = df1['close'].replace(0, np.nan)
        price_slv = df2['close'].replace(0, np.nan)

        # 1. 金銀比
        ratio = price_gld / price_slv

        # 2. 滾動統計
        roll_mean = ratio.rolling(w, min_periods=w // 2).mean()
        roll_std  = ratio.rolling(w, min_periods=w // 2).std()

        # 3. Z-score（std 為 0 時設為 0，避免除零）
        zscore = ((ratio - roll_mean) / roll_std.replace(0, np.nan)).fillna(0)

        # 寫入兩個 df，讓 row1 / row2 都能取到
        for df in [df1, df2]:
            df['ratio']      = ratio.values
            df['ratio_mean'] = roll_mean.values
            df['ratio_std']  = roll_std.values
            df['zscore']     = zscore.values

        return df1, df2

    # ------------------------------------------------------------------
    # generate_target_positions：每根 K 線的交易決策
    # ------------------------------------------------------------------

    def generate_target_positions(self, row1, row2, account1, account2):
        p            = self.params
        entry_z      = p["entry_z"]
        exit_z       = p["exit_z"]
        max_hold     = int(p["max_hold_days"])
        size         = 0.5

        z = row1.get('zscore', 0.0)

        # Warmup 期：ratio_std 還沒穩定（std == 0 代表窗口內數據不足）
        if row1.get('ratio_std', 0.0) == 0.0:
            return None, None
        
        # ---- 持倉中：計算時間出場 ----
        if self._in_position:
            self._hold_days += 1
            # 時間出場 or Z-score 回歸中性
            if self._hold_days >= max_hold and abs(z) < exit_z:
                self._in_position = False
                self._hold_days   = 0
                return 0.0, 0.0    # 平倉

            # 持倉中且未到出場條件 → 維持不動
            return None, None
        
        # ---- 未持倉：尋找進場機會 ----

        # Z > +entry_z：金銀比偏高 → 黃金高估，賣 GLD 買 SLV
        if z >= entry_z:
            self._in_position = True
            self._hold_days   = 0
            return -size, +size

        # Z < -entry_z：金銀比偏低 → 黃金低估，買 GLD 賣 SLV
        if z <= -entry_z:
            self._in_position = True
            self._hold_days   = 0
            return +size, -size

        
        return None, None