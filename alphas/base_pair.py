"""
backtesting/base_pair.py

配對交易策略的基底類別。
繼承此類別後只需要實作 generate_target_positions() 即可。
"""
import numpy as np


class BasePairAlpha:
    """
    配對交易策略基底類別。

    子類別需覆寫：
        - requirements_1  : asset1 (GLD) 所需特徵 ID 列表
        - requirements_2  : asset2 (SLV) 所需特徵 ID 列表
        - default_params  : 預設參數字典
        - prepare_features(df1, df2) : 計算額外指標（回傳 df1, df2）
        - generate_target_positions(row1, row2, account1, account2)
            → 回傳 (pos1, pos2)，各為 -1.0 ~ 1.0
              或回傳 None 代表維持現有倉位不動
    """

    # 子類別各自宣告兩個資產需要的特徵
    requirements_1 = []   # asset1 的特徵 ID
    requirements_2 = []   # asset2 的特徵 ID

    default_params = {}

    def __init__(self, params=None):
        self.params = params if params is not None else self.default_params.copy()

    # ------------------------------------------------------------------
    # 子類別覆寫區
    # ------------------------------------------------------------------

    def prepare_features(self, df1, df2):
        """
        [可覆寫] 在這裡計算兩個資產共用的指標（spread、z-score 等）。
        回傳 (df1, df2)，各 DataFrame 可新增欄位。
        """
        return df1, df2

    def generate_target_positions(self, row1, row2, account1, account2):
        """
        [必須覆寫] 根據兩個資產當前 row 產生目標倉位。

        Parameters
        ----------
        row1    : dict  — asset1 當前 K 線 + 特徵
        row2    : dict  — asset2 當前 K 線 + 特徵
        account1: VirtualAccount — asset1 虛擬帳戶（可查 position、balance）
        account2: VirtualAccount — asset2 虛擬帳戶

        Returns
        -------
        (pos1, pos2) : tuple[float | None, float | None]
            各為 -1.0 ~ 1.0 的目標倉位比例。
            回傳 None 代表該資產本次不調倉。
        """
        raise NotImplementedError("請實作 generate_target_positions 方法")

    # ------------------------------------------------------------------
    # 引擎呼叫入口（不需覆寫）
    # ------------------------------------------------------------------

    def run(self, row1, row2, account1, account2, params=None):
        """
        PairBacktestEngine 每根 K 線呼叫此方法。
        回傳 (pos1, pos2) 或 (None, None)。
        """
        if params is not None:
            self.params = params

        try:
            result = self.generate_target_positions(row1, row2, account1, account2)

            # 允許回傳 None（整體不動）
            if result is None:
                return None, None

            pos1, pos2 = result

            pos1 = float(pos1) if pos1 is not None else None
            pos2 = float(pos2) if pos2 is not None else None
            return pos1, pos2

        except Exception as e:
            # 建議開發時解除下面這行，方便除錯
            # print(f"[PairStrategy Error] {e}")
            return 0.0, 0.0