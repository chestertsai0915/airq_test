import numpy as np
import pandas as pd
from alphas.base_pair import BasePairAlpha


class Strategy(BasePairAlpha):
    """
    測試用策略：永遠持有 GLD +0.5、SLV -0.5。
    類比 alpha_bnh.py 的單資產版本，驗證雙帳戶引擎是否正常運作。
    """

    requirements_1 = []
    requirements_2 = []

    default_params = {}

    def prepare_features(self, df1, df2):
        return df1, df2

    def generate_target_positions(self, row1, row2, account1, account2):
        # 第一筆：建立倉位
        if not hasattr(self, '_entered'):
            self._entered = True
            return 0.5, 0.5

        # 之後維持不動
        return None, None