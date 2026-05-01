"""
backtesting/pair_engine.py

雙資產配對交易回測引擎。
完全複用 pure_engine.py 的 VirtualAccount，
新增 PairBacktestEngine 同步驅動兩個帳戶。
"""
import pandas as pd
import numpy as np
import sys, os

# 確保可以引用專案根目錄
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from backtesting.pure_engine import VirtualAccount


class PairBacktestEngine:
    """
    雙資產配對交易引擎。


    兩個資產各有獨立的 VirtualAccount，資金完全隔離。
    初始資金各佔 initial_balance / 2（或由 balance_split 自訂）。
    每根 K 線同步處理兩個資產，時間戳以 df_aligned 的 datetime 欄位為準。
    使用與 PureBacktestEngine 完全相同的 next_open 執行模式。

    Parameters
    ----------
    df_aligned       : pd.DataFrame  — 已對齊的雙資產 K 線，需包含
                        datetime, open_1, close_1, open_2, close_2
                        (以及策略所需的其他特徵欄位)
    initial_balance  : float  — 兩個帳戶合計初始資金（各 50%）
    balance_split    : float  — asset1 佔比（0~1），預設 0.5
    leverage         : float  — 兩個帳戶統一槓桿（預設 1x）
    tolerance        : float  — rebalance 容忍度（預設 0.05）
    """

    def __init__(self, df_aligned, initial_balance=10000.0,
                 balance_split=0.5, leverage=1.0, tolerance=0.05):

        self.df = df_aligned.copy()
        self._validate_columns()

        bal1 = initial_balance * balance_split
        bal2 = initial_balance * (1 - balance_split)

        self.account1 = VirtualAccount(bal1, leverage=leverage)
        self.account2 = VirtualAccount(bal2, leverage=leverage)

        self.tolerance = tolerance
        self.initial_balance = initial_balance

        # next_open 掛單暫存
        self._pending1 = None
        self._pending2 = None

    # ------------------------------------------------------------------
    # 主回測迴圈
    # ------------------------------------------------------------------

    def run(self, strategy_func):
        """
        strategy_func(row1, row2, account1, account2) → (pos1, pos2)
        pos 為 -1.0 ~ 1.0，None 代表不調倉。
        """
        records = self.df.to_dict('records')

        for row in records:
            ts         = row['datetime']
            open1      = row['open_1']
            close1     = row['close_1']
            open2      = row['open_2']
            close2     = row['close_2']
            fr1        = row.get('funding_rate_1', 0.0)
            fr2        = row.get('funding_rate_2', 0.0)

            # ---- 1. 資金費率結算 ----
            if fr1 != 0.0 and self.account1.position != 0:
                self.account1.pay_funding(fr1, open1)
            if fr2 != 0.0 and self.account2.position != 0:
                self.account2.pay_funding(fr2, open2)

            # ---- 2. 執行 pending order（next_open 模式）----
            eq1 = self.account1.mark_to_market(open1)
            eq2 = self.account2.mark_to_market(open2)

            if self._pending1 is not None:
                self._rebalance(self.account1, self._pending1, open1, eq1)
                self._pending1 = None
            if self._pending2 is not None:
                self._rebalance(self.account2, self._pending2, open2, eq2)
                self._pending2 = None

            # ---- 3. Mark-to-market（收盤價）& 記錄 ----
            eq1 = self.account1.mark_to_market(close1, ts, record=True)
            eq2 = self.account2.mark_to_market(close2, ts, record=True)

            # 在 equity_curve 最後一筆記錄合計權益
            if self.account1.equity_curve and self.account2.equity_curve:
                combined = (self.account1.equity_curve[-1]['equity'] +
                            self.account2.equity_curve[-1]['equity'])
                self.account1.equity_curve[-1]['combined_equity'] = combined
                self.account2.equity_curve[-1]['combined_equity'] = combined

            # ---- 4. 呼叫策略取得訊號 ----
            # 把當前 row 拆成兩個 row（含各自特徵）
            row1 = self._extract_row(row, suffix='_1')
            row2 = self._extract_row(row, suffix='_2')

            sig1, sig2 = strategy_func(row1, row2, self.account1, self.account2)

            # 記錄訊號
            if self.account1.equity_curve:
                self.account1.equity_curve[-1]['signal'] = sig1
            if self.account2.equity_curve:
                self.account2.equity_curve[-1]['signal'] = sig2

            # ---- 5. 掛 pending order ----
            if sig1 is not None:
                self._pending1 = float(sig1)
            if sig2 is not None:
                self._pending2 = float(sig2)

    # ------------------------------------------------------------------
    # 結果匯整
    # ------------------------------------------------------------------

    def get_combined_equity_df(self):
        """
        回傳合計權益曲線 DataFrame。
        欄位：datetime, equity, price_1, price_2, position_1, position_2, signal_1, signal_2
        """
        if not self.account1.equity_curve or not self.account2.equity_curve:
            return pd.DataFrame()

        df1 = pd.DataFrame(self.account1.equity_curve).rename(columns={
            'equity': 'equity_1',
            'price': 'price_1',
            'position': 'position_1',
            'signal': 'signal_1'
        })
        df2 = pd.DataFrame(self.account2.equity_curve).rename(columns={
            'equity': 'equity_2',
            'price': 'price_2',
            'position': 'position_2',
            'signal': 'signal_2'
        })

        # 以 datetime 對齊
        df = pd.merge(
            df1[['datetime', 'equity_1', 'price_1', 'position_1', 'signal_1']],
            df2[['datetime', 'equity_2', 'price_2', 'position_2', 'signal_2']],
            on='datetime', how='inner'
        )
        df['equity'] = df['equity_1'] + df['equity_2']
        df['combined_equity'] = df['equity']
        return df.sort_values('datetime').reset_index(drop=True)

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------

    def _validate_columns(self):
        required = ['datetime', 'open_1', 'close_1', 'open_2', 'close_2']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"[PairEngine] df_aligned 缺少欄位: {missing}\n"
            )

    @staticmethod
    def _extract_row(row: dict, suffix: str) -> dict:
        """
        從合併 row 中抽出指定資產的欄位（去除後綴），
        並補上 datetime。
        """
        result = {'datetime': row.get('datetime')}
        n = len(suffix)
        for k, v in row.items():
            if k.endswith(suffix):
                result[k[:-n]] = v  # e.g. 'close_1' → 'close'
            elif not k.endswith('_1') and not k.endswith('_2'):
                # 共用欄位（如 datetime）直接帶入
                result[k] = v
        return result

    def _rebalance(self, account: VirtualAccount, target_pct: float,
                   price: float, equity: float):
        """
        與 PureBacktestEngine._rebalance 邏輯完全相同。
        """
        if equity <= 0:
            return

        current_val = account.position * price
        current_pct = current_val / equity

        if abs(target_pct) > 1e-6 and abs(target_pct - current_pct) < self.tolerance:
            return

        target_val = equity * target_pct
        target_qty = target_val / price
        current_qty = account.position
        delta_qty = target_qty - current_qty

        MIN_TRADE_VALUE = 10.0
        delta_value = abs(delta_qty * price)

        if abs(target_pct) < 1e-6 and abs(current_qty) > 1e-9:
            pass  # 強制平倉放行
        elif delta_value < MIN_TRADE_VALUE:
            return

        if delta_qty > 0:
            account.execute('BUY', delta_qty, price, "Pair Rebalance Buy")
        elif delta_qty < 0:
            account.execute('SELL', abs(delta_qty), price, "Pair Rebalance Sell")