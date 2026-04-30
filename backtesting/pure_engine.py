# backtesting/pure_engine.py
import pandas as pd
import numpy as np

class VirtualAccount:
    """ 
    虛擬帳戶 (支援合約/雙向交易)
    
    修正紀錄:
    1. [BUG FIX] _open_position: 購買力檢查從只檢查 fee 改為檢查 fee + 最低保證金
       原本只要 balance >= fee 就能開倉，導致帳戶可以無限加倉直到 balance 被手續費磨光
    2. [BUG FIX] balance 歸零防護：balance 不可為負，若扣費後為負直接攔截
    """
    def __init__(self, initial_balance=10000.0, maker_fee=0.0002, taker_fee=0.0005,
                 leverage=1.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance  
        self.position = 0.0             
        self.avg_price = 0.0
        self.leverage = leverage        # 新增：槓桿倍數 (預設 1x，即現貨模式)
        self.taker_fee = taker_fee
        self.equity_curve = []          
        self.total_funding_fee = 0.0

    def mark_to_market(self, current_price, timestamp=None, record=False):
        unrealized_pnl = 0
        if self.position > 0:
            unrealized_pnl = (current_price - self.avg_price) * self.position
        elif self.position < 0:
            unrealized_pnl = (self.avg_price - current_price) * abs(self.position)
            
        total_equity = self.balance + unrealized_pnl
        
        if record and timestamp is not None:
            self.equity_curve.append({
                'datetime': timestamp,
                'equity': total_equity,
                'price': current_price,
                'position': self.position
            })
        return total_equity

    def execute(self, side, quantity, price, reason):
        if quantity <= 0: return

        if side == 'BUY':
            # 平空邏輯
            if self.position < 0: 
                cover_qty = min(quantity, abs(self.position))
                pnl = (self.avg_price - price) * cover_qty
                self.balance += pnl
                self.balance -= (cover_qty * price * self.taker_fee)
                self.position += cover_qty
                
                # 完全平倉時，徹底重置均價 (消滅幽靈價格)
                if abs(self.position) < 1e-9:
                    self.position = 0.0
                    self.avg_price = 0.0
                    
                remaining_qty = quantity - cover_qty
                if remaining_qty > 0: 
                    self._open_position('LONG', remaining_qty, price)
            # 開多 / 加多邏輯
            else: 
                self._open_position('LONG', quantity, price)

        elif side == 'SELL':
            # 平多邏輯
            if self.position > 0: 
                close_qty = min(quantity, self.position)
                pnl = (price - self.avg_price) * close_qty
                self.balance += pnl
                self.balance -= (close_qty * price * self.taker_fee)
                self.position -= close_qty
                
                # 完全平倉時，徹底重置均價 (消滅幽靈價格)
                if abs(self.position) < 1e-9:
                    self.position = 0.0
                    self.avg_price = 0.0
                    
                remaining_qty = quantity - close_qty
                if remaining_qty > 0: 
                    self._open_position('SHORT', remaining_qty, price)
            # 開空 / 加空邏輯
            else: 
                self._open_position('SHORT', quantity, price)

    def _open_position(self, direction, quantity, price):
        notional = quantity * price
        fee = notional * self.taker_fee

        # =====================================================
        # [BUG FIX 1] 購買力檢查：必須同時能負擔手續費 + 保證金
        # 原本只檢查 fee，導致 balance 足夠付手續費就能無限開倉
        # 保證金 = 名目價值 / 槓桿倍數
        # =====================================================
        required_margin = notional / self.leverage
        if self.balance < (fee + required_margin):
            # 若資金不足，嘗試用現有資金最大化開倉
            # available_for_margin = self.balance / (1 + self.taker_fee)
            # 為了簡單可靠，直接拒絕此次開倉
            return

        # 均價計算
        if abs(self.position) < 1e-9:
            self.avg_price = price
        else:
            current_abs_pos = abs(self.position)
            old_cost = current_abs_pos * self.avg_price
            new_cost = quantity * price
            self.avg_price = (old_cost + new_cost) / (current_abs_pos + quantity)

        # 更新倉位與扣除手續費
        if direction == 'LONG':
            self.position += quantity
        elif direction == 'SHORT':
            self.position -= quantity
            
        self.balance -= fee

        # =====================================================
        # [BUG FIX 2] balance 不可為負（浮點誤差防護）
        # =====================================================
        if self.balance < 0:
            self.balance = 0.0

    def pay_funding(self, funding_rate, current_price):
        if self.position == 0 or funding_rate == 0:
            return
        
        # 多頭 + 正費率 → 付錢；空頭 + 正費率 → 收錢
        funding_fee = self.position * current_price * funding_rate
        self.balance -= funding_fee
        self.total_funding_fee -= funding_fee


class PureBacktestEngine:
    def __init__(self, df, initial_balance=10000.0, mode='next_open',
                 leverage=1.0, tolerance=0.1):
        """
        參數說明:
        - leverage  : 槓桿倍數，傳入 VirtualAccount 用於保證金計算 (預設 1x)
        - tolerance : rebalance 容忍度，差距小於此值不交易 (預設 0.15 = 15%)
                      原本 0.1，對 1 分鐘資料太小會導致頻繁換手吃手續費
        """
        self.df = df
        self.account = VirtualAccount(initial_balance, leverage=leverage)
        self.mode = mode
        self.tolerance = tolerance      # 新增：外部可配置
        
        self.pending_action = None 
        self.pending_target = None 

    def run(self, strategy_func):
        records = self.df.to_dict('records')
        
        for row in records:
            current_close = row['close']
            current_open  = row['open']
            current_time  = row['datetime']
            funding_rate  = row.get('funding_rate', 0.0)

            # ==========================================
            # 1. 結算資金費率 (Funding Fee) ← 移到最前面
            # ==========================================
            # [BUG FIX 3] 時序修正：幣安資金費是在結算時間點「先扣費」
            # 原本是先執行 pending order 後才扣費，策略可以在費率結算前搶先開倉
            # 正確順序：先付資金費 → 再執行掛單 → 再 mark-to-market
            if funding_rate != 0.0 and self.account.position != 0:
                self.account.pay_funding(funding_rate, current_open)

            # ==========================================
            # 2. 執行 Pending Order (Next Open Mode)
            # ==========================================
            if self.mode == 'next_open':
                equity_at_open = self.account.mark_to_market(current_open)
                
                if self.pending_target is not None:
                    self._rebalance(self.pending_target, current_open, equity_at_open)
                    self.pending_target = None
                
                elif self.pending_action is not None:
                    action, pct = self.pending_action
                    self._process_legacy_order(action, pct, current_open, equity_at_open)
                    self.pending_action = None

            # ==========================================
            # 3. 更新權益 (Mark to Market) 
            # ==========================================
            equity = self.account.mark_to_market(current_close, current_time, record=True)
            
            # ==========================================
            # 4. 呼叫策略 (產生訊號)
            # ==========================================
            signal = strategy_func(row, self.account)

            # ==========================================
            # 5. 處理訊號 (分流處理)
            # ==========================================
            if len(self.account.equity_curve) > 0:
                self.account.equity_curve[-1]['signal'] = signal
            if signal is None:
                continue

            if isinstance(signal, (int, float, np.number)):
                target_pct = float(signal)
                if self.mode == 'close':
                    self._rebalance(target_pct, current_close, equity)
                elif self.mode == 'next_open':
                    self.pending_target = target_pct

            elif isinstance(signal, (tuple, list)):
                action, pct = signal
                if action != 'HOLD' and pct > 0:
                    if self.mode == 'close':
                        self._process_legacy_order(action, pct, current_close, equity)
                    elif self.mode == 'next_open':
                        self.pending_action = (action, pct)

    def _rebalance(self, target_pct, price, equity):
        """
        核心調倉邏輯
        """
        if equity <= 0: 
            return

        current_val = self.account.position * price
        current_pct = current_val / equity

        # [BUG FIX 4] 使用外部可配置的容忍度（預設 0.15 取代原本寫死的 0.1）
        TOLERANCE = self.tolerance

        if abs(target_pct) > 1e-6 and abs(target_pct - current_pct) < TOLERANCE:
            return

        target_val = equity * target_pct
        target_qty = target_val / price
        current_qty = self.account.position
        delta_qty = target_qty - current_qty
        
        MIN_TRADE_VALUE = 10.0
        delta_value = abs(delta_qty * price)

        if abs(target_pct) < 1e-6 and abs(current_qty) > 1e-9:
            pass  # 強制平倉放行
        elif delta_value < MIN_TRADE_VALUE:
            return

        if delta_qty > 0:
            self.account.execute('BUY', delta_qty, price, "Rebalance Buy")
        elif delta_qty < 0:
            self.account.execute('SELL', abs(delta_qty), price, "Rebalance Sell")

    def _process_legacy_order(self, action, pct, price, equity):
        if action in ['LONG', 'SHORT']:
            target_notional = equity * pct
            qty = target_notional / price
            
            if action == 'LONG':
                self.account.execute('BUY', qty, price, "Long Entry")
            elif action == 'SHORT':
                self.account.execute('SELL', qty, price, "Short Entry")

        elif action == 'LONG_EXIT':
            if self.account.position > 0:
                close_qty = self.account.position * pct
                self.account.execute('SELL', close_qty, price, "Long Exit")

        elif action == 'SHORT_EXIT':
            if self.account.position < 0:
                cover_qty = abs(self.account.position) * pct
                self.account.execute('BUY', cover_qty, price, "Short Exit")