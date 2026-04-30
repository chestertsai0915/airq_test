# research.py
import pandas as pd
import numpy as np
import sys
import os
import importlib.util

sys.path.append(os.getcwd())
try:
    from backtesting.pure_engine import PureBacktestEngine
    from backtesting.data_factory import BacktestDataFactory
except ImportError:
    pass
EVALUATION_CACHE = {}
class ResearchEnvironment:
    # 1. 新增 start_date 與 end_date，並把預設 interval 改為 1m
    def __init__(self, strategy_file, symbol="BTCUSDT", interval="1m", start_date=None, end_date=None, split_date=None):
        
        # 2. 載入策略
        self.strategy_class, self.requirements = self._load_strategy(strategy_file)
        print(f"[Research] 載入策略完成，需求特徵: {self.requirements}")

        # 3. 決定最佳化(訓練)所用的資料範圍
        # 最佳化只能看樣本內 (IS) 的資料，所以資料的終止點應該是 split_date。若沒設 split_date 才用 end_date
        train_end_date = split_date if split_date else end_date

        print(f"[Research] 正在載入 IS (訓練) 數據 (範圍: {start_date} ~ {train_end_date})...")
        factory = BacktestDataFactory()
        
        full_df = factory.prepare_features(
            symbol, interval, 
            feature_ids=self.requirements, 
            start_time=start_date,
            end_time=train_end_date
        )
        
        self.df_is = full_df.reset_index(drop=True)
        print(f"[Research] 數據準備完成: {len(self.df_is)} 筆")

    def _load_strategy(self, filepath):
        if not os.path.exists(filepath): raise FileNotFoundError(filepath)
        module_name = os.path.basename(filepath).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'Strategy'):
            StrategyClass = getattr(module, 'Strategy')
            return StrategyClass, StrategyClass.requirements
        else:
            raise ValueError("策略檔必須包含 'class Strategy(BaseAlpha):'")

    def evaluate(self, params):
        param_key = tuple(sorted(params.items()))
        
        if param_key in EVALUATION_CACHE:
            # 直接回傳歷史結果，省下 100% 的運算時間！
            return EVALUATION_CACHE[param_key]
        
        df = self.df_is.copy()
        strategy_instance = self.strategy_class(params)
        df = strategy_instance.prepare_features(df)
            
        engine = PureBacktestEngine(df, initial_balance=10000, mode='next_open')
        engine.run(strategy_instance.run)
        
        equity_curve = engine.account.equity_curve
        if len(equity_curve) < 2: return -999.0

        equity = pd.Series([r['equity'] for r in equity_curve])
        pct = equity.pct_change().fillna(0)
        
        if pct.std() == 0: return 0.0
        
        sharpe = (pct.mean() / pct.std()) * np.sqrt(365 * 24) # 如果是 1m 資料，這裡未來可以改成 * np.sqrt(365 * 24 * 60)
        total_return = (equity.iloc[-1] / 10000) - 1
        
        result = {
            "sharpe": sharpe,
            "return": total_return
        }

        EVALUATION_CACHE[param_key] = result
        
        return result