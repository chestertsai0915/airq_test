import pandas as pd
import numpy as np
import sys
import os
import importlib.util

sys.path.append(os.getcwd())
try:
    from backtesting.pair_engine import PairBacktestEngine
    from backtesting.data_factory_pairs import PairsDataFactory
except ImportError as e:
    print(f"[Error] 模組引用失敗: {e}")

EVALUATION_CACHE = {}

class PairsResearchEnvironment:
    def __init__(self, strategy_file, sym1="YF_GLD", sym2="YF_SLV", start_date=None, end_date=None, split_date=None):
        self.sym1 = sym1
        self.sym2 = sym2
        
        self.strategy_class, self.requirements = self._load_strategy(strategy_file)
        
        train_end_date = split_date if split_date else end_date
        
        factory = PairsDataFactory()
        # 接收拆分好的兩個 df
        self.df1_is, self.df2_is = factory.load_and_align_data(
            sym1=self.sym1, 
            sym2=self.sym2,
            feature_ids=self.requirements,
            start_date=start_date,
            end_date=train_end_date
        )

    def _load_strategy(self, filepath):
        module_name = os.path.basename(filepath).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'Strategy'):
            StrategyClass = getattr(module, 'Strategy')
            reqs = getattr(StrategyClass, 'requirements', [])
            if not reqs and hasattr(StrategyClass, 'requirements_1'):
                reqs = getattr(StrategyClass, 'requirements_1', [])
            return StrategyClass, reqs
        else:
            raise ValueError("策略檔必須包含 'class Strategy'")

    def evaluate(self, params):
        param_key = tuple(sorted(params.items()))
        if param_key in EVALUATION_CACHE:
            return EVALUATION_CACHE[param_key]
        
        df1 = self.df1_is.copy()
        df2 = self.df2_is.copy()
        
        strategy_instance = self.strategy_class(params)
        
        # 完美支援你原有的 prepare_features(df1, df2) 寫法
        if hasattr(strategy_instance, 'prepare_features'):
            result = strategy_instance.prepare_features(df1, df2)
            if isinstance(result, tuple) and len(result) == 2:
                df1, df2 = result
            
        # 將 df1 與 df2 合併給 PairBacktestEngine
        merge_cols = ['datetime', 'open_time']
        renamed_df1 = df1.set_index(merge_cols).add_suffix('_1').reset_index()
        renamed_df2 = df2.set_index(merge_cols).add_suffix('_2').reset_index()
        df_aligned = pd.merge(renamed_df1, renamed_df2, on=merge_cols, how='inner')
        
        # 初始化回測引擎 (設定 leverage=3.0 以防等值對沖時保證金不足)
        engine = PairBacktestEngine(
            df_aligned=df_aligned, 
            initial_balance=10000.0, 
            leverage=3.0, 
            tolerance=0.05
        )
        
        engine.run(strategy_instance.generate_target_positions)
        res_df = engine.get_combined_equity_df()
        
        if res_df.empty:
            return {"sharpe": -999.0, "return": 0.0}
            
        pct = res_df['combined_equity'].pct_change().dropna()
        
        if len(pct) < 2 or pct.std() == 0: 
            result = {"sharpe": 0.0, "return": 0.0}
        else:
            sharpe = (pct.mean() / pct.std()) * np.sqrt(252) 
            total_return = (res_df['combined_equity'].iloc[-1] / 10000.0) - 1
            result = {"sharpe": sharpe, "return": total_return}
            
        EVALUATION_CACHE[param_key] = result
        return result