import itertools
import time
import optuna
import warnings
import numpy as np

# 這裡匯入專屬配對交易的研究環境 (請確保你有 research_pairs.py)
from research_pairs import PairsResearchEnvironment
from backtesting.data_factory_pairs import PairsDataFactory
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ==========================================
# 核心轉換器：將統一格式轉為 Grid Search 所需的列表
# ==========================================
def parse_space_for_grid(search_space):
    param_grid = {}
    for key, config in search_space.items():
        if config["type"] == "categorical":
            param_grid[key] = config["choices"]
        elif config["type"] == "float":
            low, high, step = config["low"], config["high"], config.get("step", 0.1)
            steps_count = int(round((high - low) / step)) + 1
            param_grid[key] = [round(low + i * step, 4) for i in range(steps_count)]
        elif config["type"] == "int":
            low, high, step = config["low"], config["high"], config.get("step", 1)
            param_grid[key] = list(range(low, high + 1, step))
    return param_grid

def run_grid_search(env, search_space):
    print("--- 啟動 Grid Search (網格搜尋) ---")
    param_grid = parse_space_for_grid(search_space)
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"總共需要測試 {len(combinations)} 組參數組合。")
    
    best_params = None
    best_score = -999.0
    
    for i, params in enumerate(combinations):
        metrics = env.evaluate(params)
        
        # 相容字典與浮點數回傳格式
        if isinstance(metrics, dict):
            score = metrics.get('sharpe', -999)
            ret = metrics.get('return', 0.0)
        else:
            score = metrics
            ret = 0.0
            
        print(f"[Grid {i+1}/{len(combinations)}] {str(params):<65} | Sharpe: {score:>6.2f} | Ret: {ret:>8.2%}")
        
        if score > best_score:
            best_score = score
            best_params = params
            
    return best_params, best_score

def run_optuna_search(env, search_space, n_trials=100):
    print(f"--- 啟動 Optuna 貝葉斯最佳化 (預計 {n_trials} 次) ---")
    sampler = optuna.samplers.TPESampler()

    def objective(trial):
        params = {}
        for key, config in search_space.items():
            if config["type"] == "categorical":
                params[key] = trial.suggest_categorical(key, config["choices"])
            elif config["type"] == "float":
                params[key] = trial.suggest_float(key, config["low"], config["high"], step=config.get("step"))
            elif config["type"] == "int":
                params[key] = trial.suggest_int(key, config["low"], config["high"], step=config.get("step", 1))

        metrics = env.evaluate(params)
        
        if isinstance(metrics, dict):
            score = metrics.get('sharpe', -999)
            ret = metrics.get('return', 0.0)
        else:
            score = metrics
            ret = 0.0

        print(f"[Trial {trial.number:03d}] {str(params):<65} | Sharpe: {score:>6.2f} | Ret: {ret:>8.2%}")
        return score

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def main():
    # 1. 綁定你的金銀比策略檔
    strategy_file = "alphas/alpha_GS_ratio.py" 
    
    # 2. 設定回測區間 (留一部分做 Out-of-Sample 驗證)
    start_date = "2014-01-01"   
    end_date   = "2026-03-29"   
    split_date = "2026-03-01"   

    # 3. 啟動配對研究環境
    env = PairsResearchEnvironment(
        strategy_file, 
        sym1="YF_GLD",   
        sym2="YF_SLV",      
        start_date=start_date,
        end_date=end_date,
        split_date=split_date
    )

    # ==========================================
    # 定義配對策略 (alpha_GS_ratio) 的搜尋空間
    # ==========================================
    search_space = {
        "ratio_window":  {"type": "int",   "low": 30,  "high": 90,  "step": 10},  # 均線長度 (30~90天)
        "entry_z":       {"type": "float", "low": 1.5, "high": 3.0, "step": 0.5}, # 進場門檻 (高一點比較安全)
        "exit_z":        {"type": "float", "low": 0.0, "high": 1.5, "step": 0.5}, # 出場門檻 (回歸到多少平倉)
        "max_hold_days": {"type": "int",   "low": 10,  "high": 40,  "step": 10}   # 強制時間平倉天數
    }

    # 切換模式："optuna" 或是 "grid"
    MODE = "optuna"  
    
    start_time = time.time()
    
    if MODE == "grid":
        best_params, best_score = run_grid_search(env, search_space)
    elif MODE == "optuna":
        best_params, best_score = run_optuna_search(env, search_space, n_trials=100)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"🥇 最佳參數組合: {best_params}")
    print(f"🥇 最佳 Sharpe Ratio (樣本內): {best_score:.4f}")
    print(f"⏱️ 最佳化耗時: {elapsed:.2f} 秒")
    print("="*70)

if __name__ == "__main__":
    main()