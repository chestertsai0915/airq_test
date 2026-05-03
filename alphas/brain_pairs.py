import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import importlib.util
import sys
import os
import argparse
from scipy import stats

# 確保可以引用專案根目錄
sys.path.append(os.getcwd())

try:
    from backtesting.pair_engine import PairBacktestEngine
    from backtesting.data_factory import BacktestDataFactory
except ImportError as e:
    print(f"[Error] 模組引用失敗: {e}")
    sys.exit(1)


# ============================================================
# 1. 策略載入
# ============================================================

def load_pair_strategy(filepath):
    """
    載入配對交易策略檔案。
    策略檔必須包含繼承 BasePairAlpha 的 Strategy class。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到策略檔案: {filepath}")

    module_name = os.path.basename(filepath).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'Strategy'):
        raise ValueError(
            f"策略檔案 '{filepath}' 必須包含 'class Strategy(BasePairAlpha):'"
        )
    StrategyClass = getattr(module, 'Strategy')
    return StrategyClass, module_name


# ============================================================
# 2. 績效計算
# ============================================================

def calc_metrics(equity_series: pd.Series, label="") -> dict:
    """
    根據合計權益序列計算基本績效指標。
    equity_series：以 datetime 為 index 的每日（或每分鐘）權益序列。
    """
    if equity_series.empty or len(equity_series) < 2:
        return {}

    total_ret = equity_series.iloc[-1] / equity_series.iloc[0] - 1

    roll_max = equity_series.cummax()
    dd = (equity_series - roll_max) / roll_max
    max_dd = dd.min()

    # 日頻化計算 Sharpe
    daily = equity_series.resample('1D').last().dropna()
    daily_ret = daily.pct_change().dropna()
    std = daily_ret.std()
    sharpe = (daily_ret.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    return {
        "label":        label,
        "Total Return": total_ret,
        "Max Drawdown": max_dd,
        "Sharpe Ratio": sharpe,
        "Vol (Ann.)":   std * np.sqrt(252),
    }


# ============================================================
# 3. 報告輸出
# ============================================================

def save_report(result_df, metrics_is, metrics_os, split_date,
                strategy_name, funding_fee_1=0.0, funding_fee_2=0.0):
    filename = f"report_pairs_{strategy_name}.txt"
    has_os = metrics_os and metrics_os.get("Total Return") is not None

    with open(filename, "w", encoding="utf-8") as f:
        def w(text=""): f.write(text + "\n")

        w("=" * 60)
        w(f"{' 配對交易策略報告 ':^54}")
        w("=" * 60)
        w(f"策略代號  : {strategy_name}")
        w(f"樣本切割  : {split_date}")
        w(f"資金費率  : asset1={funding_fee_1:.4f}  asset2={funding_fee_2:.4f}")
        w("-" * 60)

        def fmt_pct(v):  return f"{v:>12.2%}"
        def fmt_f(v):    return f"{v:>12.4f}"

        w(f"\n{'Metric':<14} | {'IS (Train)':>12} | {'OS (Test)':>12}")
        w("-" * 44)

        def row(name, key, is_pct=False):
            vi = metrics_is.get(key, 0)
            vo = metrics_os.get(key, 0) if has_os else None
            vs_i = fmt_pct(vi) if is_pct else fmt_f(vi)
            vs_o = fmt_pct(vo) if (is_pct and vo is not None) else (fmt_f(vo) if vo is not None else "           -")
            w(f"{name:<14} | {vs_i} | {vs_o}")

        row("Return",   "Total Return", True)
        row("Sharpe",   "Sharpe Ratio")
        row("Max DD",   "Max Drawdown", True)
        row("Vol Ann.", "Vol (Ann.)")

        w("\n" + "=" * 60)
        w(f"{'各資產分開統計':^54}")
        w("=" * 60)
        # asset1 / asset2 個別統計
        for asset_key in ['equity_1', 'equity_2']:
            label = "GLD (asset1)" if asset_key == 'equity_1' else "SLV (asset2)"
            s = result_df.set_index('datetime')[asset_key]
            m = calc_metrics(s.resample('1D').last().ffill(), label=label)
            w(f"\n[{label}]")
            w(f"  Return : {m.get('Total Return', 0):.2%}")
            w(f"  Sharpe : {m.get('Sharpe Ratio', 0):.4f}")
            w(f"  Max DD : {m.get('Max Drawdown', 0):.2%}")
        w("=" * 60)

    print(f"[BRAIN_PAIRS] 報告已儲存至: {filename}")
    return filename


# ============================================================
# 4. 繪圖
# ============================================================

def plot_performance(result_df, split_date, strategy_name):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    ax1, ax2, ax3, ax4 = axes

    dt = pd.to_datetime(result_df['datetime'])
    split_dt = pd.to_datetime(split_date)

    # 合計權益
    ax1.plot(dt, result_df['equity'], label='Combined Equity', color='#1f77b4', linewidth=1.5)
    ax1.plot(dt, result_df['equity_1'], label='GLD Equity', color='#2ca02c', linewidth=0.8, alpha=0.7)
    ax1.plot(dt, result_df['equity_2'], label='SLV Equity', color='#ff7f0e', linewidth=0.8, alpha=0.7)
    ax1.axvline(split_dt, color='red', linestyle='--', alpha=0.6, label='Split')
    ax1.set_title(f"Pair Strategy: {strategy_name}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Equity")
    ax1.legend(loc='upper left', fontsize=9)

    # 倉位
    ax2.plot(dt, result_df['position_1'], label='GLD Position', color='#2ca02c', linewidth=0.8)
    ax2.plot(dt, result_df['position_2'], label='SLV Position', color='#ff7f0e', linewidth=0.8)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(split_dt, color='red', linestyle='--', alpha=0.3)
    ax2.set_ylabel("Position (qty)")
    ax2.legend(loc='upper left', fontsize=9)

   # 3. 滾動損益兩平勝率 (Break-Even Win Rate) vs 實際勝率
    # ==========================================
    eq_s = result_df.set_index('datetime')['equity']
    daily_eq = eq_s.resample('1D').last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    
    window = 60  # 60天的滾動視窗
    
    # 1. 計算滾動「實際勝率」(日報酬 > 0 的比例)
    roll_win_rate = (daily_ret > 0).rolling(window).mean()
    
    # 2. 計算滾動「平均獲利」與「平均虧損」
    # 將賺錢的日子挑出來算平均
    roll_avg_win = daily_ret.where(daily_ret > 0).rolling(window, min_periods=1).mean()
    roll_avg_loss = daily_ret.where(daily_ret < 0).abs().rolling(window, min_periods=1).mean()
    
    # 3. 計算滾動「損益兩平勝率 (BEW)」
    # 公式: BEW = Avg Loss / (Avg Win + Avg Loss)
    roll_bew = roll_avg_loss / (roll_avg_win + roll_avg_loss)
    # 防呆：如果期間內全勝、全敗或沒波動，將 NaN 填為 0
    roll_bew = roll_bew.fillna(0) 

    # 4. 繪圖
    ax3.plot(roll_win_rate.index, roll_win_rate, color='purple', linewidth=1.5, label='60D Actual Win Rate')
    ax3.plot(roll_bew.index, roll_bew, color='orange', linewidth=1.5, linestyle='-.', label='60D Break-Even Win Rate')
    
    # 5. 視覺化魔法：填滿超額期望值區域
    # 實際勝率 > BEW (綠色：期望值為正，賺錢中)
    ax3.fill_between(roll_win_rate.index, roll_bew, roll_win_rate, 
                     where=(roll_win_rate >= roll_bew), facecolor='green', alpha=0.3, interpolate=True)
    # 實際勝率 < BEW (紅色：期望值為負，虧錢中)
    ax3.fill_between(roll_win_rate.index, roll_bew, roll_win_rate, 
                     where=(roll_win_rate < roll_bew), facecolor='red', alpha=0.3, interpolate=True)

    ax3.axhline(0.5, color='black', linewidth=0.5, linestyle='--') # 50% 基準線
    ax3.axvline(split_dt, color='blue', linestyle='--', alpha=0.3)
    ax3.set_ylabel("Win Rate vs BEW")
    ax3.legend(loc='upper left', fontsize=9)
    
    # 將 Y 軸轉為百分比顯示

    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Drawdown（合計）
    eq_s = result_df.set_index('datetime')['equity']
    daily_eq = eq_s.resample('1D').last().dropna()
    roll_max = eq_s.cummax()
    dd = (eq_s - roll_max) / roll_max
    ax4.fill_between(dd.index, dd, 0, color='#d62728', alpha=0.3, label='Drawdown')
    ax4.set_ylabel("DD")
    ax4.legend(loc='lower left', fontsize=9)

   
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    out = f"report_pairs_{strategy_name}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[BRAIN_PAIRS] 圖表已儲存至: {out}")


# ============================================================
# 5. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="配對交易回測主程式")
    parser.add_argument('strategy_file', type=str, help="策略檔路徑")
    parser.add_argument('--start',  type=str, default=None, help="起始日 YYYY-MM-DD")
    parser.add_argument('--end',    type=str, default=None, help="結束日 YYYY-MM-DD")
    parser.add_argument('--split',  type=str, default=None, help="IS/OS 切分日 YYYY-MM-DD")
    parser.add_argument('--sym1',   type=str, default='YF_GLD', help="asset1 Symbol")
    parser.add_argument('--sym2',   type=str, default='YF_SLV', help="asset2 Symbol")
    parser.add_argument('--balance', type=float, default=10000.0, help="總初始資金")
    args = parser.parse_args()

    # ---- 1. 載入策略 ----
    try:
        StrategyClass, strategy_name = load_pair_strategy(args.strategy_file)
        print(f"[BRAIN_PAIRS] 成功載入策略: {strategy_name}")
    except Exception as e:
        print(f"[Error] {e}")
        return

    strategy_instance = StrategyClass()

    # ---- 2. 準備資料 ----
    print(f"[BRAIN_PAIRS] 載入配對資料: {args.sym1} & {args.sym2} ...")
    try:
        factory = BacktestDataFactory()
        df = factory.prepare_pairs_features(
            sym1=args.sym1,
            sym2=args.sym2,
            start_time=args.start,
            end_time=args.end
        )
        if df.empty:
            print("[Error] 指定時間範圍內無資料")
            return
        print(f"[BRAIN_PAIRS] 資料載入完成，共 {len(df)} 筆。"
              f" ({df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]})")
    except Exception as e:
        print(f"[Error] 資料準備失敗: {e}")
        import traceback; traceback.print_exc()
        return

    # ---- 3. 策略特徵加工 ----
    print("[BRAIN_PAIRS] 執行策略 prepare_features ...")

    # 把合併 df 拆成 df1 / df2 傳入策略
    col1 = [c for c in df.columns if c.endswith('_1') or c == 'datetime']
    col2 = [c for c in df.columns if c.endswith('_2') or c == 'datetime']
    df1 = df[col1].rename(columns=lambda x: x[:-2] if x.endswith('_1') else x)
    df2 = df[col2].rename(columns=lambda x: x[:-2] if x.endswith('_2') else x)

    df1, df2 = strategy_instance.prepare_features(df1, df2)

    # 把策略加工後的欄位重新合併回 df
    rename1 = {c: f"{c}_1" for c in df1.columns if c != 'datetime'}
    rename2 = {c: f"{c}_2" for c in df2.columns if c != 'datetime'}
    df = df[['datetime']].merge(df1.rename(columns=rename1), on='datetime', how='left') \
                         .merge(df2.rename(columns=rename2), on='datetime', how='left')
    df = df.ffill().fillna(0)

    # ---- 4. 切分 IS / OS ----
    if args.split:
        target_split = pd.to_datetime(args.split)
        if df['datetime'].max() < target_split:
            idx = int(len(df) * 0.7)
            SPLIT_DATE = df['datetime'].iloc[idx]
            print(f"[BRAIN_PAIRS] 資料不足，自動切在 70% 處: {SPLIT_DATE}")
        else:
            SPLIT_DATE = target_split
            print(f"[BRAIN_PAIRS] IS/OS 切分點: {SPLIT_DATE}")
    else:
        idx = int(len(df) * 0.7)
        SPLIT_DATE = df['datetime'].iloc[idx]
        print(f"[BRAIN_PAIRS] 未指定切分點，自動切在 70%: {SPLIT_DATE}")

    # ---- 5. 執行回測 ----
    print("[BRAIN_PAIRS] 開始回測 ...")
    engine = PairBacktestEngine(df, initial_balance=args.balance, tolerance=0.05)
    engine.run(strategy_instance.run)

    result_df = engine.get_combined_equity_df()
    if result_df.empty:
        print("[Error] 回測結果為空（可能完全沒有交易）")
        return

    # ---- 6. 輸出 CSV ----
    result_df.to_csv(f"backtest_pairs_{strategy_name}.csv", index=False, na_rep='None')
    print(f"[BRAIN_PAIRS] 回測明細已存: backtest_pairs_{strategy_name}.csv")

    # ---- 7. IS / OS 績效計算 ----
    result_df['datetime'] = pd.to_datetime(result_df['datetime'])
    result_df_idx = result_df.set_index('datetime')

    eq_full = result_df_idx['equity']

    is_eq = eq_full[eq_full.index < SPLIT_DATE]
    os_eq = eq_full[eq_full.index >= SPLIT_DATE]

    metrics_is = calc_metrics(is_eq, label="IS")
    metrics_os = calc_metrics(os_eq, label="OS")

    # ---- 8. 印出摘要 ----
    print("\n" + "=" * 50)
    print(f"{'Metric':<14} | {'IS':>10} | {'OS':>10}")
    print("-" * 40)
    for key, is_pct in [("Total Return", True), ("Sharpe Ratio", False),
                         ("Max Drawdown", True), ("Vol (Ann.)",   False)]:
        vi = metrics_is.get(key, 0)
        vo = metrics_os.get(key, 0)
        if is_pct:
            print(f"{key:<14} | {vi:>10.2%} | {vo:>10.2%}")
        else:
            print(f"{key:<14} | {vi:>10.4f} | {vo:>10.4f}")

    print("=" * 50)
    print(f"  Asset1 資金費用: {engine.account1.total_funding_fee:.4f}")
    print(f"  Asset2 資金費用: {engine.account2.total_funding_fee:.4f}")
    print("=" * 50)

    # ---- 9. 報告 & 圖表 ----
    save_report(result_df, metrics_is, metrics_os, SPLIT_DATE, strategy_name,
                engine.account1.total_funding_fee, engine.account2.total_funding_fee)
    plot_performance(result_df, SPLIT_DATE, strategy_name)

    print("[BRAIN_PAIRS] 回測流程全部完成！")


if __name__ == "__main__":
    main()