"""
run_pairs.py

配對交易策略啟動器。
使用方式：python run_pairs.py
"""
import os
import subprocess
import sys


def main():
    alphas_dir  = "alphas"
    brain_script = os.path.join(alphas_dir, "brain_pairs.py")

    if not os.path.exists(alphas_dir) or not os.path.exists(brain_script):
        print(f"錯誤：找不到 '{alphas_dir}' 或 '{brain_script}'。")
        return

    # 只列出以 "pairs" 或 "pair" 命名的策略（可視需求改成列全部）
    files = sorted([
        f for f in os.listdir(alphas_dir)
        if f.endswith(".py")
        and f not in ["__init__.py", "brain.py", "brain_pairs.py", "base.py", "base_pair.py"]
    ])

    if not files:
        print("警告：找不到任何策略檔案。")
        return

    print("\n" + "=" * 35)
    print("配對交易策略啟動器")
    print("=" * 35)
    for i, f in enumerate(files):
        print(f"[{i+1}] {f}")
    print("-" * 35)

    selected_file = None
    while True:
        user_input = input("請輸入策略編號 (或 q 離開): ").strip()
        if user_input.lower() == 'q':
            return
        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(files):
                selected_file = files[idx]
                break
            else:
                print("編號無效，請重新輸入。")
        except ValueError:
            print("請輸入有效的數字。")

    # ==========================================
    # 在這裡設定回測參數
    # ==========================================
    start_date = "2014-01-01"   # 起始日
    end_date   = "2026-04-01"   # 結束日
    split_date = "2022-01-01"   # IS/OS 切分點
    sym1       = "YF_GLD"       # asset1
    sym2       = "YF_SLV"       # asset2
    balance    = "10000"        # 總初始資金（兩個帳戶合計）
    # ==========================================

    strategy_path = os.path.join(alphas_dir, selected_file)
    print(f"\n>> 正在啟動配對回測: {selected_file} ...\n")

    cmd = [
        sys.executable, brain_script, strategy_path,
        "--sym1",  sym1,
        "--sym2",  sym2,
        "--balance", balance,
    ]
    if start_date: cmd.extend(["--start", start_date])
    if end_date:   cmd.extend(["--end",   end_date])
    if split_date: cmd.extend(["--split", split_date])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] 執行錯誤 (Exit Code: {e.returncode})")
    except KeyboardInterrupt:
        print("\n使用者中斷。")


if __name__ == "__main__":
    main()