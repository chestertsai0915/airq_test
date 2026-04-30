import os
import subprocess
import sys

def main():
    alphas_dir = "alphas"
    brain_script = os.path.join(alphas_dir, "brain.py")
    
    if not os.path.exists(alphas_dir) or not os.path.exists(brain_script):
        print(f"錯誤：找不到 '{alphas_dir}' 或 '{brain_script}'。")
        return

    files = [f for f in os.listdir(alphas_dir) 
             if f.endswith(".py") 
             and f not in ["__init__.py", "brain.py"]] 
    
    if not files:
        print(f"警告：找不到任何策略檔案。")
        return

    print("\n" + "="*30)
    print("策略啟動器")
    print("="*30)
    for i, f in enumerate(files):
        print(f"[{i+1}] {f}")
    print("-" * 30)

    selected_file = None
    while True:
        user_input = input("請輸入策略編號 (或是按 q 離開): ").strip()
        if user_input.lower() == 'q': return
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
    # 在這裡寫死你要的回測時間 (若不限制請填 None 或 "")
    # ==========================================
    start_date = "2026-03-25"   # 例如: "2024-01-01"
    end_date   = "2026-03-29"   # 例如: "2025-01-01"
    split_date = "2026-03-29"        # 若設為 None，brain.py 會自動切在 70% 處

    strategy_path = os.path.join(alphas_dir, selected_file)
    print(f"\n>> 正在啟動策略回測: {selected_file} ...\n")
    
    # 動態組裝系統指令
    cmd = [sys.executable, brain_script, strategy_path]
    if start_date:
        cmd.extend(["--start", start_date])
    if end_date:
        cmd.extend(["--end", end_date])
    if split_date:
        cmd.extend(["--split", split_date])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] 執行過程中發生錯誤 (Exit Code: {e.returncode})")
    except KeyboardInterrupt:
        print("\n使用者中斷執行。")

if __name__ == "__main__":
    main()