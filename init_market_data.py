import os
from binance.um_futures import UMFutures
from pybit.unified_trading import HTTP
from utils.database import DatabaseHandler
from utils.data_filler import DataGapFiller
from data_loader import DataLoader 

def run_fill_data(source, symbol, interval, start_time, end_time):
    print(f"=== 啟動資料補齊任務: {source.upper()} ===")
    
    db = DatabaseHandler()
    binance_key = os.getenv('BINANCE_API_KEY')
    binance_secret = os.getenv('BINANCE_SECRET_KEY')
    client = UMFutures(key=binance_key, secret=binance_secret) if binance_key else None
    
    bybit_key = os.getenv('BYBIT_API_KEY')
    bybit_secret = os.getenv('BYBIT_SECRET_KEY')
    bybit_client = HTTP(testnet=False, api_key=bybit_key, api_secret=bybit_secret) if bybit_key else None

    # 初始化 Loader
    loader = DataLoader(
        client=client, 
        bybit_client=bybit_client,
        db=db
    )

    # 根據來源切換抓取函數與存入的 Symbol 名稱
    if source == 'binance':
        target_fetch_func = loader.get_binance_klines
        target_db_symbol = symbol
        api_limit = 1000
    elif source == 'bybit':
        target_fetch_func = loader.get_bybit_klines
        target_db_symbol = f"BYBIT_{symbol}"
        api_limit = 1000
    elif source == 'yfinance':
        target_fetch_func = loader.get_yfinance_klines
        target_db_symbol = f"YF_{symbol}"           # 存進資料庫會變成 YF_GLD / YF_SLV
        api_limit = 1000                            # 日線每次抓 1000 天很夠用
    else:
        print("錯誤：不支援的交易所來源")
        return

    # 執行補齊
    filler = DataGapFiller(
        db_handler=db,
        fetch_func=target_fetch_func,
        symbol=symbol,
        db_symbol=target_db_symbol,
        interval=interval,
        api_limit=api_limit
    )

    filler.check_and_fill(start_date=start_time, end_date=end_time)
    print("=== 任務執行結束 ===")

if __name__ == "__main__":
    
    # 修改這裡來抓取 Yahoo Finance 的黃金 (GLD)
    TARGET_SOURCE = "yfinance"               
    TARGET_SYMBOL = "SLV"               # 若要抓白銀請改為 "SLV"
    TARGET_INTERVAL = "1d"              # yfinance 日線的參數格式為 '1d'
    START_TIME = "2014-01-01 00:00:00" 
    END_TIME   = "2026-04-30 00:00:00" 
    
    run_fill_data(
        source=TARGET_SOURCE,
        symbol=TARGET_SYMBOL,
        interval=TARGET_INTERVAL,
        start_time=START_TIME,
        end_time=END_TIME
    )