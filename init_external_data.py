# init_external_data.py
from utils.database import DatabaseHandler
from utils.data_filler import DataGapFiller

# 載入您寫好的抓取器
from data_sources.funding_rate import FundingRateFetcher
from data_sources.funding_rate_bybit import BybitFundingRateFetcher

def run_fill_external_data(source, symbol, metric, interval, start_time, end_time, api_limit):
    print(f"=== 啟動外部數據補齊任務: {source.upper()} ({metric}) ===")
    
    db = DatabaseHandler()
    
    # 1. 根據來源選擇對應的抓取器與命名規則
    if source == 'binance':
        fetcher = FundingRateFetcher()
        target_db_symbol = symbol             # 存成 BTCUSDT
    elif source == 'bybit':
        fetcher = BybitFundingRateFetcher()
        target_db_symbol = f"BYBIT_{symbol}"  # 存成 BYBIT_BTCUSDT 避免污染
    else:
        print(f"錯誤：不支援的資料來源 '{source}'")
        return
        
    # 2. 建立 DataGapFiller，並傳入 metric 讓它知道要補齊 external_data
    filler = DataGapFiller(
        db_handler=db,
        fetch_func=fetcher.fetch_data,
        symbol=symbol,
        db_symbol=target_db_symbol,
        interval=interval,
        api_limit=api_limit,
        metric=metric  # 
    )

    # 3. 呼叫外部數據專用的補齊函數
    filler.check_and_fill_event_driven(start_date=start_time, end_date=end_time)
    print("=== 任務執行結束 ===\n")

if __name__ == "__main__":
    
    #抓取 Binance 的資金費率
    run_fill_external_data(
        source="binance",
        symbol="BTCUSDT",
        metric="funding_rate",
        interval="dynamic",
        start_time="2025-03-29 00:00:00" ,
        end_time="2026-03-29 00:00:00",
        api_limit=200             #  API 限制一次最多 200 以免
    )
    