# data_sources/funding_rate_bybit.py
from .base_source import BaseDataSource
import pandas as pd
import os
from pybit.unified_trading import HTTP

class BybitFundingRateFetcher(BaseDataSource):
    name = "funding_rate_bybit"
    
    def __init__(self):
        # 讀取 Bybit 的 API Key
        key = os.getenv('BYBIT_API_KEY')
        secret = os.getenv('BYBIT_SECRET_KEY')
        
        # 建立連線 (使用 Bybit V5 API)
        self.client = HTTP(
            testnet=False,
            api_key=key,
            api_secret=secret
        )

    def fetch_data(self, symbol="BTCUSDT", limit=200, startTime=None, endTime=None):
        """
        新增 startTime 與 endTime 支援，供補齊機制調用
        """
        try:
            # 1. 整理 Bybit API 需要的參數字典
            params = {
                "category": "linear",
                "symbol": symbol,
                "limit": limit
            }
            
            # Bybit V5 資金費率歷史的參數名稱也是 startTime 和 endTime
            if startTime is not None:
                params["startTime"] = int(startTime)
            if endTime is not None:
                params["endTime"] = int(endTime)

            # 2. 呼叫 API
            response = self.client.get_funding_rate_history(**params)
            
            # 加上 API 回傳錯誤的防呆檢查
            if response.get('retCode') != 0:
                print(f"[BybitFundingRate] API 拒絕請求: {response.get('retMsg')}")
                return pd.DataFrame()
            
            # 3. 轉成 DataFrame
            data_list = response.get('result', {}).get('list', [])
            if not data_list: 
                return pd.DataFrame()
                
            df = pd.DataFrame(data_list)
            
            # 4. 標準化欄位 (對齊 external_data 表格結構)
            result_df = pd.DataFrame()
            # 【關鍵】確保時間戳轉為整數格式，避免存入 DB 時型態錯誤
            result_df['open_time'] = df['fundingRateTimestamp'].astype('int64') 
            result_df['symbol'] = df['symbol']
            result_df['metric'] = 'funding_rate_bybit'
            # Bybit 回傳的 fundingRate 是字串格式，需要轉成 float
            result_df['value'] = df['fundingRate'].astype(float)
            
            return result_df
            
        except Exception as e:
            print(f"[BybitFundingRate] Error: {e}")
            return pd.DataFrame()