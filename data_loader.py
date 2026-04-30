import pandas as pd
from binance.um_futures import UMFutures
from utils.database import DatabaseHandler
from pybit.unified_trading import HTTP as BybitHTTP  # 引入 Bybit V5 SDK
import yfinance as yf
class DataLoader:
    def __init__(self, client: UMFutures, bybit_client: BybitHTTP=None, db: DatabaseHandler = None):
        self.client = client
        self.bybit_client = bybit_client
        self.db_handler = db
        
        
    def get_binance_klines(self, symbol, interval, limit=1000, startTime=None, endTime=None):
        """ 
        幣安 K 線抓取，支援指定起迄時間
        """
        try:
            # 將額外參數打包
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            if startTime is not None:
                params["startTime"] = int(startTime)
            if endTime is not None:
                params["endTime"] = int(endTime)

            klines = self.client.klines(**params)
            
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'q_vol', 'trades', 'taker_buy_vol', 'taker_buy_q_vol', 'ignore'
            ])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            return df
        except Exception as e:
            print(f"幣安數據抓取失敗: {e}")
            return pd.DataFrame()
    def get_bybit_klines(self, symbol, interval, limit=100, startTime=None, endTime=None):
        
        bybit_interval = str(interval)
        if bybit_interval.endswith('m'):
            bybit_interval = bybit_interval[:-1]  
        elif bybit_interval == '1h':
            bybit_interval = '60'                 
        elif bybit_interval == '1d':
            bybit_interval = 'D'                  
            
        try:
            # 2. 整理參數
            params = {
                "category": "linear", 
                "symbol": symbol,
                "interval": bybit_interval, 
                "limit": limit
            }
            
            # Bybit 的時間參數名稱為 start 和 end
            if startTime is not None:
                params["start"] = int(startTime)
            if endTime is not None:
                params["end"] = int(endTime)

            # 3. 呼叫 Bybit V5 API
            response = self.bybit_client.get_kline(**params)
            
            # 加上錯誤檢查，如果 Bybit 報錯 (例如參數不對)，直接印出來方便除錯
            if response.get('retCode') != 0:
                print(f"Bybit API 拒絕請求: {response.get('retMsg')}")
                return pd.DataFrame()
            
            # Bybit 回傳的資料在 result['list'] 中
            klines = response.get('result', {}).get('list', [])
            
            if not klines:
                return pd.DataFrame()
            
            # 【關鍵】Bybit 的資料是「由新到舊」，必須反轉陣列以對齊幣安的「由舊到新」
            klines = klines[::-1]
            
            # Bybit 欄位定義: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            df['open_time'] = df['open_time'].astype(int) # 確保時間為整數毫秒
            
            return df
            
        except Exception as e:
            print(f"Bybit K線抓取發生系統錯誤: {e}")
            return pd.DataFrame()  

    def get_yfinance_gold(self, symbol="GLD", interval="1d", start_date=None, end_date=None):
        """ 
        Yahoo Finance 黃金 K 線抓取，支援指定起迄時間 (格式: 'YYYY-MM-DD')
        """
        try:
            # 將參數打包
            params = {
                "tickers": symbol,
                "interval": interval,
                "progress": False # 關閉下載進度條避免干擾終端機
            }
            if start_date is not None:
                params["start"] = start_date
            if end_date is not None:
                params["end"] = end_date
                
            # 若無指定時間，預設抓取全部歷史資料
            if start_date is None and end_date is None:
                params["period"] = "max"

            # 抓取數據
            df = yf.download(**params)
            
            if df.empty:
                return pd.DataFrame()

            # 將 Index (Date/Datetime) 轉為一般欄位
            df = df.reset_index()
            date_col = 'Date' if 'Date' in df.columns else 'Datetime'
            
            # 重新命名欄位以對齊 Binance 的格式
            df = df.rename(columns={
                date_col: 'open_time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # 保留需要的欄位並確保數值型態
            # 注意: yfinance 沒有 q_vol, trades 等深度數據，因此只保留基本 OHLCV
            keep_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
            df = df[keep_cols]
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            return df
            
        except Exception as e:
            print(f"Yahoo Finance 黃金數據抓取失敗: {e}")
            return pd.DataFrame()

    def get_yfinance_silver(self, symbol="SLV", interval="1d", start_date=None, end_date=None):
        """ 
        Yahoo Finance 白銀 K 線抓取，支援指定起迄時間 (格式: 'YYYY-MM-DD')
        """
        try:
            # 將參數打包
            params = {
                "tickers": symbol,
                "interval": interval,
                "progress": False
            }
            if start_date is not None:
                params["start"] = start_date
            if end_date is not None:
                params["end"] = end_date
                
            # 若無指定時間，預設抓取全部歷史資料
            if start_date is None and end_date is None:
                params["period"] = "max"

            # 抓取數據
            df = yf.download(**params)
            
            if df.empty:
                return pd.DataFrame()

            # 將 Index 轉為一般欄位
            df = df.reset_index()
            date_col = 'Date' if 'Date' in df.columns else 'Datetime'
            
            # 重新命名欄位以對齊 Binance 的格式
            df = df.rename(columns={
                date_col: 'open_time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # 保留需要的欄位並確保數值型態
            keep_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
            df = df[keep_cols]
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            return df
            
        except Exception as e:
            print(f"Yahoo Finance 白銀數據抓取失敗: {e}")
            return pd.DataFrame()  
    def get_yfinance_klines(self, symbol, interval, limit=1000, startTime=None, endTime=None):
        """ 
        Yahoo Finance K 線抓取，自動相容秒/毫秒/字串格式
        """
        try:
            params = {
                "tickers": symbol,
                "interval": interval,
                "progress": False
            }
            
            # --- 智慧型時間戳轉換模組 ---
            def parse_to_datetime(ts):
                if ts is None:
                    return None
                if isinstance(ts, str):
                    return pd.to_datetime(ts)
                # 數字長度超過 1e11 (11位數) 通常是毫秒，否則視為秒
                if ts > 1e11:
                    return pd.to_datetime(ts, unit='ms')
                else:
                    return pd.to_datetime(ts, unit='s')

            # 處理 startTime
            if startTime is not None:
                start_dt = parse_to_datetime(startTime)
                params["start"] = start_dt.strftime('%Y-%m-%d')
                
            # 處理 endTime
            if endTime is not None:
                end_dt = parse_to_datetime(endTime)
                # Yahoo Finance 的 end_date 是「不包含 (exclusive)」的，所以我們加 1 天確保抓完整
                end_dt = end_dt + pd.Timedelta(days=1)
                params["end"] = end_dt.strftime('%Y-%m-%d')

            # --- 執行抓取 ---
            df = yf.download(**params)
            
            if df.empty:
                return pd.DataFrame()

            # 處理最新版 yfinance 可能回傳 MultiIndex 欄位的狀況
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 將 Index (Date/Datetime) 轉為一般欄位
            df = df.reset_index()
            date_col = 'Date' if 'Date' in df.columns else 'Datetime'
            
            # 重新命名欄位以對齊資料庫架構
            df = df.rename(columns={
                date_col: 'open_time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # 保留需要的欄位並確保數值型態
            keep_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
            df = df[keep_cols]
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # --- 確保輸出回完美的毫秒整數 ---
            # 避免不同 pandas 版本 astype('int64') 的底層解析差異
            if pd.api.types.is_datetime64_any_dtype(df['open_time']):
                df['open_time'] = df['open_time'].dt.tz_localize(None)
                # 最安全的作法：直接相減 1970 年然後轉為毫秒，保證 100% 輸出 13 碼整數
                df['open_time'] = (df['open_time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')

            return df
            
        except Exception as e:
            print(f"Yahoo Finance 數據抓取失敗: {e}")
            return pd.DataFrame()
    
    # ==========================================
    #  改成從 DB 讀取的方法
    # ==========================================

    def get_google_trends_from_db(self, limit=1):
        """ 從 DB 讀取最新的 Google Trends """
        return self.db.load_external_data(
            symbol='GLOBAL', 
            metric='google_trends', 
            limit=limit
        )

    def get_fear_and_greed_from_db(self, limit=1):
        """ 從 DB 讀取恐慌指數 """
        return self.db.load_external_data(
            symbol='GLOBAL', 
            metric='fear_greed', 
            limit=limit
        )

    def get_macro_data_from_db(self, limit=1):
        """ 
        從 DB 讀取總經數據 
        因為 metric 有很多種，這裡可以一次讀出來
        """
        metrics = ['fed_assets', 'yield_10y', 'yield_2y']
        results = {}
        
        for m in metrics:
            df = self.db.load_external_data(symbol='US_MACRO', metric=m, limit=limit)
            if not df.empty:
                results[m] = df.iloc[-1]['value'] # 取最新一筆
            else:
                results[m] = 0
        return results

    def get_qqq_klines_from_db(self, limit=100):
        """ 從 market_data 表讀取 QQQ """
        return self.db.load_market_data(symbol='QQQ', interval='1d', limit=limit)

