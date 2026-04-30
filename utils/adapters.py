# data_sources/adapters.py
import pandas as pd

class BinanceAdapter:
    def __init__(self, client):
        self.client = client
        self.name = "Binance"

    def get_limit(self): return 1000
    def get_delay(self): return 0.5
    
    def get_db_symbol(self, symbol): 
        return symbol # 幣安是主市場，維持原名

    def fetch_klines(self, symbol, interval, start_ts, end_ts):
        """ 統一標準的抓取介面 """
        try:
            klines = self.client.klines(
                symbol=symbol,
                interval=interval,
                limit=self.get_limit(),
                startTime=start_ts,
                endTime=end_ts
            )
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'q_vol', 'trades', 'taker_buy_vol', 'taker_buy_q_vol', 'ignore'
            ])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            return df
        except Exception as e:
            print(f"[{self.name} API 錯誤] {e}")
            return pd.DataFrame()


class BybitAdapter:
    def __init__(self, client):
        self.client = client
        self.name = "Bybit"

    def get_limit(self): return 1000
    def get_delay(self): return 0.5
    
    def get_db_symbol(self, symbol): 
        return f"BYBIT_{symbol}" # Bybit 自動加上前綴避免污染

    def _convert_interval(self, interval):
        # 封裝 Bybit 專屬的特例轉換
        if interval.endswith('m'): return interval[:-1]
        if interval == '1h': return '60'
        if interval == '1d': return 'D'
        return interval

    def fetch_klines(self, symbol, interval, start_ts, end_ts):
        """ 統一標準的抓取介面 """
        try:
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=self._convert_interval(interval), # 內部自動轉換
                start=start_ts,                            # Bybit 專屬的 start
                end=end_ts,                                # Bybit 專屬的 end
                limit=self.get_limit()
            )
            
            if response.get('retCode') != 0:
                print(f"[{self.name} API 錯誤] {response.get('retMsg')}")
                return pd.DataFrame()

            klines = response.get('result', {}).get('list', [])
            if not klines: return pd.DataFrame()

            klines = klines[::-1] # Bybit 由新到舊，需反轉對齊幣安
            
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            df['open_time'] = df['open_time'].astype(int)
            return df
            
        except Exception as e:
            print(f"[{self.name} 系統錯誤] {e}")
            return pd.DataFrame()