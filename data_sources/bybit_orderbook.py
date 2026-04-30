import pandas as pd
from pybit.unified_trading import HTTP
import os

from data_sources.base_source import BaseDataSource

class Bybit_orderbookFetcher(BaseDataSource):
    name = "bybit_orderbook"
    
    def __init__(self): # 改成不需要外部傳入 client
        key = os.getenv('BYBIT_API_KEY')
        secret = os.getenv('BYBIT_SECRET_KEY')
        # 自己建立連線
        self.client = HTTP(testnet=False, api_key=key, api_secret=secret)

    def fetch_data(self, symbol, limit=5):
        try:
            # 【關鍵】Bybit U本位深度 limit 只能是 1, 50, 200, 500。這裡請求 50 檔。
            response = self.client.get_orderbook(
                category="linear",
                symbol=symbol,
                limit=50
            )
            
            result = response.get('result', {})
            # 利用切片 [:limit] 只取出前 5 檔
            bids = result.get('b', [])[:limit]
            asks = result.get('a', [])[:limit]
            timestamp = result.get('ts')
            
            df_depth = pd.DataFrame({
                'timestamp': int(timestamp),
                'level': range(1, len(bids) + 1),
                'bid_price': [float(b[0]) for b in bids],
                'bid_qty': [float(b[1]) for b in bids],
                'ask_price': [float(a[0]) for a in asks],
                'ask_qty': [float(a[1]) for a in asks]
            })
            
            return df_depth

        except Exception as e:
            print(f"Bybit 深度數據抓取失敗: {e}")
            return pd.DataFrame()    