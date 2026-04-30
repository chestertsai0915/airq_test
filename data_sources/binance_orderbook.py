from .base_source import BaseDataSource
import pandas as pd
import os
from binance.um_futures import UMFutures

class Binance_orderbookFetcher(BaseDataSource):
    name = "binance_orderbook"
    
    def __init__(self): # 改成不需要外部傳入 client
        key = os.getenv('BINANCE_API_KEY')
        secret = os.getenv('BINANCE_SECRET_KEY')
        # 自己建立連線
        self.client = UMFutures(key=key, secret=secret)

    def fetch_data(self, symbol, limit=5):
        try:
            depth_data = self.client.depth(symbol=symbol, limit=limit)
            
            bids = depth_data.get('bids', [])
            asks = depth_data.get('asks', [])
            
            # 取得撮合引擎時間戳 (毫秒)，方便後續與 K 線對齊
            timestamp = depth_data.get('T') 
            
            # 將資料轉為浮點數，並整理成 DataFrame
            df_depth = pd.DataFrame({
                'timestamp': timestamp,
                'level': range(1, len(bids) + 1),               # 檔位 1~5
                'bid_price': [float(b[0]) for b in bids],       # 買價
                'bid_qty': [float(b[1]) for b in bids],         # 買單量
                'ask_price': [float(a[0]) for a in asks],       # 賣價
                'ask_qty': [float(a[1]) for a in asks]          # 賣單量
            })
            
            return df_depth

        except Exception as e:
            print(f"幣安深度數據抓取失敗: {e}")
            return pd.DataFrame()    