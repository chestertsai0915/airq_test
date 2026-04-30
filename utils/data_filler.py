# utils/data_filler.py
import pandas as pd
import time

class DataGapFiller:
    def __init__(self, db_handler, fetch_func, symbol="BTCUSDT", db_symbol=None, interval="1m", api_limit=1000, metric=None):
        self.db = db_handler
        self.fetch_func = fetch_func
        self.api_symbol = symbol                     # 餵給 API 的名稱 (如 BTCUSDT)
        self.db_symbol = db_symbol or symbol         # 存進 DB 的名稱 (如 BYBIT_BTCUSDT)
        self.interval = str(interval)
        self.api_limit = api_limit
        self.metric = metric                         # 專門給 external_data 用的特徵名稱 (如 funding_rate)

    def _get_interval_ms(self):
        unit = self.interval[-1]
        value = int(self.interval[:-1])
        multiplier = {'m': 60 * 1000, 'h': 60 * 60 * 1000, 'd': 24 * 60 * 60 * 1000}
        return value * multiplier[unit]

    def _get_pandas_freq(self):
        unit = self.interval[-1]
        value = int(self.interval[:-1])
        
        # 智慧判斷：如果是來自 Yahoo Finance 的日線數據，使用 'B' (Business Day)
        # 這樣就不會把星期六、星期日列為「預期應該要有的資料」，避免無限重抓
        if unit == 'd' and 'YF_' in self.db_symbol:
            return f"{value}B"
            
        mapping = {'m': 'min', 'h': 'h', 'd': 'D'}
        return f"{value}{mapping[unit]}"

    # 1. 專門給 Market Data (K線) 用的補齊
   
    def check_and_fill(self, start_date, end_date):
        print(f"開始檢查 K線數據 {self.db_symbol} ({self.interval}) 從 {start_date} 到 {end_date} ...")

        
        freq = self._get_pandas_freq()
        expected_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # 防彈版毫秒轉換：直接與 1970 基準相減，無懼 Pandas 版本差異
        expected_ts = set((expected_dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms'))

        existing_ts_list = self.db.get_existing_timestamps(
            symbol=self.db_symbol, 
            interval=self.interval, 
            start_ts=min(expected_ts), 
            end_ts=max(expected_ts)
        )
        existing_ts = set(existing_ts_list if existing_ts_list else [])
        missing_ts = sorted(list(expected_ts - existing_ts))
        
        if not missing_ts:
            print("資料庫資料完整，無缺失區塊！\n")
            return

        print(f"發現 {len(missing_ts)} 筆缺失資料，開始分批下載...")

        interval_ms = self._get_interval_ms()
        
        # 智慧判斷容許的斷層間隔：
        # 如果是傳統金融 (YF_ 開頭的日線)，容許最多 5 天的週末或國定長假間隔
        # 如果是加密貨幣，則保持嚴格的 1 個 interval 單位
        if 'YF_' in self.db_symbol and 'd' in self.interval:
            allowed_gap = interval_ms * 5
        else:
            allowed_gap = interval_ms

        chunks = []
        current_chunk = [missing_ts[0]]

        for ts in missing_ts[1:]:
            # 將原本嚴格的 ts == current_chunk[-1] + interval_ms 
            # 改為較寬鬆的 <= allowed_gap
            if (ts - current_chunk[-1] <= allowed_gap) and len(current_chunk) < self.api_limit:
                current_chunk.append(ts)
            else:
                chunks.append((current_chunk[0], current_chunk[-1]))
                current_chunk = [ts]
                
        if current_chunk:
            chunks.append((current_chunk[0], current_chunk[-1]))

        for start_ts, end_ts in chunks:
            print(f"下載缺失區塊: {pd.to_datetime(start_ts, unit='ms')} -> {pd.to_datetime(end_ts, unit='ms')}")
            
            df_new = self.fetch_func(
                symbol=self.api_symbol, 
                interval=self.interval, 
                limit=self.api_limit,
                startTime=start_ts,
                endTime=end_ts
            )

            if not df_new.empty:
                self.db.save_market_data(self.db_symbol, self.interval, df_new)
                print(f"  └─ 成功寫入 {len(df_new)} 筆資料到 {self.db_symbol}")
            else:
                print("  抓取回傳為空")
            
            time.sleep(0.5) 
            
        print("所有缺失資料補齊完畢！\n")

   
    # 2. 專門給 External Data (如資金費率) 用的補齊
    
    def check_and_fill_external(self, start_date, end_date):
        if not self.metric:
            print("錯誤：執行 check_and_fill_external 必須在初始化時提供 metric 參數！")
            return

        print(f"開始檢查外部數據 {self.db_symbol} (特徵: {self.metric}, 頻率: {self.interval}) 從 {start_date} 到 {end_date} ...")

        freq = self._get_pandas_freq()
        expected_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        expected_ts = set((expected_dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms'))

        interval_ms = self._get_interval_ms()
        try:
            cursor = self.db.conn.cursor()
            cursor.execute('''
                SELECT timestamp FROM external_data 
                WHERE symbol = ? AND metric = ? AND timestamp >= ? AND timestamp <= ?
            ''', (self.db_symbol, self.metric, min(expected_ts) - interval_ms, max(expected_ts) + interval_ms))
            
            raw_existing_ts = set(row[0] for row in cursor.fetchall())
            
            existing_ts = set(round(ts / interval_ms) * interval_ms for ts in raw_existing_ts)
            
        except Exception as e:
            print(f"查詢 external_data 發生錯誤: {e}")
            existing_ts = set()

        missing_ts = sorted(list(expected_ts - existing_ts))
        
        if not missing_ts:
            print(f"[{self.metric}] 資料庫資料完整，無缺失區塊！\n")
            return

        print(f"發現 {len(missing_ts)} 筆缺失資料，開始分批下載...")

        chunks = []
        current_chunk = [missing_ts[0]]

        for ts in missing_ts[1:]:
            if ts == current_chunk[-1] + interval_ms and len(current_chunk) < self.api_limit:
                current_chunk.append(ts)
            else:
                chunks.append((current_chunk[0], current_chunk[-1]))
                current_chunk = [ts]
        if current_chunk:
            chunks.append((current_chunk[0], current_chunk[-1]))

        for start_ts, end_ts in chunks:
            window_end_ts = end_ts + interval_ms - 1
            
            print(f"下載缺失區塊: {pd.to_datetime(start_ts, unit='ms')} -> {pd.to_datetime(window_end_ts, unit='ms')}")
            
            df_new = self.fetch_func(
                symbol=self.api_symbol, 
                limit=self.api_limit,
                startTime=start_ts,
                endTime=window_end_ts
            )

            if not df_new.empty:
                df_new['symbol'] = self.db_symbol 
                self.db.save_generic_external_data(df_new)
                print(f"  └─ 成功寫入 {len(df_new)} 筆資料到 external_data ({self.metric})")
            else:
                print("  抓取回傳為空")
            
            time.sleep(0.5) 
            
        print(f"[{self.metric}] 所有缺失資料補齊完畢！\n")
    def check_and_fill_event_driven(self, start_date, end_date):
        """
        專為「可能改變結算頻率」的外部數據 (如資金費率) 設計的游標分頁抓取法
        """
        if not self.metric:
            print("錯誤：執行此方法必須提供 metric 參數！")
            return

        print(f"開始以游標模式(Cursor-based)補齊 {self.db_symbol} ({self.metric}) 從 {start_date} 到 {end_date} ...")

        # 將起訖時間轉為毫秒整數
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

        # 先查資料庫，看該區間最後一筆資料的時間，從那裡接續抓
        try:
            cursor = self.db.conn.cursor()
            cursor.execute('''
                SELECT MAX(timestamp) FROM external_data 
                WHERE symbol = ? AND metric = ? AND timestamp >= ? AND timestamp <= ?
            ''', (self.db_symbol, self.metric, start_ts, end_ts))
            result = cursor.fetchone()
            
            # 如果資料庫已經有部分資料，就從資料庫最後一筆的下一個毫秒開始抓
            if result and result[0]:
                current_start_ts = int(result[0]) + 1
            else:
                current_start_ts = start_ts
                
        except Exception as e:
            print(f"查詢 external_data 發生錯誤: {e}")
            current_start_ts = start_ts

        # 如果已經抓好抓滿，就結束
        if current_start_ts >= end_ts:
            print(f"[{self.metric}] 區間資料已達最新，無須補齊。\n")
            return

        print(f"開始從 {pd.to_datetime(current_start_ts, unit='ms')} 推進抓取...")

        # 核心邏輯：不斷向前推進，不依賴固定 interval
        while current_start_ts < end_ts:
            df_new = self.fetch_func(
                symbol=self.api_symbol, 
                limit=self.api_limit,
                startTime=current_start_ts,
                endTime=end_ts
            )

            if df_new.empty:
                print("  抓取完畢，已無更多資料。")
                break
                
            df_new['symbol'] = self.db_symbol
            
            # 寫入資料庫 (您現有的通用儲存法)
            self.db.save_generic_external_data(df_new)
            
            # 取得這批資料的「最大時間戳」，作為下一次的起點
            max_fetched_ts = int(df_new['open_time'].max())
            print(f"  └─ 成功寫入 {len(df_new)} 筆，推進至 {pd.to_datetime(max_fetched_ts, unit='ms')}")
            
            # 【推進游標】下一批從這批的最後一筆 + 1毫秒開始
            current_start_ts = max_fetched_ts + 1
            
            time.sleep(0.5) # 防止觸發 Rate Limit
            
        print(f"[{self.metric}] 所有區段資料補齊完畢！\n")
   
    # 3. 僅檢查不下載 
    
    def only_check(self, start_date, end_date):
        print(f"開始檢查 {self.db_symbol} ({self.interval}) 從 {start_date} 到 {end_date} ...")

        freq = self._get_pandas_freq()
        expected_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        expected_ts = set(expected_dates.astype('int64') // 10**6)

        existing_ts_list = self.db.get_existing_timestamps(
            symbol=self.db_symbol, 
            interval=self.interval, 
            start_ts=min(expected_ts), 
            end_ts=max(expected_ts)
        )
        existing_ts = set(existing_ts_list if existing_ts_list else [])
        missing_ts = sorted(list(expected_ts - existing_ts))
        
        if not missing_ts:
            print("資料庫資料完整，無缺失區塊！\n")
            return

        print(f"發現 {len(missing_ts)} 筆缺失資料\n")