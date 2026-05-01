from base import BaseAlpha

class Strategy(BaseAlpha):
    """
    這是一個用來「測試引擎底層數學與對齊」的 Dummy 策略。
    邏輯：無腦做多黃金、放空白銀，且永遠不平倉。
    """
    
    # 測試策略不需要任何指標，所以需求特徵留空
    requirements = []  
    
    default_params = {}

    def prepare_features(self, df):
        # 什麼指標都不用算，直接回傳乾淨的原始 K 線
        return df

    def generate_target_position(self, row, account):
        # 永遠分配 50% 的資金買多黃金，50% 的資金放空白銀
        return {
            'YF_GLD': 0.5, 
            'YF_SLV': -0.5
        }