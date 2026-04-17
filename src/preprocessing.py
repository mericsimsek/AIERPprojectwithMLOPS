import pandas as pd
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, model_instance=None):
        self.model = model_instance
        
    def prepare_for_inference(self, data: dict) -> pd.DataFrame:
        df = pd.DataFrame([data])
        
        if self.model:
            all_features = self.model.feature_name()
            
            # API'den gelen veriler (Gelmezse varsayılan değerler)
            store = float(data.get('store', 1))
            item = float(data.get('item', 1))
            month = float(data.get('month', 1))
            day_of_week = float(data.get('day_of_week', 1))
            is_wknd = float(data.get('is_wknd', 0))
            trend = float(data.get('sales_trend_1month', 1.05))
            crowd = float(data.get('store_general_crowd', 45.0))
            pricing = data.get('pricing_strategy', 'normal')
            
            # 1. ZEKİ TABAN SATIŞ HESAPLAMASI (Deterministik)
            # Mağaza 1 daha çok satar, Ürün 1 daha çok satar.
            base_sales = 40 + (10 / store) + (20 / item) 
            
            # 2. MEVSİMSELLİK ETKİSİ (Sinüs dalgası: Yazın artar, kışın azalır)
            seasonality = math.sin((month - 3) / 12 * math.pi) * 20
            
            # 3. GÜN VE KAMPANYA ETKİSİ
            weekend_boost = 15 if is_wknd == 1 else 0
            promo_boost = 25 if pricing == "discount" else 0
            
            # 4. GECİKME (LAG) VERİLERİNİ HESAPLA (Artık rastgele DEĞİL!)
            calculated_history = base_sales + seasonality + weekend_boost + promo_boost
            
            for col in all_features:
                if col not in df.columns:
                    if 'lag' in col:
                        # Geçmiş veriler, hesapladığımız trende ve crowd(kalabalık) oranına göre belirlenir
                        df[col] = max(5, calculated_history * trend * (crowd / 50.0))
                    elif 'trend' in col:
                        df[col] = trend
                    else:
                        df[col] = 0 # Diğer özellikler 0
            
            # Kolonları modelin beklediği sıraya diz
            df = df[all_features]
            
            # LightGBM için Float dönüşümü
            float_cols = [c for c in df.columns if 'lag' in c or 'trend' in c]
            for col in float_cols:
                df[col] = df[col].astype(float)
                
        return df