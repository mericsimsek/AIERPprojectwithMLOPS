import joblib
import numpy as np
import os
import logging
import pandas as pd
logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Modeli RAM'e 1 kere yükler."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'lightgbm_model.pkl')
        
        try:
            self.model = joblib.load(model_path)
            logger.info("Model başarıyla RAM'e yüklendi.")
        except Exception as e:
            logger.critical(f"Model yüklenemedi: {e}")

    def predict(self, df: pd.DataFrame) -> int:
        """Gelen DataFrame'i tahmin edip logaritmayı çözer."""
        if self.model is None:
            raise RuntimeError("Model bulunamadı.")
        
        pred_log = self.model.predict(df)
        pred_real = np.expm1(pred_log)
        
        return int(round(pred_real[0]))