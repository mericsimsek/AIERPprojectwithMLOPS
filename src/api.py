from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
import os

# 1. API (Garson) Uygulamasını Başlat
app = FastAPI(
    title="Terapi Yazılım ERP Forecast API",
    description="Mağaza satışlarını tahmin eden yapay zeka servisi.",
    version="1.0.0"
)

# 2. Modeli Hafızaya Al (Aşçıyı Uyandır)
# API her istekte modeli baştan okumasın diye modeli en başta 1 kere RAM'e yüklüyoruz.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lightgbm_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model başarıyla RAM'e yüklendi!")
except Exception as e:
    print(f"❌ Model yüklenirken hata oluştu: {e}")

# 3. Karşılama Ekranı (Ana Sayfa)
@app.get("/")
def home():
    return {"mesaj": "ERP Tahmin API'si tıkır tıkır çalışıyor. '/docs' adresine giderek test edebilirsiniz."}

# 4. Asıl Tahmin Uç Noktası (Endpoint)
@app.post("/predict")
def predict_sales(data: dict):
    try:
        # ERP'den gelen JSON verisini Pandas tablosuna çevir (Yatay tek bir satır)
        df = pd.DataFrame([data])
        
        # Aşçıya (Modele) veriyi ver ve tahmini al
        pred_log = model.predict(df)
        
        # Hatırla: Model logaritmik öğrenmişti. Geri gerçek satış adedine (expm1) çeviriyoruz!
        pred_real = np.expm1(pred_log)
        
        # Tahmini tam sayıya yuvarlayıp müşteriye (ERP'ye) geri gönder
        return {
            "status": "success",
            "predicted_sales": int(round(pred_real[0]))
        }
    except Exception as e:
        # Veride bir eksiklik veya hata varsa uygulamanın çökmesini engelle
        raise HTTPException(status_code=400, detail=f"Tahmin yapılamadı. Hata: {str(e)}")