import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# --- AYARLAR VE YOLLAR ---
# --- AYARLAR VE YOLLAR ---
# --- MLOPS DİNAMİK YOL BULUCU (KESİN ÇÖZÜM) ---
# Bu kod, train.py nerede olursa olsun ana proje klasörünü (erpai) otomatik bulur.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'train_cleaned.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODEL_DIR, exist_ok=True)
def load_data(path):
    print("1. Veri Parquet formatında yükleniyor...")
    df = pd.read_parquet(path)
    return df

def time_based_split(df):
    print("2. Veri kronolojik olarak (Time-Based Split) bölünüyor...")
    # Eğitim: 2017 öncesi
    X_train = df[df['year'] < 2017].drop(['sales'], axis=1)
    y_train = df[df['year'] < 2017]['sales']
    
    # Validasyon: 2017'nin ilk 3 ayı
    X_val = df[(df['year'] == 2017) & (df['month'] <= 3)].drop(['sales'], axis=1)
    y_val = df[(df['year'] == 2017) & (df['month'] <= 3)]['sales']
    
    return X_train, X_val, y_train, y_val

def train_and_log_model(X_train, X_val, y_train, y_val):
    print("3. MLflow başlatılıyor ve Model eğitiliyor...")
    
    # MLflow Deney Adı (Terapi Yazılım Mülakatı için şık bir isim)
    mlflow.set_experiment("Terapi_Yazilim_ERP_Tahmini")
    
    with mlflow.start_run():
        # LightGBM Parametreleri
        params = {
            'metric': 'mae',
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'max_depth': 6,
            'num_leaves': 63,
            'objective': 'regression',
            'random_state': 42,
            'verbose': -1
        }
        
        # Parametreleri MLflow'a kaydet
        mlflow.log_params(params)
        
        # LightGBM özel veri yapısı
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        # Modeli Eğit
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=1500,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        # Tahmin ve Metrik Hesaplama (Logaritmayı geri çeviriyoruz: expm1)
        print("4. Model test ediliyor ve metrikler hesaplanıyor...")
        preds = gbm.predict(X_val)
        y_val_real = np.expm1(y_val)
        preds_real = np.expm1(preds)
        
        mae = mean_absolute_error(y_val_real, preds_real)
        rmse = np.sqrt(mean_squared_error(y_val_real, preds_real))
        
        print(f"\n🔥 BAŞARI METRİKLERİ 🔥")
        print(f"Doğrulama MAE (Ortalama Hata): {mae:.2f} Adet")
        print(f"Doğrulama RMSE: {rmse:.2f} Adet")
        
        # Metrikleri ve Modeli MLflow'a kaydet
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_rmse", rmse)
        mlflow.lightgbm.log_model(gbm, "lightgbm-model")
        
        # Gelecekte FastAPI'da kullanmak için lokal .pkl kaydı
        model_path = os.path.join(MODEL_DIR, 'lightgbm_model.pkl')
        joblib.dump(gbm, model_path)
        print(f"✅ Model lokal olarak kaydedildi: {model_path}")
        print("✅ Pipeline başarıyla tamamlandı!")

if __name__ == "__main__":
    # Pipeline Akışı (Orkestrasyon)
    dataset = load_data(DATA_PATH)
    X_train, X_val, y_train, y_val = time_based_split(dataset)
    train_and_log_model(X_train, X_val, y_train, y_val)