# OptiDemand AI - Enterprise ERP Demand Forecasting

Bu proje, mağaza ve ürün bazlı gelecek dönem satışlarını tahmin etmek amacıyla geliştirilmiş, uçtan uca (End-to-End) çalışan bir **MLOps & ERP** modülüdür. Geleneksel "geçmişe dönük" ERP raporlamalarının aksine, bu sistem makine öğrenmesi algoritmalarını kullanarak işletmelere **gelecek vizyonu** katar; stok maliyetlerini minimize etmeyi ve kayıp satışları engellemeyi hedefler.

## 🚀 Mimari Kararlar ve Teknoloji Yığını

Bu projenin her bir katmanı, kurumsal seviyede sürdürülebilirlik ve performans göz önüne alınarak seçilmiştir:
## 🚀 Teknolojiler
* **Backend:** Python, FastAPI, Pydantic, Uvicorn
* **Machine Learning:** LightGBM (Gradient Boosting Tree)
* **Frontend:** HTML5, CSS3 (Enterprise UI), Vanilla JS, Chart.js

## 🧠 Özellikler
* **Dinamik Tahmin Motoru:** Sezonluk etkiler, hafta sonu, özel günler (Black Friday) ve fiyat stratejilerine göre anlık reaksiyon gösteren model.
* **Kurşungeçirmez API:** Pydantic şemalarıyla korunan, veri hatalarında çökmeyen (Crash-proof) yapı.
* **Canlı Log Terminali:** Kullanıcı arayüzünde API ve kural motoru hareketlerini anlık izleme.
* **Stok İzleme:** AI tahminlerine göre güncellenen dinamik depo uyarı seviyeleri.

## 🛠️ Nasıl Çalıştırılır?
1. Repoyu klonlayın: `git clone https://github.com/mericsimsek/AIERPprojectwithMLOPS.git`
2. Gereksinimleri yükleyin: `pip install -r requirements.txt` (Opsiyonel)
3. API'yi başlatın: `uvicorn src.main:app --reload`
4. Arayüzü başlatın: `python -m http.server 3000`
5. Tarayıcıdan `http://localhost:3000` adresine gidin.

## 📂 Proje Dizini

```text
├── data/
│   ├── raw/                 # Ham veriler (GitHub'da hariç tutulur)
│   └── processed/           # İşlenmiş .parquet dosyaları
├── models/
│   └── lightgbm_model.pkl   # Eğitilmiş model dosyası
├── src/
│   ├── main.py              # FastAPI Orkestrasyonu
│   ├── preprocessing.py     # Veri Manipülasyonu ve Feature Engineering
│   ├── inference.py         # Model Tahmin Sınıfı
│   └── schemas.py           # Pydantic Doğrulama Şemaları
├── index.html               # Enterprise Dashboard UI
└── README.md