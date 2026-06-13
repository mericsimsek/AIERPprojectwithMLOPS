# 📦 OptiDemand AI — Enterprise ERP Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-ASGI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-F7931E?style=for-the-badge)
![MLOps](https://img.shields.io/badge/Architecture-MLOps-blueviolet?style=for-the-badge)

Mağaza ve ürün bazlı gelecek dönem satışlarını yüksek isabet oranıyla tahmin etmek amacıyla geliştirilmiş, uçtan uca (End-to-End) çalışan bir **MLOps ve ERP (Kurumsal Kaynak Planlaması)** modülüdür.

Geleneksel "geçmişe dönük" ERP raporlamalarının aksine, bu sistem makine öğrenmesi algoritmalarını kullanarak işletmelere **gelecek vizyonu** katar.

> **💼 Temel İş Hedefi (Business Objective):** Tedarik zincirindeki kamçı etkisini (bullwhip effect) kırmak, aşırı stok maliyetlerini minimize etmek ve yok-satma (out-of-stock) kaynaklı gelir kayıplarını engellemek.

---

## 📋 İçindekiler

- [Mimari Kararlar ve Teknoloji Yığını](#-mimari-kararlar-ve-teknoloji-yığını)
- [Core Özellikler ve İş Değeri](#-core-özellikler-ve-iş-değeri)
- [Proje Dizini ve MLOps Yaşam Döngüsü](#-proje-dizini-ve-mlops-yaşam-döngüsü)
- [API Kullanım Örneği](#-api-kullanım-örneği)
- [Kurulum ve Çalıştırma](#️-kurulum-ve-çalıştırma-rehberi)
- [Gelecek Yol Haritası](#-gelecek-yol-haritası-roadmap)

---

## 🚀 Mimari Kararlar ve Teknoloji Yığını

Bu projenin her bir katmanı, kurumsal seviyede **sürdürülebilirlik (maintainability)**, yüksek performans ve ölçeklenebilirlik göz önüne alınarak tasarlanmıştır.

| Katman | Teknoloji | Seçim Nedeni (Architecture Decision) |
|--------|-----------|---------------------------------------|
| **Backend / API** | Python, FastAPI, Uvicorn | Asenkron (ASGI) yapısı sayesinde anlık çoklu tahmin isteklerini darboğaz yaratmadan karşılar. Swagger/OpenAPI entegrasyonu otomatiktir. |
| **Veri Doğrulama** | Pydantic | API'ye gelen isteklerdeki hatalı veya eksik verileri (Tip güvenliği) yakalayarak sistemin sessizce çökmesini engeller (Crash-proof). |
| **Makine Öğrenmesi** | LightGBM (GBDT) | Zaman serisi ve karmaşık tablo (tabular) verilerinde XGBoost'a kıyasla daha hızlı eğitim ve çıkarım (inference) süresi sunar. Gürültülü satış verilerini iyi tolere eder. |
| **Arayüz (UI)** | HTML5, CSS3, Vanilla JS | Ağır framework'lere (React/Angular) bağımlı kalmadan tarayıcı üzerinde en hafif ve hızlı çalışan "Enterprise" standartlarında arayüz. |
| **Veri Görselleştirme** | Chart.js | Satış trendlerini ve güven aralıklarını performans kaybı yaşatmadan canvas üzerinde render eder. |

---

## 🧠 Core Özellikler ve İş Değeri

### 1. Zeki Preprocessing Motoru
*Otonom Özellik Çıkarımı*

Kullanıcıdan arayüz aracılığıyla yalnızca temel veriler (Mağaza ID, Ürün ID, Yıl, Ay, Fiyat Stratejisi vb.) alınır. Sistem, `preprocessing.py` modülü üzerinden arka planda **60'tan fazla karmaşık özelliği** (Hareketli ortalamalar, mevsimsellik endeksleri, lag değerleri) otonom olarak hesaplayıp tahmin modeline besler.

Bu soyutlama (abstraction), son kullanıcının sistemi karmaşık veri bilimi süreçlerini bilmeden kullanabilmesini sağlar.

---

### 2. Dinamik Senaryo Simülasyonu
*What-If Analysis*

Kullanıcılar arayüz üzerinden aşağıdaki dışsal faktörleri anlık olarak değiştirerek satışların nasıl etkilendiğini simüle edebilir:

- 🗓️ **Tatil günleri** — Black Friday, Yılbaşı vb.
- 🌦️ **Hava durumu etkileri**
- 💰 **Fiyatlandırma stratejileri** — İndirimli, Premium
- 📈 **Rakip pazar trendi** — Büyüme/Küçülme endeksi

Model, bu faktörlere göre tahminlerini anlık revize eder.

---

### 3. Fail-Safe (Hata Toleranslı) API ve Canlı Terminal

Yanlış veri girildiğinde veya arka plan servisleri çöktüğünde arayüz asla kilitlenmez. Kullanıcıya **Canlı Log Terminali** üzerinden hatanın Pydantic veri doğrulamasından mı yoksa ağ bağlantısından mı kaynaklandığı anlık olarak, bir geliştirici konsolu ciddiyetinde bildirilir.

---

### 4. Akıllı Stok İzleme Entegrasyonu

AI'dan gelen **30 günlük talep tahminleri**, mevcut depo stoklarıyla karşılaştırılarak otonom uyarılar üretir:

| Durum | Uyarı |
|-------|-------|
| Tahmin > Stok | 🔴 Stok Yetersiz |
| Tahmin ≤ Stok | 🟢 Güvenli |

Bu sayede satın alma departmanı için doğrudan eyleme dönüştürülebilir (actionable) içgörüler sunulur.

---

## 📂 Proje Dizini ve MLOps Yaşam Döngüsü

```
AIERPprojectwithMLOPS/
├── data/
│   ├── raw/                 # Ham satış verileri (Veri gizliliği için repoya eklenmez)
│   └── processed/           # Feature engineering uygulanmış .parquet dosyaları
├── models/
│   └── lightgbm_model.pkl   # Eğitilmiş ve serileştirilmiş model dosyası
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI Orkestrasyonu ve Endpointler
│   ├── preprocessing.py     # Veri Manipülasyonu ve Dinamik Mocking
│   ├── inference.py         # Model Tahmin (Predict) Sınıfı
│   └── schemas.py           # Pydantic Doğrulama Şemaları (Data Contracts)
├── index.html               # Enterprise Dashboard UI (Frontend)
└── README.md                # Proje Dokümantasyonu
```

---

## 🔌 API Kullanım Örneği

Sistem sadece bir arayüzden ibaret değildir; diğer ERP veya mobil uygulamaların da tüketebileceği standart bir **REST API** sunar.

**Örnek POST İsteği — `/predict`:**

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "store": 1,
    "item": 1,
    "year": 2026,
    "month": 11,
    "day_of_week": 4,
    "is_wknd": 0,
    "holiday_effect": "black_friday",
    "pricing_strategy": "discount",
    "weather_impact": "normal",
    "sales_trend_1month": 1.15
  }'
```

---

## 🛠️ Kurulum ve Çalıştırma Rehberi

**1. Repoyu Klonlayın:**
```bash
git clone https://github.com/mericsimsek/AIERPprojectwithMLOPS.git
cd AIERPprojectwithMLOPS
```

**2. Gerekli Kütüphaneleri Yükleyin:**
```bash
# Tercihen bir sanal ortam (venv) içinde çalıştırın
pip install fastapi uvicorn pandas numpy lightgbm joblib pydantic
```

**3. Backend (API) Sunucusunu Başlatın:**
```bash
uvicorn src.main:app --reload
```

API ayağa kalktıktan sonra [http://localhost:8000/docs](http://localhost:8000/docs) adresinden otomatik oluşturulan **Swagger UI** dokümantasyonuna erişebilirsiniz.

**4. Frontend (Arayüz) Sunucusunu Başlatın:**

CORS politikalarına takılmamak ve modern web standartlarında çalıştırmak için yeni bir terminal sekmesi açıp şu komutu girin:

```bash
python -m http.server 3000
```

**5. Sistemi Deneyimleyin:**

Tarayıcınızdan [http://localhost:3000](http://localhost:3000) adresine gidin ve **OptiDemand AI** panelini kullanmaya başlayın.

---

## 🔮 Gelecek Yol Haritası (Roadmap)

- [ ] **Dockerization** — Tüm projenin (API + UI) tek bir `docker-compose.yml` ile konteynerize edilmesi
- [ ] **Veritabanı Entegrasyonu** — SQLite/PostgreSQL kullanılarak geçmiş tahmin loglarının tutulması
- [ ] **Model Drift Monitoring** — Gelen verilerin zamanla modelin eğitim verisinden sapıp sapmadığını (data drift) ölçecek arayüz bileşenlerinin eklenmesi
