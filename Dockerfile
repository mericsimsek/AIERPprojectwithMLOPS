# 1. Hangi Python sürümünü kullanacağız? (Temel İskelet)
FROM python:3.11-slim

# 2. Konteynerin içindeki çalışma klasörümüzün adını belirliyoruz
WORKDIR /app

# 3. Kütüphane listesini konteynere kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Projedeki her şeyi (src, models, data) konteynerin içine kopyala
COPY . .

# 5. API'nin çalışacağı portu dışarıya aç
EXPOSE 8000

# 6. Konteyner ayağa kalktığında otomatik çalışacak komut (Garsonu başlat)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]