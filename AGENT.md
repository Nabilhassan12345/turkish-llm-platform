# Türk LLM Sistemi - Agent Talimatları

## Oluşturma/Test Komutları
- `pip install -r requirements_benchmark.txt` - Bağımlılıkları yükle
- `python scripts/test_a4_components.py` - Bileşen testlerini çalıştır  
- `pytest` - Python birim testlerini çalıştır
- `black .` - Kodu formatla
- `flake8 .` - Python kodunu lint et

## Tek Test Komutları
- `python scripts/test_a4_components.py` - Tüm bileşenleri test et
- `python scripts/benchmark_metrics.py` - Performans kıyaslamalarını çalıştır
- `python scripts/run_benchmarks.py --host http://localhost:8000` - Yük testlerini çalıştır

## Mimari
- **Teknoloji**: PyTorch, Transformers, FastAPI, Streamlit ile Python ML/AI projesi
- **Temel Bileşenler**: Akıllı yönlendirme ile 22 sektöre özel Türk LLM adaptörleri
- **Yapı**: services/ (yönlendirici, çıkarım), scripts/ (testler, kıyaslamalar), ui/ (Streamlit), configs/ (sectors.yaml)
- **Ana Servisler**: SectorRouter (services/router.py), çıkarım API, web UI

## Kod Stili
- **İçe Aktarmalar**: Önce standart kütüphane, sonra üçüncü taraf, sonra yerel içe aktarmalar
- **İsimlendirme**: Değişkenler/fonksiyonlar için snake_case, sınıflar için PascalCase
- **Tipler**: Dataclass (@dataclass) ve tip ipuçları (typing modülü) kullan
- **Hata Yönetimi**: Yapılandırılmış günlük kaydı ile logging modülü
- **Dokümantasyon**: Fonksiyonlar ve sınıflar için docstring
