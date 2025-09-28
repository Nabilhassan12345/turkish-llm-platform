#!/usr/bin/env python3
"""
Locust load testing script for Turkish LLM inference benchmarking.
Bu script, LLM sisteminin ölçeklenebilirliğini ve performansını farklı yük
desenleri ve sektöre özel sorgularla test eder.
"""

import time
import json
import random
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from locust import HttpUser, task, between, events
import numpy as np
from services.router import SectorRouter
from collections import defaultdict
import psutil
import GPUtil
import os
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Benchmark parametreleri için yapılandırma."""

    target_rps: int = 100  # Hedef saniye başına istek
    latency_threshold_ms: int = 200  # Gecikme eşiği (ms)
    max_concurrent_users: int = 200  # Maksimum eşzamanlı kullanıcı
    test_duration_seconds: int = 600  # Test süresi (10 dakika)
    warmup_seconds: int = 60  # Isınma süresi
    sector_weights: Dict[str, float] = None  # Sektör ağırlıkları


class TurkishLLMUser(HttpUser):
    """
    Türk LLM çıkarım performansını test etmek için Locust kullanıcı sınıfı.
    """

    wait_time = between(0.5, 2)  # İstekler arası 0.5-2 saniye bekle

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = SectorRouter()
        self.sector_queries = self._load_sector_queries()
        self.benchmark_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "latencies": [],
            "sector_distribution": defaultdict(int),
            "error_types": defaultdict(int),
            "response_sizes": [],
        }

        # Sektör ağırlıkları (gerçek dünya kullanım desenlerine göre)
        self.sector_weights = {
            "finance_banking": 0.25,  # %25 - En yüksek kullanım
            "healthcare": 0.20,  # %20
            "education": 0.15,  # %15
            "ecommerce": 0.12,  # %12
            "public_administration": 0.10,  # %10
            "legal": 0.08,  # %8
            "manufacturing": 0.05,  # %5
            "tourism_hospitality": 0.03,  # %3
            "energy": 0.02,  # %2
        }

    def _load_sector_queries(self) -> Dict[str, List[str]]:
        """Sektöre özel test sorgularını yükle."""
        return {
            "finance_banking": [
                "Banka kredisi almak istiyorum, faiz oranları nedir?",
                "Döviz kuru tahminleri nasıl yapılır?",
                "Yatırım portföyümü nasıl çeşitlendirebilirim?",
                "Sigorta poliçesi seçerken nelere dikkat etmeliyim?",
                "Borsa analizi için hangi göstergeleri takip etmeliyim?",
                "Kredi kartı limitimi nasıl artırabilirim?",
                "Mevduat hesabı açmak için hangi belgeler gerekli?",
                "Fon yatırımı yapmak istiyorum, risk seviyeleri nelerdir?",
            ],
            "healthcare": [
                "Hastane randevusu almak istiyorum, hangi doktorlar müsait?",
                "Bu ilacın yan etkileri nelerdir?",
                "Sağlık sigortası kapsamında hangi tedaviler var?",
                "Doktor muayenesi öncesi nasıl hazırlanmalıyım?",
                "Eczane çalışma saatleri nedir?",
                "Acil servis nerede bulunuyor?",
                "Laboratuvar sonuçlarım ne zaman hazır olur?",
                "Hangi aşıları olmam gerekiyor?",
            ],
            "education": [
                "Üniversite sınavına hazırlanıyorum, hangi kurslar önerilir?",
                "Online eğitim platformları hangileri?",
                "Öğretmenlik sertifikası nasıl alınır?",
                "Yabancı dil öğrenmek için en iyi yöntemler neler?",
                "Sınav stresiyle nasıl başa çıkabilirim?",
                "Burs başvurusu için son tarih ne zaman?",
                "Hangi bölümlerde iş imkanı daha fazla?",
                "Yaz okulu programları nelerdir?",
            ],
            "ecommerce": [
                "Ürün iadesi nasıl yapılır?",
                "Kargo takip numarası nerede bulunur?",
                "Ödeme seçenekleri nelerdir?",
                "Ürün garantisi ne kadar sürer?",
                "Stok durumu nasıl kontrol edilir?",
                "İndirim kuponları nasıl kullanılır?",
                "Müşteri hizmetleri ile nasıl iletişime geçebilirim?",
                "Ürün karşılaştırması nasıl yapılır?",
            ],
            "public_administration": [
                "Kimlik kartı yenileme işlemi nasıl yapılır?",
                "Pasaport başvurusu için hangi belgeler gerekli?",
                "Vergi ödemeleri ne zaman yapılmalı?",
                "Belediye hizmetleri nelerdir?",
                "Sosyal güvenlik başvurusu nasıl yapılır?",
                "Askerlik işlemleri nerede yapılır?",
                "Tapu işlemleri için randevu nasıl alınır?",
                "İşsizlik maaşı başvurusu nasıl yapılır?",
            ],
            "legal": [
                "Avukat ücretleri nasıl belirlenir?",
                "Dava süreci ne kadar sürer?",
                "Hukuki danışmanlık ücreti nedir?",
                "Sözleşme hazırlama maliyeti nedir?",
                "Noter işlemleri nelerdir?",
                "Aile hukuku konularında hangi haklarım var?",
                "İş hukuku kapsamında neler yapabilirim?",
                "Ceza hukuku süreçleri nasıl işler?",
            ],
            "manufacturing": [
                "Fabrikada üretim süreçlerini optimize etmek istiyorum",
                "Kalite kontrol süreçleri nasıl iyileştirilir?",
                "Makine bakım planlaması nasıl yapılır?",
                "Tedarik zinciri yönetimi stratejileri",
                "Üretim maliyetlerini nasıl düşürebilirim?",
                "İş güvenliği önlemleri nelerdir?",
                "Çevre dostu üretim nasıl yapılır?",
                "Otomasyon süreçleri nasıl planlanır?",
            ],
            "tourism_hospitality": [
                "Otel rezervasyonu nasıl yapılır?",
                "Restoran menüsünde neler var?",
                "Tur rehberi hizmetleri nelerdir?",
                "Havalimanı transfer hizmeti var mı?",
                "Oda servisi saatleri nelerdir?",
                "Spa ve wellness hizmetleri nelerdir?",
                "Konferans salonu kiralama ücreti nedir?",
                "Özel etkinlik organizasyonu yapılıyor mu?",
            ],
            "energy": [
                "Elektrik faturası nasıl ödenir?",
                "Doğalgaz kesintisi ne zaman sona erecek?",
                "Güneş enerjisi kurulum maliyeti nedir?",
                "Enerji tasarrufu önerileri nelerdir?",
                "Elektrik arıza bildirimi nasıl yapılır?",
                "Yenilenebilir enerji teşvikleri nelerdir?",
                "Enerji verimliliği danışmanlığı var mı?",
                "Akıllı sayaç kurulumu nasıl yapılır?",
            ],
        }

    def on_start(self):
        """Kullanıcı başladığında çağrılır."""
        logger.info(f"Kullanıcı {self.client.base_url} başladı")
        self._log_system_info()

    def _log_system_info(self):
        """Sistem bilgilerini logla."""
        try:
            # CPU ve RAM bilgileri
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # GPU bilgileri (varsa)
            gpu_info = "GPU bilgisi bulunamadı"
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info = f"GPU: {gpus[0].name}, Memory: {gpus[0].memoryUsed}MB/{gpus[0].memoryTotal}MB"
            except:
                pass

            logger.info(
                f"Sistem Bilgileri - CPU: {cpu_percent}%, RAM: {memory.percent}%, {gpu_info}"
            )

        except Exception as e:
            logger.warning(f"Sistem bilgileri alınamadı: {e}")

    @task(3)
    def test_sector_specific_query(self):
        """Sektöre özel sorgu testi - %75 ağırlık."""
        # Ağırlıklara göre sektör seç
        sector = self._select_sector_by_weight()
        queries = self.sector_queries.get(sector, [])

        if not queries:
            logger.warning(f"{sector} için sorgu bulunamadı")
            return

        query = random.choice(queries)

        # Router endpoint'ini test et
        start_time = time.time()

        try:
            response = self.client.post("/classify", json={"text": query})

            if response.status_code == 200:
                result = response.json()
                latency = (time.time() - start_time) * 1000  # ms cinsinden

                # Metrikleri güncelle
                self.benchmark_metrics["total_requests"] += 1
                self.benchmark_metrics["successful_requests"] += 1
                self.benchmark_metrics["latencies"].append(latency)
                self.benchmark_metrics["sector_distribution"][sector] += 1
                self.benchmark_metrics["response_sizes"].append(len(response.content))

                # Başarılı yanıt kontrolü
                if result.get("sector") == sector:
                    logger.info(
                        f"✅ {sector}: Doğru sınıflandırma, Gecikme: {latency:.2f}ms"
                    )
                else:
                    logger.warning(
                        f"⚠️ {sector}: Yanlış sınıflandırma, Beklenen: {sector}, Alınan: {result.get('sector')}"
                    )

            else:
                self.benchmark_metrics["failed_requests"] += 1
                self.benchmark_metrics["error_types"][
                    f"HTTP_{response.status_code}"
                ] += 1
                logger.error(f"❌ {sector}: HTTP {response.status_code} hatası")

        except Exception as e:
            self.benchmark_metrics["failed_requests"] += 1
            self.benchmark_metrics["error_types"][str(type(e).__name__)] += 1
            logger.error(f"❌ {sector}: İstek hatası: {e}")

    def _select_sector_by_weight(self) -> str:
        """Ağırlıklara göre sektör seç."""
        sectors = list(self.sector_weights.keys())
        weights = list(self.sector_weights.values())
        return random.choices(sectors, weights=weights, k=1)[0]

    @task(1)
    def test_general_query(self):
        """Genel sorgu testi - %25 ağırlık."""
        general_queries = [
            "Merhaba, nasılsınız?",
            "Bugün hava nasıl?",
            "Türkiye'nin başkenti neresidir?",
            "Matematik problemlerini nasıl çözebilirim?",
            "Yemek tarifi arıyorum",
            "Spor müsabakası sonuçları neler?",
            "Kitap önerisi istiyorum",
            "Müzik türleri hakkında bilgi verir misiniz?",
        ]

        query = random.choice(general_queries)
        start_time = time.time()

        try:
            response = self.client.post(
                "/infer", json={"text": query, "max_length": 150, "temperature": 0.7}
            )

            if response.status_code == 200:
                result = response.json()
                latency = (time.time() - start_time) * 1000

                self.benchmark_metrics["total_requests"] += 1
                self.benchmark_metrics["successful_requests"] += 1
                self.benchmark_metrics["latencies"].append(latency)
                self.benchmark_metrics["sector_distribution"]["general"] += 1
                self.benchmark_metrics["response_sizes"].append(len(response.content))

                logger.info(f"✅ Genel sorgu: Yanıt alındı, Gecikme: {latency:.2f}ms")

            else:
                self.benchmark_metrics["failed_requests"] += 1
                self.benchmark_metrics["error_types"][
                    f"HTTP_{response.status_code}"
                ] += 1
                logger.error(f"❌ Genel sorgu: HTTP {response.status_code} hatası")

        except Exception as e:
            self.benchmark_metrics["failed_requests"] += 1
            self.benchmark_metrics["error_types"][str(type(e).__name__)] += 1
            logger.error(f"❌ Genel sorgu hatası: {e}")

    @task(1)
    def test_router_endpoint(self):
        """Router endpoint testi."""
        test_texts = [
            "Banka kredisi almak istiyorum",
            "Hastane randevusu almak istiyorum",
            "Üniversite sınavına hazırlanıyorum",
            "E-ticaret sitesi kurmak istiyorum",
            "Fabrikada üretim süreçlerini optimize etmek istiyorum",
        ]

        text = random.choice(test_texts)
        start_time = time.time()

        try:
            response = self.client.post("/classify", json={"text": text})

            if response.status_code == 200:
                result = response.json()
                latency = (time.time() - start_time) * 1000

                self.benchmark_metrics["total_requests"] += 1
                self.benchmark_metrics["successful_requests"] += 1
                self.benchmark_metrics["latencies"].append(latency)
                self.benchmark_metrics["response_sizes"].append(len(response.content))

                logger.info(
                    f"✅ Router test: {result.get('sector')}, Gecikme: {latency:.2f}ms"
                )

            else:
                self.benchmark_metrics["failed_requests"] += 1
                self.benchmark_metrics["error_types"][
                    f"HTTP_{response.status_code}"
                ] += 1
                logger.error(f"❌ Router test: HTTP {response.status_code} hatası")

        except Exception as e:
            self.benchmark_metrics["failed_requests"] += 1
            self.benchmark_metrics["error_types"][str(type(e).__name__)] += 1
            logger.error(f"❌ Router test hatası: {e}")

    def on_stop(self):
        """Kullanıcı durduğunda çağrılır."""
        logger.info(f"Kullanıcı {self.client.base_url} durdu")
        self._log_metrics()

    def _log_metrics(self):
        """Metrikleri logla."""
        if self.benchmark_metrics["total_requests"] > 0:
            success_rate = (
                self.benchmark_metrics["successful_requests"]
                / self.benchmark_metrics["total_requests"]
            ) * 100
            avg_latency = (
                np.mean(self.benchmark_metrics["latencies"])
                if self.benchmark_metrics["latencies"]
                else 0
            )

            logger.info(
                f"📊 Metrikler - Toplam: {self.benchmark_metrics['total_requests']}, "
                f"Başarı: {success_rate:.1f}%, Ortalama Gecikme: {avg_latency:.2f}ms"
            )


class BenchmarkReporter:
    """Benchmark sonuçlarını raporlayan sınıf."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()

    def on_request_success(self, request_type, name, response_time, response_length):
        """Başarılı istek metriklerini kaydet."""
        self.metrics["successful_requests"].append(
            {
                "type": request_type,
                "name": name,
                "response_time": response_time,
                "response_length": response_length,
                "timestamp": time.time(),
            }
        )

    def on_request_failure(self, request_type, name, response_time, exception):
        """Başarısız istek metriklerini kaydet."""
        self.metrics["failed_requests"].append(
            {
                "type": request_type,
                "name": name,
                "response_time": response_time,
                "exception": str(exception),
                "timestamp": time.time(),
            }
        )

    def generate_report(self) -> Dict:
        """Benchmark raporu oluştur."""
        end_time = time.time()
        duration = end_time - self.start_time

        successful = self.metrics["successful_requests"]
        failed = self.metrics["failed_requests"]

        if successful:
            response_times = [req["response_time"] for req in successful]
            avg_response_time = np.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = p95_response_time = p99_response_time = 0

        return {
            "test_duration_seconds": duration,
            "total_requests": len(successful) + len(failed),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate_percent": (
                (len(successful) / (len(successful) + len(failed))) * 100
                if (len(successful) + len(failed)) > 0
                else 0
            ),
            "avg_response_time_ms": avg_response_time,
            "p95_response_time_ms": p95_response_time,
            "p99_response_time_ms": p99_response_time,
            "requests_per_second": (
                (len(successful) + len(failed)) / duration if duration > 0 else 0
            ),
            "error_summary": (
                dict(Counter([req["exception"] for req in failed])) if failed else {}
            ),
        }


# Global reporter instance
reporter = BenchmarkReporter()


@events.request.add_listener
def on_request_success(request_type, name, response_time, response_length, **kwargs):
    """Başarılı istek event listener'ı."""
    reporter.on_request_success(request_type, name, response_time, response_length)


@events.request.add_listener
def on_request_failure(request_type, name, response_time, exception, **kwargs):
    """Başarısız istek event listener'ı."""
    reporter.on_request_failure(request_type, name, response_time, exception)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Test durduğunda final raporu oluştur."""
    report = reporter.generate_report()

    # Raporu dosyaya kaydet
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"benchmark_results/locust_report_{timestamp}.json"

    os.makedirs("benchmark_results", exist_ok=True)

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"📊 Benchmark raporu kaydedildi: {report_file}")
    logger.info(
        f"📈 Test Sonuçları: {report['successful_requests']} başarılı, "
        f"{report['failed_requests']} başarısız, "
        f"Ortalama gecikme: {report['avg_response_time_ms']:.2f}ms"
    )
