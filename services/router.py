import yaml
import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time
import threading
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SectorInfo:
    name: str
    description: str
    keywords: List[str]
    adapter_path: str
    priority: int
    expertise_domains: List[str]  # Uzmanlık alanları
    confidence_threshold: float  # Güven eşiği


@dataclass
class RouterConfig:
    default_adapter: str
    fallback_threshold: float
    max_experts_per_token: int
    load_balancing: str
    enable_moe: bool  # Mixture of Experts aktif
    expert_selection_strategy: str  # Uzman seçim stratejisi


class SectorRouter:
    """
    Sektör meta verilerine göre adapter seçen ve MoE (Mixture of Experts) sistemi için
    load balancing uygulayan router.
    """

    def __init__(self, config_path: str = "configs/sectors.yaml"):
        self.config_path = config_path
        self.sectors: Dict[str, SectorInfo] = {}
        self.router_config: Optional[RouterConfig] = None
        self.adapter_loads = defaultdict(int)
        self.load_lock = threading.Lock()
        self.last_round_robin = 0
        self.expert_performance = defaultdict(
            lambda: {"success_rate": 0.95, "avg_latency": 100}
        )

        # TF-IDF vektörizer için
        self.vectorizer = None
        self.sector_vectors = {}

        self._load_config()
        self._build_keyword_index()
        self._initialize_tfidf()

    def _load_config(self):
        """Sektör yapılandırmasını YAML dosyasından yükle."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Sektörleri yükle
            for sector_id, sector_data in config["sectors"].items():
                self.sectors[sector_id] = SectorInfo(
                    name=sector_data["name"],
                    description=sector_data["description"],
                    keywords=sector_data["keywords"],
                    adapter_path=sector_data["adapter_path"],
                    priority=sector_data["priority"],
                    expertise_domains=sector_data.get("expertise_domains", []),
                    confidence_threshold=sector_data.get("confidence_threshold", 0.7),
                )

            # Router yapılandırmasını yükle
            router_data = config["router"]
            self.router_config = RouterConfig(
                default_adapter=router_data["default_adapter"],
                fallback_threshold=router_data["fallback_threshold"],
                max_experts_per_token=router_data["max_experts_per_token"],
                load_balancing=router_data["load_balancing"],
                enable_moe=router_data.get("enable_moe", True),
                expert_selection_strategy=router_data.get(
                    "expert_selection_strategy", "confidence_weighted"
                ),
            )

            logger.info(f"{len(self.sectors)} sektör ve router yapılandırması yüklendi")

        except Exception as e:
            logger.error(f"Yapılandırma yüklenemedi: {e}")
            raise

    def _build_keyword_index(self):
        """Hızlı sektör eşleştirmesi için anahtar kelime indeksi oluştur."""
        self.keyword_to_sector = {}
        for sector_id, sector in self.sectors.items():
            for keyword in sector.keywords:
                self.keyword_to_sector[keyword.lower()] = sector_id

    def _initialize_tfidf(self):
        """TF-IDF vektörizer'ı başlat ve sektör vektörlerini oluştur."""
        try:
            # Tüm sektör açıklamalarını ve anahtar kelimelerini birleştir
            sector_texts = []
            sector_ids = []

            for sector_id, sector in self.sectors.items():
                text = f"{sector.description} {' '.join(sector.keywords)} {' '.join(sector.expertise_domains)}"
                sector_texts.append(text)
                sector_ids.append(sector_id)

            # TF-IDF vektörizer'ı eğit
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=None,  # Türkçe stop words eklenebilir
                ngram_range=(1, 2),
            )

            # Vektörleri oluştur
            vectors = self.vectorizer.fit_transform(sector_texts)

            # Her sektör için vektörü sakla
            for i, sector_id in enumerate(sector_ids):
                self.sector_vectors[sector_id] = vectors[i]

            logger.info("TF-IDF vektörizer başlatıldı ve sektör vektörleri oluşturuldu")

        except Exception as e:
            logger.warning(f"TF-IDF başlatılamadı, keyword matching kullanılacak: {e}")
            self.vectorizer = None

    def classify_sector(self, text: str) -> List[Tuple[str, float]]:
        """
        Metni sektörlere sınıflandır ve güven skorları ile birlikte döndür.

        Args:
            text: Sınıflandırılacak metin

        Returns:
            (sector_id, confidence_score) tuple'larının güven skoruna göre sıralanmış listesi
        """
        text_lower = text.lower()
        sector_scores = defaultdict(float)

        # 1. Anahtar kelime eşleştirmesi (hızlı)
        keyword_matches = self._keyword_matching(text_lower)
        for sector_id, score in keyword_matches.items():
            sector_scores[sector_id] += score * 0.6  # %60 ağırlık

        # 2. TF-IDF benzerlik skoru (daha doğru)
        if self.vectorizer is not None:
            tfidf_scores = self._tfidf_similarity(text)
            for sector_id, score in tfidf_scores.items():
                sector_scores[sector_id] += score * 0.4  # %40 ağırlık

        # 3. Uzmanlık alanı eşleştirmesi
        expertise_scores = self._expertise_matching(text_lower)
        for sector_id, score in expertise_scores.items():
            sector_scores[sector_id] += score * 0.2  # %20 bonus

        # 4. Öncelik bonusu
        for sector_id, score in sector_scores.items():
            priority_bonus = (10 - self.sectors[sector_id].priority) * 0.01
            sector_scores[sector_id] += priority_bonus

        # Güven skorlarını normalize et (0-1 arası)
        max_score = max(sector_scores.values()) if sector_scores else 1
        if max_score > 0:
            for sector_id in sector_scores:
                sector_scores[sector_id] = min(
                    1.0, sector_scores[sector_id] / max_score
                )

        # Güven skoruna göre sırala
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)

        # Sadece güven eşiğini geçen sektörleri döndür
        threshold = self.router_config.fallback_threshold
        filtered_sectors = [
            (sector_id, score)
            for sector_id, score in sorted_sectors
            if score >= threshold
        ]

        logger.info(f"Metin sınıflandırıldı: {text[:50]}... -> {filtered_sectors[:3]}")
        return filtered_sectors

    def _keyword_matching(self, text: str) -> Dict[str, float]:
        """Anahtar kelime eşleştirmesi ile sektör skorları hesapla."""
        sector_scores = defaultdict(float)

        for keyword, sector_id in self.keyword_to_sector.items():
            if keyword in text:
                # Anahtar kelime uzunluğuna göre bonus
                keyword_bonus = len(keyword) * 0.1
                sector_scores[sector_id] += 1.0 + keyword_bonus

        return sector_scores

    def _tfidf_similarity(self, text: str) -> Dict[str, float]:
        """TF-IDF benzerlik skorları hesapla."""
        try:
            # Metni vektörize et
            text_vector = self.vectorizer.transform([text])

            # Her sektör ile benzerlik hesapla
            similarities = {}
            for sector_id, sector_vector in self.sector_vectors.items():
                similarity = cosine_similarity(text_vector, sector_vector)[0][0]
                similarities[sector_id] = similarity

            return similarities

        except Exception as e:
            logger.warning(f"TF-IDF benzerlik hesaplanamadı: {e}")
            return {}

    def _expertise_matching(self, text: str) -> Dict[str, float]:
        """Uzmanlık alanı eşleştirmesi ile bonus skorlar hesapla."""
        expertise_scores = defaultdict(float)

        for sector_id, sector in self.sectors.items():
            for domain in sector.expertise_domains:
                if domain.lower() in text:
                    expertise_scores[sector_id] += 0.5  # Uzmanlık bonusu

        return expertise_scores

    def select_adapters(self, text: str, num_experts: int = None) -> List[str]:
        """
        MoE sistemi için en uygun adapter'ları seç.

        Args:
            text: Giriş metni
            num_experts: Kullanılacak uzman sayısı

        Returns:
            Seçilen adapter yollarının listesi
        """
        if not self.router_config.enable_moe:
            # MoE kapalıysa sadece en iyi sektörü seç
            sectors = self.classify_sector(text)
            if sectors:
                return [self.sectors[sectors[0][0]].adapter_path]
            return [self.router_config.default_adapter]

        # Sektörleri sınıflandır
        classified_sectors = self.classify_sector(text)

        if not classified_sectors:
            logger.warning(
                "Hiçbir sektör sınıflandırılamadı, varsayılan adapter kullanılıyor"
            )
            return [self.router_config.default_adapter]

        # Uzman sayısını belirle
        if num_experts is None:
            num_experts = min(
                self.router_config.max_experts_per_token, len(classified_sectors)
            )

        # En iyi N sektörü seç
        selected_sectors = classified_sectors[:num_experts]
        selected_adapters = []

        for sector_id, confidence in selected_sectors:
            sector_info = self.sectors[sector_id]

            # Güven eşiğini kontrol et
            if confidence >= sector_info.confidence_threshold:
                selected_adapters.append(sector_info.adapter_path)
                logger.info(
                    f"✅ Adapter seçildi: {sector_id} (Güven: {confidence:.3f})"
                )
            else:
                logger.warning(
                    f"⚠️ Güven eşiği düşük: {sector_id} (Güven: {confidence:.3f}, Eşik: {sector_info.confidence_threshold})"
                )

        # Hiçbir adapter seçilmediyse varsayılanı kullan
        if not selected_adapters:
            logger.warning("Hiçbir adapter seçilemedi, varsayılan kullanılıyor")
            selected_adapters = [self.router_config.default_adapter]

        # Load balancing uygula
        if self.router_config.load_balancing == "round_robin":
            selected_adapters = self._apply_round_robin(selected_adapters)
        elif self.router_config.load_balancing == "performance_weighted":
            selected_adapters = self._apply_performance_weighted(selected_adapters)

        # Adapter yüklerini güncelle
        with self.load_lock:
            for adapter in selected_adapters:
                self.adapter_loads[adapter] += 1

        logger.info(f"🎯 {len(selected_adapters)} adapter seçildi: {selected_adapters}")
        return selected_adapters

    def _apply_round_robin(self, adapters: List[str]) -> List[str]:
        """Round-robin load balancing uygula."""
        if not adapters:
            return adapters

        with self.load_lock:
            self.last_round_robin = (self.last_round_robin + 1) % len(adapters)
            # Rotasyon uygula
            rotated = (
                adapters[self.last_round_robin :] + adapters[: self.last_round_robin]
            )
            return rotated

    def _apply_performance_weighted(self, adapters: List[str]) -> List[str]:
        """Performans ağırlıklı load balancing uygula."""
        if not adapters:
            return adapters

        # Performans skorlarına göre sırala
        performance_scores = []
        for adapter in adapters:
            # Adapter yükü ve performans metriklerini kullan
            load = self.adapter_loads.get(adapter, 0)
            success_rate = self.expert_performance[adapter]["success_rate"]
            avg_latency = self.expert_performance[adapter]["avg_latency"]

            # Performans skoru: yüksek başarı oranı, düşük gecikme, düşük yük
            performance_score = (
                (success_rate * 0.5)
                + (1.0 / (1 + avg_latency / 100) * 0.3)
                + (1.0 / (1 + load / 10) * 0.2)
            )
            performance_scores.append((adapter, performance_score))

        # Performans skoruna göre sırala
        performance_scores.sort(key=lambda x: x[1], reverse=True)
        return [adapter for adapter, _ in performance_scores]

    def update_expert_performance(
        self, adapter_path: str, success: bool, latency: float
    ):
        """Uzman performans metriklerini güncelle."""
        with self.load_lock:
            current = self.expert_performance[adapter_path]

            # Başarı oranını güncelle (exponential moving average)
            alpha = 0.1
            current["success_rate"] = (
                alpha * (1.0 if success else 0.0)
                + (1 - alpha) * current["success_rate"]
            )

            # Ortalama gecikmeyi güncelle
            current["avg_latency"] = (
                alpha * latency + (1 - alpha) * current["avg_latency"]
            )

            logger.debug(
                f"Performans güncellendi: {adapter_path} - Başarı: {current['success_rate']:.3f}, Gecikme: {current['avg_latency']:.2f}ms"
            )

    def get_load_statistics(self) -> Dict[str, int]:
        """Adapter yük istatistiklerini döndür."""
        with self.load_lock:
            return dict(self.adapter_loads)

    def reset_load_statistics(self):
        """Yük istatistiklerini sıfırla."""
        with self.load_lock:
            self.adapter_loads.clear()
            logger.info("Yük istatistikleri sıfırlandı")

    def get_sector_info(self, sector_id: str) -> Optional[SectorInfo]:
        """Belirli bir sektörün bilgilerini döndür."""
        return self.sectors.get(sector_id)

    def list_sectors(self) -> List[str]:
        """Tüm sektör ID'lerini listele."""
        return list(self.sectors.keys())

    def get_router_status(self) -> Dict:
        """Router durum bilgilerini döndür."""
        return {
            "total_sectors": len(self.sectors),
            "enabled_moe": self.router_config.enable_moe,
            "load_balancing_strategy": self.router_config.load_balancing,
            "max_experts_per_token": self.router_config.max_experts_per_token,
            "adapter_loads": dict(self.adapter_loads),
            "expert_performance": dict(self.expert_performance),
            "last_round_robin_index": self.last_round_robin,
        }

    def health_check(self) -> Dict[str, bool]:
        """Router sağlık kontrolü."""
        try:
            # Yapılandırma dosyası kontrolü
            config_exists = Path(self.config_path).exists()

            # Sektör yükleme kontrolü
            sectors_loaded = len(self.sectors) > 0

            # TF-IDF kontrolü
            tfidf_ready = self.vectorizer is not None

            return {
                "config_loaded": config_exists,
                "sectors_loaded": sectors_loaded,
                "tfidf_ready": tfidf_ready,
                "overall_healthy": config_exists and sectors_loaded,
            }

        except Exception as e:
            logger.error(f"Sağlık kontrolü hatası: {e}")
            return {
                "config_loaded": False,
                "sectors_loaded": False,
                "tfidf_ready": False,
                "overall_healthy": False,
                "error": str(e),
            }


# Example usage and testing
if __name__ == "__main__":
    # Initialize router
    router = SectorRouter()

    # Test cases
    test_cases = [
        "Banka kredisi almak istiyorum, faiz oranları nedir?",
        "Hastane randevusu almak istiyorum, hangi doktorlar müsait?",
        "Üniversite sınavına hazırlanıyorum, hangi kurslar önerilir?",
        "Yazılım geliştirme projesi için teknoloji çözümleri arıyorum",
        "Mağazada ürün satışı yapıyorum, müşteri memnuniyeti nasıl artırılır?",
        "Fabrikada üretim süreçlerini optimize etmek istiyorum",
    ]

    print("=== Sector Router Test Results ===")
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_text}")

        # Classify sector
        sector_scores = router.classify_sector(test_text)
        print(f"Classification: {sector_scores}")

        # Select adapters
        adapters = router.select_adapters(test_text)
        print(f"Selected adapters: {adapters}")

    # Show load statistics
    print(f"\n=== Load Statistics ===")
    print(router.get_load_statistics())
