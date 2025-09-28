#!/usr/bin/env python3
"""
Turkish LLM Training Data Generator
Generates sector-specific training data for the 22 Turkish business sectors.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TurkishDataGenerator:
    """Generator for Turkish sector-specific training data."""

    def __init__(self, config_path: str = "configs/sectors.yaml"):
        self.config_path = config_path
        self.sectors = self._load_sectors()

    def _load_sectors(self) -> Dict:
        """Load sector configuration."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config["sectors"]

    def _generate_sector_queries(self, sector_id: str, sector_info: Dict) -> List[str]:
        """Generate queries for a specific sector."""

        # Base query templates for each sector
        query_templates = {
            "finance_banking": [
                "Banka kredisi almak istiyorum, nasıl başvurabilirim?",
                "Yatırım danışmanlığı hizmetleri hakkında bilgi istiyorum",
                "Faiz oranları nasıl hesaplanır?",
                "Portföy yönetimi stratejileri nelerdir?",
                "Kredi kartı başvurusu için gerekli belgeler neler?",
                "Döviz kuru tahminleri nasıl yapılır?",
                "Borsa yatırımı için önerileriniz neler?",
                "Mevduat hesabı açmak istiyorum",
                "Kredi notu nasıl yükseltilir?",
                "Finansal planlama için danışmanlık arıyorum",
            ],
            "healthcare": [
                "Hastane randevusu almak istiyorum",
                "Doktor muayenesi için ne yapmam gerekiyor?",
                "İlaç kullanımı hakkında bilgi istiyorum",
                "Tedavi süreci nasıl işler?",
                "Sağlık sigortası kapsamı nedir?",
                "Laboratuvar sonuçlarımı nasıl anlayabilirim?",
                "Ameliyat öncesi hazırlık süreci nedir?",
                "Klinik randevusu nasıl alınır?",
                "Radyoloji raporu ne anlama geliyor?",
                "Eczane hizmetleri hakkında bilgi istiyorum",
            ],
            "education": [
                "Üniversite sınavına hazırlanıyorum, hangi kurslar önerilir?",
                "Kurs seçimi nasıl yapılır?",
                "Eğitim danışmanlığı hizmetleri nelerdir?",
                "Öğretmen ataması süreci nasıl işler?",
                "Öğrenci bursu başvurusu nasıl yapılır?",
                "Ders programı nasıl oluşturulur?",
                "Sınav hazırlık stratejileri nelerdir?",
                "Akademi programları hakkında bilgi istiyorum",
                "Seminer kayıtları nasıl yapılır?",
                "Workshop etkinlikleri ne zaman düzenlenir?",
            ],
            "media_publishing": [
                "Gazete için muhabir arıyoruz, başvuru süreci nasıl?",
                "Dergi editörü pozisyonu açık mı?",
                "Televizyon programı yapımı hakkında bilgi",
                "Radyo yayını için teknik destek arıyoruz",
                "Medya kuruluşu kurmak istiyorum",
                "Yayın lisansı nasıl alınır?",
                "Haber editörü iş tanımı nedir?",
                "Reporter pozisyonu için başvuru süreci",
                "Dijital medya projesi geliştirmek istiyorum",
                "Yayıncılık sektörü trendleri nelerdir?",
            ],
            "legal": [
                "Hukuki danışmanlık için avukat arıyorum",
                "İş sözleşmesi hazırlamak istiyorum",
                "Mahkeme süreci nasıl işler?",
                "Dava takibi hizmetleri nelerdir?",
                "Yasa değişiklikleri hakkında bilgi istiyorum",
                "Kanun yorumu nasıl yapılır?",
                "Hukuki görüş alma süreci nedir?",
                "Noter işlemleri nasıl yapılır?",
                "Sözleşme inceleme hizmeti arıyorum",
                "Anlaşma maddeleri nasıl değerlendirilir?",
            ],
            "public_administration": [
                "Belediye hizmetleri hakkında bilgi istiyorum",
                "Kamu kurumu için memur alımı ne zaman?",
                "Valilik başvuru süreci nasıl işler?",
                "Kaymakamlık hizmetleri nelerdir?",
                "Bakanlık projeleri hakkında bilgi",
                "Müdürlük organizasyon yapısı nedir?",
                "Memur ataması süreci nasıl işler?",
                "Kamu hizmeti başvurusu nasıl yapılır?",
                "Resmi evrak işlemleri nelerdir?",
                "Devlet kurumu iletişim bilgileri",
            ],
            "manufacturing": [
                "Fabrikada üretim süreçlerini optimize etmek istiyorum",
                "Kalite kontrol süreçleri nasıl iyileştirilir?",
                "Üretim hattı verimliliği nasıl artırılır?",
                "Montaj süreçleri optimizasyonu",
                "İmalatçı firma seçimi kriterleri nelerdir?",
                "Sanayi tesisleri kurulum süreci",
                "Endüstriyel üretim planlaması",
                "Makine bakım programı nasıl oluşturulur?",
                "Kalite yönetim sistemi kurulumu",
                "Üretim süreçleri analizi nasıl yapılır?",
            ],
            "asset_tracking": [
                "Varlık takibi için RFID sistemi kurmak istiyorum",
                "Envanter yönetimi yazılımı önerileriniz neler?",
                "Stok kontrol sistemi nasıl kurulur?",
                "Barkod sistemi entegrasyonu",
                "Depo yönetimi optimizasyonu",
                "Warehouse otomasyonu hakkında bilgi",
                "Inventory tracking sistemi kurulumu",
                "Asset management yazılımı seçimi",
                "Varlık yönetimi stratejileri",
                "Stok takip süreçleri nasıl iyileştirilir?",
            ],
            "insurance": [
                "Sigorta poliçesi seçerken nelere dikkat etmeliyim?",
                "Hasar tazminatı için başvuru süreci nasıl?",
                "Risk değerlendirmesi nasıl yapılır?",
                "Teminat kapsamı nedir?",
                "Sigorta şirketi seçimi kriterleri",
                "Broker hizmetleri nelerdir?",
                "Aktüer hesaplamaları nasıl yapılır?",
                "Sigortalı hakları nelerdir?",
                "Tazminat hesaplama yöntemleri",
                "Güvence kapsamı analizi",
            ],
            "tourism_hospitality": [
                "Otel rezervasyonu yapmak istiyorum",
                "Restoran için catering hizmeti arıyorum",
                "Seyahat planlaması nasıl yapılır?",
                "Rezervasyon sistemi kurulumu",
                "Tatil paketi seçimi kriterleri",
                "Tur rehberi hizmetleri nelerdir?",
                "Misafir memnuniyeti nasıl artırılır?",
                "Konaklama hizmetleri optimizasyonu",
                "Catering menü planlaması",
                "Turist rehberliği sertifikası nasıl alınır?",
            ],
            "ecommerce": [
                "E-ticaret sitesi kurulumu için adımlar neler?",
                "Online satış platformu için ödeme sistemi entegrasyonu",
                "Alışveriş sepeti optimizasyonu",
                "Ecommerce pazarlama stratejileri",
                "Marketplace kurulum süreci",
                "Dijital satış kanalları nelerdir?",
                "Ürün katalog yönetimi",
                "Kargo entegrasyonu nasıl yapılır?",
                "Teslimat süreçleri optimizasyonu",
                "Online ödeme güvenliği nasıl sağlanır?",
            ],
            "energy": [
                "Güneş enerjisi sistemi kurmak istiyorum",
                "Enerji tasarrufu için önerileriniz neler?",
                "Elektrik üretimi süreçleri",
                "Doğalgaz dağıtım sistemi",
                "Petrol rafineri operasyonları",
                "Yenilenebilir enerji projeleri",
                "Rüzgar enerjisi santrali kurulumu",
                "Hidroelektrik santral operasyonları",
                "Enerji üretimi optimizasyonu",
                "Dağıtım şebekesi yönetimi",
            ],
            "agriculture": [
                "Çiftlik yönetimi için tarımsal danışmanlık arıyorum",
                "Organik tarım sertifikası nasıl alınır?",
                "Ekin yetiştirme teknikleri",
                "Hasat zamanlaması nasıl belirlenir?",
                "Gıda güvenliği standartları",
                "Organik ürün sertifikasyonu",
                "Pestisit kullanımı kuralları",
                "Sulama sistemi kurulumu",
                "Tarımsal üretim planlaması",
                "Çiftçilik teknikleri eğitimi",
            ],
            "transportation": [
                "Toplu taşıma sistemi optimizasyonu için öneriler",
                "Trafik yönetimi için akıllı sistemler",
                "Otobüs hattı planlaması",
                "Metro sistemi operasyonları",
                "Tren seferi optimizasyonu",
                "Taksi hizmetleri yönetimi",
                "Trafik akışı analizi",
                "Yol bakım programı",
                "Köprü geçiş sistemleri",
                "Tünel güvenlik protokolleri",
            ],
            "logistics": [
                "Lojistik süreçlerini optimize etmek istiyorum",
                "Tedarik zinciri yönetimi stratejileri",
                "Kargo takip sistemi kurulumu",
                "Nakliye operasyonları optimizasyonu",
                "Depo yerleşim planlaması",
                "Warehouse otomasyonu",
                "Supply chain analizi",
                "Sevkiyat planlaması",
                "Dağıtım ağı optimizasyonu",
                "Depolama stratejileri",
            ],
            "telecommunications": [
                "Fiber internet altyapısı kurulumu",
                "Mobil iletişim teknolojileri hakkında bilgi",
                "Telefon şebekesi optimizasyonu",
                "Mobil operatör hizmetleri",
                "Fiber optik kablo döşeme",
                "Broadband internet hizmetleri",
                "İletişim altyapısı planlaması",
                "Network güvenlik protokolleri",
                "Telekomünikasyon lisansı",
                "Data center operasyonları",
            ],
            "construction_architecture": [
                "Mimari proje tasarımı için danışmanlık arıyorum",
                "İnşaat projesi yönetimi süreçleri",
                "Yapı tasarımı optimizasyonu",
                "Müteahhit seçimi kriterleri",
                "Mimar hizmetleri nelerdir?",
                "Şantiye yönetimi süreçleri",
                "Yapı yönetimi stratejileri",
                "Konut projesi geliştirme",
                "Ticari bina tasarımı",
                "Altyapı projesi planlaması",
            ],
            "smart_cities": [
                "Akıllı şehir teknolojileri uygulaması",
                "Kentsel altyapı planlaması için öneriler",
                "Smart city proje yönetimi",
                "Kentsel planlama stratejileri",
                "Şehir yönetimi teknolojileri",
                "Akıllı altyapı sistemleri",
                "Kentsel gelişim planlaması",
                "Şehir planlaması optimizasyonu",
                "Akıllı ulaşım sistemleri",
                "Kentsel teknoloji entegrasyonu",
            ],
            "mobility": [
                "Mobilite çözümleri için teknoloji önerileri",
                "Araç paylaşım sistemi kurulumu",
                "Hareketlilik hizmetleri optimizasyonu",
                "Ulaşım teknolojisi entegrasyonu",
                "Mobility platformu geliştirme",
                "Araç paylaşımı operasyonları",
                "Bisiklet yolu planlaması",
                "Yaya dostu şehir tasarımı",
                "Mobilite çözümü geliştirme",
                "Hareketlilik hizmeti yönetimi",
            ],
            "defense_security": [
                "Güvenlik sistemi kurulumu için danışmanlık",
                "Savunma sanayi projeleri hakkında bilgi",
                "Koruma hizmetleri optimizasyonu",
                "Güvenlik sistemi entegrasyonu",
                "Savunma sanayi teknolojileri",
                "Güvenlik hizmeti yönetimi",
                "Koruma protokolleri",
                "Surveillance sistemi kurulumu",
                "Defense teknolojileri",
                "Security protokolleri",
            ],
            "emergency_disaster": [
                "Acil durum müdahale sistemi kurulumu",
                "Afet yönetimi planlaması için öneriler",
                "Kriz iletişimi protokolleri",
                "Acil durum müdahale operasyonları",
                "Afet yönetimi stratejileri",
                "Kriz iletişimi sistemleri",
                "Emergency response planlaması",
                "İtfaiye operasyonları",
                "Ambulans hizmetleri optimizasyonu",
                "Afet yönetimi koordinasyonu",
            ],
        }

        # Get templates for this sector
        templates = query_templates.get(
            sector_id,
            [
                f"{sector_info['name']} sektörü hakkında bilgi istiyorum",
                f"{sector_info['name']} hizmetleri nelerdir?",
                f"{sector_info['name']} sektöründe çalışmak istiyorum",
            ],
        )

        # Generate variations
        queries = []
        for template in templates:
            # Add some variations
            variations = [
                template,
                template + "?",
                template + ".",
                "Merhaba, " + template.lower(),
                "Selam, " + template.lower(),
                "İyi günler, " + template.lower(),
            ]
            queries.extend(variations)

        return queries

    def _generate_sector_responses(
        self, sector_id: str, sector_info: Dict
    ) -> List[str]:
        """Generate responses for a specific sector."""

        # Base response templates
        response_templates = [
            f"Merhaba! {sector_info['name']} sektöründe uzmanlaşmış bir sistemim. Size nasıl yardımcı olabilirim?",
            f"Selam! {sector_info['name']} alanında deneyimli bir danışman olarak size rehberlik edebilirim.",
            f"İyi günler! {sector_info['name']} sektörü hakkında detaylı bilgi verebilirim.",
            f"Merhaba! {sector_info['name']} konusunda uzman bir sistemim ve size yardımcı olmaktan memnuniyet duyarım.",
            f"Selam! {sector_info['name']} sektöründe çalışan bir uzman olarak sorularınızı yanıtlayabilirim.",
            f"İyi günler! {sector_info['name']} alanında size en iyi şekilde yardımcı olmaya çalışacağım.",
            f"Merhaba! {sector_info['name']} sektörü hakkında bilgi almak istediğiniz konuları detaylandırabilir misiniz?",
            f"Selam! {sector_info['name']} konusunda size nasıl yardımcı olabilirim?",
            f"İyi günler! {sector_info['name']} sektöründe uzmanlaşmış bir sistemim.",
            f"Merhaba! {sector_info['name']} hakkında sorularınızı yanıtlamaya hazırım.",
        ]

        return response_templates

    def generate_sector_data(
        self, sector_id: str, num_examples: int = 100
    ) -> List[Dict]:
        """Generate training data for a specific sector."""

        if sector_id not in self.sectors:
            raise ValueError(f"Sector {sector_id} not found in configuration")

        sector_info = self.sectors[sector_id]
        logger.info(f"Generating data for sector: {sector_info['name']}")

        # Generate queries and responses
        queries = self._generate_sector_queries(sector_id, sector_info)
        responses = self._generate_sector_responses(sector_id, sector_info)

        # Create training examples
        training_data = []

        for i in range(num_examples):
            # Randomly select query and response
            query = random.choice(queries)
            response = random.choice(responses)

            # Create training example
            example = {
                "id": f"{sector_id}_{i:04d}",
                "sector_id": sector_id,
                "sector_name": sector_info["name"],
                "query": query,
                "response": response,
                "full_text": f"{query} {response}",
                "keywords": sector_info["keywords"],
            }

            training_data.append(example)

        return training_data

    def generate_all_data(
        self, output_dir: str = "data", examples_per_sector: int = 100
    ):
        """Generate training data for all sectors."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_data = {}
        total_examples = 0

        for sector_id, sector_info in self.sectors.items():
            try:
                logger.info(f"Generating data for {sector_info['name']}...")

                sector_data = self.generate_sector_data(sector_id, examples_per_sector)
                all_data[sector_id] = sector_data
                total_examples += len(sector_data)

                # Save individual sector data
                sector_file = output_path / f"{sector_id}_data.json"
                with open(sector_file, "w", encoding="utf-8") as f:
                    json.dump(sector_data, f, indent=2, ensure_ascii=False)

                logger.info(
                    f"✅ Generated {len(sector_data)} examples for {sector_info['name']}"
                )

            except Exception as e:
                logger.error(f"Failed to generate data for {sector_id}: {e}")

        # Save combined data
        combined_file = output_path / "all_sectors_data.json"
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        # Save summary
        summary = {
            "total_sectors": len(self.sectors),
            "total_examples": total_examples,
            "examples_per_sector": examples_per_sector,
            "sectors": {
                sector_id: {
                    "name": sector_info["name"],
                    "examples": len(all_data.get(sector_id, [])),
                }
                for sector_id, sector_info in self.sectors.items()
            },
        }

        summary_file = output_path / "data_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(
            f"✅ Generated {total_examples} total examples across {len(self.sectors)} sectors"
        )
        logger.info(f"✅ Data saved to: {output_path}")

        return all_data


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate Turkish sector-specific training data"
    )
    parser.add_argument(
        "--sector", type=str, help="Specific sector to generate data for"
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate data for all sectors"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Output directory"
    )
    parser.add_argument(
        "--examples", type=int, default=100, help="Number of examples per sector"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = TurkishDataGenerator()

    if args.all:
        # Generate data for all sectors
        generator.generate_all_data(args.output_dir, args.examples)

    elif args.sector:
        # Generate data for specific sector
        if args.sector not in generator.sectors:
            print(f"Error: Sector '{args.sector}' not found")
            print("Available sectors:")
            for sector_id, sector_info in generator.sectors.items():
                print(f"  {sector_id}: {sector_info['name']}")
            return

        sector_data = generator.generate_sector_data(args.sector, args.examples)

        # Save to file
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"{args.sector}_data.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sector_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Generated {len(sector_data)} examples for {args.sector}")
        print(f"✅ Data saved to: {output_file}")

    else:
        print("Please specify --sector or --all")
        print("Available sectors:")
        for sector_id, sector_info in generator.sectors.items():
            print(f"  {sector_id}: {sector_info['name']}")


if __name__ == "__main__":
    main()
