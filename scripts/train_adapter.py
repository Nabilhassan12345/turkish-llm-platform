#!/usr/bin/env python3
"""
Turkish LLM Adapter Training Script
Trains sector-specific adapters for the 22 Turkish business sectors.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
import yaml
from datasets import Dataset as HFDataset
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TurkishSectorDataset(Dataset):
    """Dataset for Turkish sector-specific training data."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


class TurkishAdapterTrainer:
    """Trainer for Turkish sector-specific adapters."""

    def __init__(self, config_path: str = "configs/sectors.yaml"):
        self.config_path = config_path
        self.sectors = self._load_sectors()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def _load_sectors(self) -> Dict:
        """Load sector configuration."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config["sectors"]

    def _load_base_model(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Load base model and tokenizer."""
        logger.info(f"Loading base model: {model_name}")

        # Quantization config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )

        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure LoRA adapter
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"],
        )

        # Add LoRA adapter to the model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model, tokenizer

    def _generate_sector_data(self, sector_id: str, sector_info: Dict) -> List[str]:
        """Generate training data for a specific sector."""
        logger.info(f"Generating training data for sector: {sector_info['name']}")

        # Base prompts for each sector
        base_prompts = {
            "finance_banking": [
                "Banka kredisi almak istiyorum. ",
                "Yatırım danışmanlığı hakkında bilgi istiyorum. ",
                "Faiz oranları nasıl hesaplanır? ",
                "Portföy yönetimi stratejileri nelerdir? ",
                "Kredi kartı başvurusu nasıl yapılır? ",
                "Döviz kuru tahminleri nasıl yapılır? ",
                "Borsa yatırımı için önerileriniz neler? ",
                "Mevduat hesabı açmak istiyorum. ",
                "Kredi notu nasıl yükseltilir? ",
                "Finansal planlama için danışmanlık arıyorum. ",
            ],
            "healthcare": [
                "Hastane randevusu almak istiyorum. ",
                "Doktor muayenesi için ne yapmam gerekiyor? ",
                "İlaç kullanımı hakkında bilgi istiyorum. ",
                "Tedavi süreci nasıl işler? ",
                "Sağlık sigortası kapsamı nedir? ",
                "Laboratuvar sonuçlarımı nasıl anlayabilirim? ",
                "Ameliyat öncesi hazırlık süreci nedir? ",
                "Klinik randevusu nasıl alınır? ",
                "Radyoloji raporu ne anlama geliyor? ",
                "Eczane hizmetleri hakkında bilgi istiyorum. ",
            ],
            "education": [
                "Üniversite sınavına hazırlanıyorum. ",
                "Kurs seçimi nasıl yapılır? ",
                "Eğitim danışmanlığı hizmetleri nelerdir? ",
                "Öğretmen ataması süreci nasıl işler? ",
                "Öğrenci bursu başvurusu nasıl yapılır? ",
                "Ders programı nasıl oluşturulur? ",
                "Sınav hazırlık stratejileri nelerdir? ",
                "Akademi programları hakkında bilgi istiyorum. ",
                "Seminer kayıtları nasıl yapılır? ",
                "Workshop etkinlikleri ne zaman düzenlenir? ",
            ],
            "media_publishing": [
                "Gazete için muhabir arıyoruz. ",
                "Dergi editörü pozisyonu açık mı? ",
                "Televizyon programı yapımı hakkında bilgi. ",
                "Radyo yayını için teknik destek arıyoruz. ",
                "Medya kuruluşu kurmak istiyorum. ",
                "Yayın lisansı nasıl alınır? ",
                "Haber editörü iş tanımı nedir? ",
                "Reporter pozisyonu için başvuru süreci. ",
                "Dijital medya projesi geliştirmek istiyorum. ",
                "Yayıncılık sektörü trendleri nelerdir? ",
            ],
            "legal": [
                "Hukuki danışmanlık için avukat arıyorum. ",
                "İş sözleşmesi hazırlamak istiyorum. ",
                "Mahkeme süreci nasıl işler? ",
                "Dava takibi hizmetleri nelerdir? ",
                "Yasa değişiklikleri hakkında bilgi istiyorum. ",
                "Kanun yorumu nasıl yapılır? ",
                "Hukuki görüş alma süreci nedir? ",
                "Noter işlemleri nasıl yapılır? ",
                "Sözleşme inceleme hizmeti arıyorum. ",
                "Anlaşma maddeleri nasıl değerlendirilir? ",
            ],
            "public_administration": [
                "Belediye hizmetleri hakkında bilgi istiyorum. ",
                "Kamu kurumu için memur alımı ne zaman? ",
                "Valilik başvuru süreci nasıl işler? ",
                "Kaymakamlık hizmetleri nelerdir? ",
                "Bakanlık projeleri hakkında bilgi. ",
                "Müdürlük organizasyon yapısı nedir? ",
                "Memur ataması süreci nasıl işler? ",
                "Kamu hizmeti başvurusu nasıl yapılır? ",
                "Resmi evrak işlemleri nelerdir? ",
                "Devlet kurumu iletişim bilgileri. ",
            ],
            "manufacturing": [
                "Fabrikada üretim süreçlerini optimize etmek istiyorum. ",
                "Kalite kontrol süreçleri nasıl iyileştirilir? ",
                "Üretim hattı verimliliği nasıl artırılır? ",
                "Montaj süreçleri optimizasyonu. ",
                "İmalatçı firma seçimi kriterleri nelerdir? ",
                "Sanayi tesisleri kurulum süreci. ",
                "Endüstriyel üretim planlaması. ",
                "Makine bakım programı nasıl oluşturulur? ",
                "Kalite yönetim sistemi kurulumu. ",
                "Üretim süreçleri analizi nasıl yapılır? ",
            ],
            "asset_tracking": [
                "Varlık takibi için RFID sistemi kurmak istiyorum. ",
                "Envanter yönetimi yazılımı önerileriniz neler? ",
                "Stok kontrol sistemi nasıl kurulur? ",
                "Barkod sistemi entegrasyonu. ",
                "Depo yönetimi optimizasyonu. ",
                "Warehouse otomasyonu hakkında bilgi. ",
                "Inventory tracking sistemi kurulumu. ",
                "Asset management yazılımı seçimi. ",
                "Varlık yönetimi stratejileri. ",
                "Stok takip süreçleri nasıl iyileştirilir? ",
            ],
            "insurance": [
                "Sigorta poliçesi seçerken nelere dikkat etmeliyim? ",
                "Hasar tazminatı için başvuru süreci nasıl? ",
                "Risk değerlendirmesi nasıl yapılır? ",
                "Teminat kapsamı nedir? ",
                "Sigorta şirketi seçimi kriterleri. ",
                "Broker hizmetleri nelerdir? ",
                "Aktüer hesaplamaları nasıl yapılır? ",
                "Sigortalı hakları nelerdir? ",
                "Tazminat hesaplama yöntemleri. ",
                "Güvence kapsamı analizi. ",
            ],
            "tourism_hospitality": [
                "Otel rezervasyonu yapmak istiyorum. ",
                "Restoran için catering hizmeti arıyorum. ",
                "Seyahat planlaması nasıl yapılır? ",
                "Rezervasyon sistemi kurulumu. ",
                "Tatil paketi seçimi kriterleri. ",
                "Tur rehberi hizmetleri nelerdir? ",
                "Misafir memnuniyeti nasıl artırılır? ",
                "Konaklama hizmetleri optimizasyonu. ",
                "Catering menü planlaması. ",
                "Turist rehberliği sertifikası nasıl alınır? ",
            ],
            "ecommerce": [
                "E-ticaret sitesi kurulumu için adımlar neler? ",
                "Online satış platformu için ödeme sistemi entegrasyonu. ",
                "Alışveriş sepeti optimizasyonu. ",
                "Ecommerce pazarlama stratejileri. ",
                "Marketplace kurulum süreci. ",
                "Dijital satış kanalları nelerdir? ",
                "Ürün katalog yönetimi. ",
                "Kargo entegrasyonu nasıl yapılır? ",
                "Teslimat süreçleri optimizasyonu. ",
                "Online ödeme güvenliği nasıl sağlanır? ",
            ],
            "energy": [
                "Güneş enerjisi sistemi kurmak istiyorum. ",
                "Enerji tasarrufu için önerileriniz neler? ",
                "Elektrik üretimi süreçleri. ",
                "Doğalgaz dağıtım sistemi. ",
                "Petrol rafineri operasyonları. ",
                "Yenilenebilir enerji projeleri. ",
                "Rüzgar enerjisi santrali kurulumu. ",
                "Hidroelektrik santral operasyonları. ",
                "Enerji üretimi optimizasyonu. ",
                "Dağıtım şebekesi yönetimi. ",
            ],
            "agriculture": [
                "Çiftlik yönetimi için tarımsal danışmanlık arıyorum. ",
                "Organik tarım sertifikası nasıl alınır? ",
                "Ekin yetiştirme teknikleri. ",
                "Hasat zamanlaması nasıl belirlenir? ",
                "Gıda güvenliği standartları. ",
                "Organik ürün sertifikasyonu. ",
                "Pestisit kullanımı kuralları. ",
                "Sulama sistemi kurulumu. ",
                "Tarımsal üretim planlaması. ",
                "Çiftçilik teknikleri eğitimi. ",
            ],
            "transportation": [
                "Toplu taşıma sistemi optimizasyonu için öneriler. ",
                "Trafik yönetimi için akıllı sistemler. ",
                "Otobüs hattı planlaması. ",
                "Metro sistemi operasyonları. ",
                "Tren seferi optimizasyonu. ",
                "Taksi hizmetleri yönetimi. ",
                "Trafik akışı analizi. ",
                "Yol bakım programı. ",
                "Köprü geçiş sistemleri. ",
                "Tünel güvenlik protokolleri. ",
            ],
            "logistics": [
                "Lojistik süreçlerini optimize etmek istiyorum. ",
                "Tedarik zinciri yönetimi stratejileri. ",
                "Kargo takip sistemi kurulumu. ",
                "Nakliye operasyonları optimizasyonu. ",
                "Depo yerleşim planlaması. ",
                "Warehouse otomasyonu. ",
                "Supply chain analizi. ",
                "Sevkiyat planlaması. ",
                "Dağıtım ağı optimizasyonu. ",
                "Depolama stratejileri. ",
            ],
            "telecommunications": [
                "Fiber internet altyapısı kurulumu. ",
                "Mobil iletişim teknolojileri hakkında bilgi. ",
                "Telefon şebekesi optimizasyonu. ",
                "Mobil operatör hizmetleri. ",
                "Fiber optik kablo döşeme. ",
                "Broadband internet hizmetleri. ",
                "İletişim altyapısı planlaması. ",
                "Network güvenlik protokolleri. ",
                "Telekomünikasyon lisansı. ",
                "Data center operasyonları. ",
            ],
            "construction_architecture": [
                "Mimari proje tasarımı için danışmanlık arıyorum. ",
                "İnşaat projesi yönetimi süreçleri. ",
                "Yapı tasarımı optimizasyonu. ",
                "Müteahhit seçimi kriterleri. ",
                "Mimar hizmetleri nelerdir? ",
                "Şantiye yönetimi süreçleri. ",
                "Yapı yönetimi stratejileri. ",
                "Konut projesi geliştirme. ",
                "Ticari bina tasarımı. ",
                "Altyapı projesi planlaması. ",
            ],
            "smart_cities": [
                "Akıllı şehir teknolojileri uygulaması. ",
                "Kentsel altyapı planlaması için öneriler. ",
                "Smart city proje yönetimi. ",
                "Kentsel planlama stratejileri. ",
                "Şehir yönetimi teknolojileri. ",
                "Akıllı altyapı sistemleri. ",
                "Kentsel gelişim planlaması. ",
                "Şehir planlaması optimizasyonu. ",
                "Akıllı ulaşım sistemleri. ",
                "Kentsel teknoloji entegrasyonu. ",
            ],
            "mobility": [
                "Mobilite çözümleri için teknoloji önerileri. ",
                "Araç paylaşım sistemi kurulumu. ",
                "Hareketlilik hizmetleri optimizasyonu. ",
                "Ulaşım teknolojisi entegrasyonu. ",
                "Mobility platformu geliştirme. ",
                "Araç paylaşımı operasyonları. ",
                "Bisiklet yolu planlaması. ",
                "Yaya dostu şehir tasarımı. ",
                "Mobilite çözümü geliştirme. ",
                "Hareketlilik hizmeti yönetimi. ",
            ],
            "defense_security": [
                "Güvenlik sistemi kurulumu için danışmanlık. ",
                "Savunma sanayi projeleri hakkında bilgi. ",
                "Koruma hizmetleri optimizasyonu. ",
                "Güvenlik sistemi entegrasyonu. ",
                "Savunma sanayi teknolojileri. ",
                "Güvenlik hizmeti yönetimi. ",
                "Koruma protokolleri. ",
                "Surveillance sistemi kurulumu. ",
                "Defense teknolojileri. ",
                "Security protokolleri. ",
            ],
            "emergency_disaster": [
                "Acil durum müdahale sistemi kurulumu. ",
                "Afet yönetimi planlaması için öneriler. ",
                "Kriz iletişimi protokolleri. ",
                "Acil durum müdahale operasyonları. ",
                "Afet yönetimi stratejileri. ",
                "Kriz iletişimi sistemleri. ",
                "Emergency response planlaması. ",
                "İtfaiye operasyonları. ",
                "Ambulans hizmetleri optimizasyonu. ",
                "Afet yönetimi koordinasyonu. ",
            ],
        }

        # Get prompts for this sector
        prompts = base_prompts.get(
            sector_id,
            [
                f"{sector_info['name']} sektörü hakkında bilgi istiyorum. ",
                f"{sector_info['name']} hizmetleri nelerdir? ",
                f"{sector_info['name']} sektöründe çalışmak istiyorum. ",
            ],
        )

        # Generate responses for each prompt
        training_data = []
        for prompt in prompts:
            # Create training examples with different response styles
            responses = [
                f"{prompt}Bu konuda size yardımcı olabilirim. {sector_info['name']} sektöründe uzmanlaşmış bir sistemim.",
                f"{prompt}Merhaba! {sector_info['name']} alanında deneyimli bir danışman olarak size rehberlik edebilirim.",
                f"{prompt}Tabii ki! {sector_info['name']} sektörü hakkında detaylı bilgi verebilirim.",
                f"{prompt}Elbette! {sector_info['name']} konusunda uzman bir sistemim ve size yardımcı olmaktan memnuniyet duyarım.",
            ]

            for response in responses:
                training_data.append(f"{prompt}{response}")

        return training_data

    def train_sector_adapter(
        self,
        sector_id: str,
        output_dir: str = "adapters",
        model_name: str = "microsoft/DialoGPT-medium",
        num_epochs: int = 3,
        batch_size: int = 4,
    ):
        """Train adapter for a specific sector."""

        if sector_id not in self.sectors:
            raise ValueError(f"Sector {sector_id} not found in configuration")

        sector_info = self.sectors[sector_id]
        logger.info(f"Training adapter for sector: {sector_info['name']}")

        # Load model and tokenizer
        model, tokenizer = self._load_base_model(model_name)

        # Generate training data
        training_texts = self._generate_sector_data(sector_id, sector_info)
        logger.info(f"Generated {len(training_texts)} training examples")

        # Create dataset
        dataset = TurkishSectorDataset(training_texts, tokenizer)

        # Create output directory
        adapter_path = Path(output_dir) / f"{sector_id}_adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(adapter_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=100,
            save_total_limit=2,
            logging_steps=10,
            learning_rate=5e-5,
            warmup_steps=100,
            weight_decay=0.01,
            fp16=True,
            gradient_accumulation_steps=4,
            dataloader_pin_memory=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save model and tokenizer
        trainer.save_model()
        tokenizer.save_pretrained(str(adapter_path))

        # Save sector info
        sector_metadata = {
            "sector_id": sector_id,
            "sector_name": sector_info["name"],
            "description": sector_info["description"],
            "keywords": sector_info["keywords"],
            "training_examples": len(training_texts),
            "model_name": model_name,
            "training_args": training_args.to_dict(),
        }

        with open(adapter_path / "sector_metadata.json", "w", encoding="utf-8") as f:
            json.dump(sector_metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Adapter saved to: {adapter_path}")
        return str(adapter_path)

    def train_all_adapters(
        self,
        output_dir: str = "adapters",
        model_name: str = "microsoft/DialoGPT-medium",
        num_epochs: int = 3,
        batch_size: int = 4,
    ):
        """Train adapters for all sectors."""
        logger.info(f"Training adapters for all {len(self.sectors)} sectors")

        results = {}
        for sector_id, sector_info in self.sectors.items():
            try:
                logger.info(f"Training adapter for {sector_info['name']}...")
                adapter_path = self.train_sector_adapter(
                    sector_id, output_dir, model_name, num_epochs, batch_size
                )
                results[sector_id] = {
                    "status": "success",
                    "path": adapter_path,
                    "name": sector_info["name"],
                }
            except Exception as e:
                logger.error(f"Failed to train adapter for {sector_id}: {e}")
                results[sector_id] = {
                    "status": "failed",
                    "error": str(e),
                    "name": sector_info["name"],
                }

        # Save training summary
        summary_path = Path(output_dir) / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Training summary saved to: {summary_path}")
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train Turkish sector-specific adapters"
    )
    parser.add_argument("--sector", type=str, help="Specific sector to train")
    parser.add_argument("--all", action="store_true", help="Train all sectors")
    parser.add_argument(
        "--output-dir", type=str, default="adapters", help="Output directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="Base model name",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")

    args = parser.parse_args()

    # Initialize trainer
    trainer = TurkishAdapterTrainer()

    if args.all:
        # Train all adapters
        results = trainer.train_all_adapters(
            args.output_dir, args.model_name, args.epochs, args.batch_size
        )

        # Print summary
        print("\n=== Training Summary ===")
        for sector_id, result in results.items():
            status = "✅" if result["status"] == "success" else "❌"
            print(f"{status} {result['name']}: {result['status']}")

    elif args.sector:
        # Train specific sector
        adapter_path = trainer.train_sector_adapter(
            args.sector, args.output_dir, args.model_name, args.epochs, args.batch_size
        )
        print(f"✅ Adapter trained successfully: {adapter_path}")

    else:
        print("Please specify --sector or --all")
        print("Available sectors:")
        for sector_id, sector_info in trainer.sectors.items():
            print(f"  {sector_id}: {sector_info['name']}")


if __name__ == "__main__":
    main()
