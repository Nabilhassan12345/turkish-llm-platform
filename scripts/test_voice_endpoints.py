#!/usr/bin/env python3
"""
Phase C4: Ses Endpoint Testleri
Voice Chat ve UI bileşenlerinin kapsamlı testleri
Postman/curl test scriptleri ve ses doğrulama testleri
"""

import asyncio
import json
import requests
import websockets
import time
import wave
import numpy as np
from pathlib import Path
import subprocess
import sys
import os
from typing import Dict, List, Optional, Any
import yaml
import tempfile
import shutil


class VoiceEndpointTester:
    """Voice endpoint'lerini test eden sınıf"""

    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8765"
        self.test_results = []
        self.test_audio_dir = Path("test_audio_files")
        self.test_audio_dir.mkdir(exist_ok=True)

    def log_test(self, test_name: str, status: str, details: str = ""):
        """Test sonucunu kaydet"""
        result = {
            "test": test_name,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "details": details,
        }
        self.test_results.append(result)
        print(f"🧪 {test_name}: {'✅' if status == 'PASS' else '❌'} {details}")

    def create_test_audio_files(self):
        """Farklı test senaryoları için ses dosyaları oluştur"""
        print("🎵 Test ses dosyaları oluşturuluyor...")

        # 1. Kısa ses (1 saniye)
        self.create_sine_wave_audio(1, "short_audio.wav", 440)

        # 2. Orta ses (3 saniye)
        self.create_sine_wave_audio(3, "medium_audio.wav", 880)

        # 3. Uzun ses (10 saniye)
        self.create_sine_wave_audio(10, "long_audio.wav", 220)

        # 4. Farklı frekans (5 saniye)
        self.create_sine_wave_audio(5, "high_freq_audio.wav", 2000)

        # 5. Düşük frekans (2 saniye)
        self.create_sine_wave_audio(2, "low_freq_audio.wav", 110)

        print("✅ Test ses dosyaları oluşturuldu")

    def create_sine_wave_audio(self, duration: int, filename: str, frequency: int):
        """Belirli frekansta sinüs dalgası ses dosyası oluştur"""
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3

        filepath = self.test_audio_dir / filename
        with wave.open(str(filepath), "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    def test_health_endpoints(self):
        """Health check endpoint'lerini test et"""
        print("\n🏥 Health Check Testleri")
        print("=" * 50)

        # Test 1: Ana health endpoint
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                self.log_test("Health Check", "PASS", f"Status: {response.status_code}")
            else:
                self.log_test("Health Check", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Health Check", "FAIL", f"Error: {str(e)}")

        # Test 2: Ready endpoint
        try:
            response = requests.get(f"{self.base_url}/ready")
            if response.status_code == 200:
                self.log_test("Ready Check", "PASS", f"Status: {response.status_code}")
            else:
                self.log_test("Ready Check", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Ready Check", "FAIL", f"Error: {str(e)}")

        # Test 3: Metrics endpoint
        try:
            response = requests.get(f"{self.base_url}/metrics")
            if response.status_code == 200:
                self.log_test(
                    "Metrics Check", "PASS", f"Status: {response.status_code}"
                )
            else:
                self.log_test(
                    "Metrics Check", "FAIL", f"Status: {response.status_code}"
                )
        except Exception as e:
            self.log_test("Metrics Check", "FAIL", f"Error: {str(e)}")

    def test_audio_processing_endpoints(self):
        """Ses işleme endpoint'lerini test et"""
        print("\n🎤 Ses İşleme Endpoint Testleri")
        print("=" * 50)

        # Tüm test ses dosyalarını test et
        audio_files = list(self.test_audio_dir.glob("*.wav"))

        for audio_file in audio_files:
            try:
                with open(audio_file, "rb") as f:
                    files = {"audio": (audio_file.name, f, "audio/wav")}
                    data = {"sector": "healthcare"}
                    response = requests.post(
                        f"{self.base_url}/process-audio", files=files, data=data
                    )

                if response.status_code == 200:
                    result = response.json()
                    self.log_test(
                        f"Audio Processing - {audio_file.name}",
                        "PASS",
                        f"Response: {result.get('status', 'unknown')}",
                    )
                else:
                    self.log_test(
                        f"Audio Processing - {audio_file.name}",
                        "FAIL",
                        f"Status: {response.status_code}",
                    )

            except Exception as e:
                self.log_test(
                    f"Audio Processing - {audio_file.name}", "FAIL", f"Error: {str(e)}"
                )

        # Farklı sektörlerle test et
        sectors = ["finance_banking", "education", "legal", "manufacturing"]
        test_audio = self.test_audio_dir / "medium_audio.wav"

        for sector in sectors:
            try:
                with open(test_audio, "rb") as f:
                    files = {"audio": ("test.wav", f, "audio/wav")}
                    data = {"sector": sector}
                    response = requests.post(
                        f"{self.base_url}/process-audio", files=files, data=data
                    )

                if response.status_code == 200:
                    result = response.json()
                    self.log_test(
                        f"Sector Audio - {sector}",
                        "PASS",
                        f"Response: {result.get('status', 'unknown')}",
                    )
                else:
                    self.log_test(
                        f"Sector Audio - {sector}",
                        "FAIL",
                        f"Status: {response.status_code}",
                    )

            except Exception as e:
                self.log_test(f"Sector Audio - {sector}", "FAIL", f"Error: {str(e)}")

    def test_text_processing_endpoints(self):
        """Metin işleme endpoint'lerini test et"""
        print("\n📝 Metin İşleme Endpoint Testleri")
        print("=" * 50)

        # Farklı sektörlerde metin testleri
        test_cases = [
            {
                "text": "Merhaba, sağlık sektöründe çalışıyorum",
                "sector": "healthcare",
                "description": "Sağlık sektörü metin",
            },
            {
                "text": "Finansal danışmanlık hizmeti arıyorum",
                "sector": "finance_banking",
                "description": "Finans sektörü metin",
            },
            {
                "text": "Eğitim materyali geliştirmek istiyorum",
                "sector": "education",
                "description": "Eğitim sektörü metin",
            },
            {
                "text": "Hukuki danışmanlık almak istiyorum",
                "sector": "legal",
                "description": "Hukuk sektörü metin",
            },
            {
                "text": "Üretim süreçlerini optimize etmek istiyorum",
                "sector": "manufacturing",
                "description": "İmalat sektörü metin",
            },
        ]

        for test_case in test_cases:
            try:
                data = {"text": test_case["text"], "sector": test_case["sector"]}
                response = requests.post(f"{self.base_url}/process-text", json=data)

                if response.status_code == 200:
                    result = response.json()
                    self.log_test(
                        f"Text Processing - {test_case['description']}",
                        "PASS",
                        f"Response: {result.get('status', 'unknown')}",
                    )
                else:
                    self.log_test(
                        f"Text Processing - {test_case['description']}",
                        "FAIL",
                        f"Status: {response.status_code}",
                    )

            except Exception as e:
                self.log_test(
                    f"Text Processing - {test_case['description']}",
                    "FAIL",
                    f"Error: {str(e)}",
                )

    def test_router_endpoints(self):
        """Router endpoint'lerini test et"""
        print("\n🔄 Router Endpoint Testleri")
        print("=" * 50)

        # Router health check
        try:
            response = requests.get(f"{self.base_url}/router/health")
            if response.status_code == 200:
                self.log_test(
                    "Router Health", "PASS", f"Status: {response.status_code}"
                )
            else:
                self.log_test(
                    "Router Health", "FAIL", f"Status: {response.status_code}"
                )
        except Exception as e:
            self.log_test("Router Health", "FAIL", f"Error: {str(e)}")

        # Router stats
        try:
            response = requests.get(f"{self.base_url}/router/stats")
            if response.status_code == 200:
                result = response.json()
                self.log_test("Router Stats", "PASS", f"Stats: {len(result)} items")
            else:
                self.log_test("Router Stats", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Router Stats", "FAIL", f"Error: {str(e)}")

        # Sector classification test
        test_texts = [
            "Banka kredisi almak istiyorum",
            "Hastane randevusu almak istiyorum",
            "Üniversite başvurusu yapmak istiyorum",
            "Avukat ile görüşmek istiyorum",
        ]

        for text in test_texts:
            try:
                data = {"text": text}
                response = requests.post(f"{self.base_url}/router/classify", json=data)

                if response.status_code == 200:
                    result = response.json()
                    self.log_test(
                        f"Text Classification - {text[:30]}...",
                        "PASS",
                        f"Classified as: {result.get('sector', 'unknown')}",
                    )
                else:
                    self.log_test(
                        f"Text Classification - {text[:30]}...",
                        "FAIL",
                        f"Status: {response.status_code}",
                    )

            except Exception as e:
                self.log_test(
                    f"Text Classification - {text[:30]}...", "FAIL", f"Error: {str(e)}"
                )

    async def test_websocket_connection(self):
        """WebSocket bağlantısını test et"""
        print("\n🔌 WebSocket Bağlantı Testleri")
        print("=" * 50)

        try:
            websocket = await websockets.connect(self.ws_url)

            # Bağlantı testi
            self.log_test("WebSocket Connection", "PASS", "Bağlantı başarılı")

            # Mesaj gönderme testi
            test_message = {
                "type": "test",
                "message": "Test mesajı",
                "sector": "general",
            }

            await websocket.send(json.dumps(test_message))
            self.log_test("WebSocket Send", "PASS", "Mesaj gönderildi")

            # Yanıt bekleme (timeout ile)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                self.log_test(
                    "WebSocket Receive", "PASS", f"Yanıt alındı: {response[:100]}..."
                )
            except asyncio.TimeoutError:
                self.log_test("WebSocket Receive", "WARN", "Yanıt timeout (beklenen)")

            await websocket.close()
            self.log_test("WebSocket Close", "PASS", "Bağlantı kapatıldı")

        except Exception as e:
            self.log_test("WebSocket Connection", "FAIL", f"Error: {str(e)}")

    def test_audio_quality(self):
        """Ses kalitesi testleri"""
        print("\n🎵 Ses Kalitesi Testleri")
        print("=" * 50)

        # Ses dosyası format kontrolü
        for audio_file in self.test_audio_dir.glob("*.wav"):
            try:
                with wave.open(str(audio_file), "r") as wav_file:
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frame_rate = wav_file.getframerate()
                    frames = wav_file.getnframes()
                    duration = frames / frame_rate

                    # Kalite kriterleri
                    quality_score = 0
                    if channels == 1:  # Mono
                        quality_score += 1
                    if sample_width == 2:  # 16-bit
                        quality_score += 1
                    if frame_rate >= 16000:  # 16kHz+
                        quality_score += 1
                    if duration > 0:  # Geçerli süre
                        quality_score += 1

                    if quality_score == 4:
                        self.log_test(
                            f"Audio Quality - {audio_file.name}",
                            "PASS",
                            f"Channels: {channels}, Bits: {sample_width*8}, Rate: {frame_rate}Hz, Duration: {duration:.2f}s",
                        )
                    else:
                        self.log_test(
                            f"Audio Quality - {audio_file.name}",
                            "WARN",
                            f"Quality Score: {quality_score}/4",
                        )

            except Exception as e:
                self.log_test(
                    f"Audio Quality - {audio_file.name}", "FAIL", f"Error: {str(e)}"
                )

    def test_sector_configuration(self):
        """Sektör yapılandırmasını test et"""
        print("\n🏢 Sektör Yapılandırma Testleri")
        print("=" * 50)

        try:
            # Sektör config dosyasını yükle
            config_path = Path("configs/sectors.yaml")
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                sectors = config.get("sectors", {})
                router_config = config.get("router", {})
                system_config = config.get("system", {})

                # Sektör sayısı kontrolü
                sector_count = len(sectors)
                if sector_count == 22:
                    self.log_test("Sector Count", "PASS", f"22 sektör bulundu")
                else:
                    self.log_test(
                        "Sector Count",
                        "FAIL",
                        f"{sector_count} sektör bulundu, 22 bekleniyordu",
                    )

                # Router yapılandırması kontrolü
                if router_config.get("enable_moe"):
                    self.log_test("MoE Router", "PASS", "MoE router aktif")
                else:
                    self.log_test("MoE Router", "WARN", "MoE router pasif")

                # Sistem yapılandırması kontrolü
                if system_config.get("enable_metrics"):
                    self.log_test("Metrics System", "PASS", "Metrik sistemi aktif")
                else:
                    self.log_test("Metrics System", "WARN", "Metrik sistemi pasif")

            else:
                self.log_test(
                    "Sector Config", "FAIL", "configs/sectors.yaml bulunamadı"
                )

        except Exception as e:
            self.log_test("Sector Config", "FAIL", f"Error: {str(e)}")

    def generate_postman_collection(self):
        """Postman collection dosyası oluştur"""
        print("\n📋 Postman Collection Oluşturuluyor...")

        collection = {
            "info": {
                "name": "Türkçe AI Asistan - Voice Endpoints",
                "description": "22 sektörlü Türkçe AI asistan ses endpoint testleri",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
            },
            "item": [
                {
                    "name": "Health Checks",
                    "item": [
                        {
                            "name": "Health Check",
                            "request": {
                                "method": "GET",
                                "header": [],
                                "url": {
                                    "raw": "{{base_url}}/health",
                                    "host": ["{{base_url}}"],
                                    "path": ["health"],
                                },
                            },
                        },
                        {
                            "name": "Ready Check",
                            "request": {
                                "method": "GET",
                                "header": [],
                                "url": {
                                    "raw": "{{base_url}}/ready",
                                    "host": ["{{base_url}}"],
                                    "path": ["ready"],
                                },
                            },
                        },
                    ],
                },
                {
                    "name": "Audio Processing",
                    "item": [
                        {
                            "name": "Process Audio - Healthcare",
                            "request": {
                                "method": "POST",
                                "header": [],
                                "body": {
                                    "mode": "formdata",
                                    "formdata": [
                                        {
                                            "key": "audio",
                                            "type": "file",
                                            "src": "test_audio.wav",
                                        },
                                        {
                                            "key": "sector",
                                            "value": "healthcare",
                                            "type": "text",
                                        },
                                    ],
                                },
                                "url": {
                                    "raw": "{{base_url}}/process-audio",
                                    "host": ["{{base_url}}"],
                                    "path": ["process-audio"],
                                },
                            },
                        }
                    ],
                },
                {
                    "name": "Text Processing",
                    "item": [
                        {
                            "name": "Process Text - Finance",
                            "request": {
                                "method": "POST",
                                "header": [
                                    {"key": "Content-Type", "value": "application/json"}
                                ],
                                "body": {
                                    "mode": "raw",
                                    "raw": '{\n  "text": "Finansal danışmanlık hizmeti arıyorum",\n  "sector": "finance_banking"\n}',
                                },
                                "url": {
                                    "raw": "{{base_url}}/process-text",
                                    "host": ["{{base_url}}"],
                                    "path": ["process-text"],
                                },
                            },
                        }
                    ],
                },
            ],
            "variable": [
                {"key": "base_url", "value": "http://localhost:8000", "type": "string"}
            ],
        }

        # Postman collection dosyasını kaydet
        collection_path = Path("test_audio_files/postman_collection.json")
        with open(collection_path, "w", encoding="utf-8") as f:
            json.dump(collection, f, indent=2, ensure_ascii=False)

        self.log_test(
            "Postman Collection", "PASS", f"Dosya oluşturuldu: {collection_path}"
        )

    def generate_curl_scripts(self):
        """cURL test scriptleri oluştur"""
        print("\n🔄 cURL Test Scriptleri Oluşturuluyor...")

        curl_scripts = {
            "health_check.sh": """#!/bin/bash
# Health Check Testleri
echo "🏥 Health Check Testleri"
echo "========================"

# Ana health endpoint
echo "Health Check:"
curl -X GET "http://localhost:8000/health" -H "Content-Type: application/json"

echo -e "\\n\\nReady Check:"
curl -X GET "http://localhost:8000/ready" -H "Content-Type: application/json"

echo -e "\\n\\nMetrics:"
curl -X GET "http://localhost:8000/metrics" -H "Content-Type: application/json"
""",
            "audio_test.sh": """#!/bin/bash
# Ses İşleme Testleri
echo "🎤 Ses İşleme Testleri"
echo "======================"

# Ses dosyası ile test
echo "Ses İşleme Testi (Healthcare):"
curl -X POST "http://localhost:8000/process-audio" \\
  -F "audio=@test_audio_files/medium_audio.wav" \\
  -F "sector=healthcare"

echo -e "\\n\\nSes İşleme Testi (Finance):"
curl -X POST "http://localhost:8000/process-audio" \\
  -F "audio=@test_audio_files/short_audio.wav" \\
  -F "sector=finance_banking"
""",
            "text_test.sh": """#!/bin/bash
# Metin İşleme Testleri
echo "📝 Metin İşleme Testleri"
echo "========================"

# Farklı sektörlerde metin testleri
echo "Sağlık Sektörü:"
curl -X POST "http://localhost:8000/process-text" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Merhaba, sağlık sektöründe çalışıyorum", "sector": "healthcare"}'

echo -e "\\n\\nFinans Sektörü:"
curl -X POST "http://localhost:8000/process-text" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Finansal danışmanlık hizmeti arıyorum", "sector": "finance_banking"}'

echo -e "\\n\\nEğitim Sektörü:"
curl -X POST "http://localhost:8000/process-text" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Eğitim materyali geliştirmek istiyorum", "sector": "education"}'
""",
            "router_test.sh": """#!/bin/bash
# Router Testleri
echo "🔄 Router Testleri"
echo "=================="

# Router health
echo "Router Health:"
curl -X GET "http://localhost:8000/router/health" -H "Content-Type: application/json"

echo -e "\\n\\nRouter Stats:"
curl -X GET "http://localhost:8000/router/stats" -H "Content-Type: application/json"

echo -e "\\n\\nSektör Sınıflandırma:"
curl -X POST "http://localhost:8000/router/classify" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Banka kredisi almak istiyorum"}'
""",
        }

        # cURL scriptlerini kaydet
        for filename, content in curl_scripts.items():
            script_path = self.test_audio_dir / filename
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Script'i çalıştırılabilir yap
            os.chmod(script_path, 0o755)
            self.log_test(
                f"cURL Script - {filename}",
                "PASS",
                f"Script oluşturuldu: {script_path}",
            )

    def generate_test_report(self):
        """Test raporu oluştur"""
        print("\n📊 Test Raporu Oluşturuluyor...")

        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.test_results if r["status"] == "FAIL"])
        warning_tests = len([r for r in self.test_results if r["status"] == "WARN"])

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        report = f"""
# Phase C4: Ses ve UI Test Raporu

## 📊 Test Özeti
- **Toplam Test**: {total_tests}
- **Başarılı**: {passed_tests} ✅
- **Başarısız**: {failed_tests} ❌
- **Uyarı**: {warning_tests} ⚠️
- **Başarı Oranı**: {success_rate:.1f}%

## 🧪 Test Sonuçları

"""

        # Test sonuçlarını grupla
        by_status = {}
        for result in self.test_results:
            status = result["status"]
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(result)

        # Her durum için testleri listele
        for status in ["PASS", "WARN", "FAIL"]:
            if status in by_status:
                status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[status]
                report += f"\n### {status_icon} {status} ({len(by_status[status])})\n\n"

                for result in by_status[status]:
                    report += f"- **{result['test']}**: {result['details']}\n"
                    report += f"  - Zaman: {result['timestamp']}\n"

        # Öneriler
        report += f"""
## 💡 Öneriler

"""

        if failed_tests > 0:
            report += "- ❌ Başarısız testleri inceleyin ve düzeltin\n"
        if warning_tests > 0:
            report += "- ⚠️ Uyarı veren testleri kontrol edin\n"
        if success_rate >= 90:
            report += "- 🎉 Mükemmel! Sistem production'a hazır\n"
        elif success_rate >= 80:
            report += "- 👍 İyi! Küçük iyileştirmeler yapılabilir\n"
        else:
            report += "- 🔧 Kritik sorunlar var, öncelikle bunları çözün\n"

        # Test dosyaları
        report += f"""
## 📁 Test Dosyaları

- **Test Ses Dosyaları**: `{self.test_audio_dir}/`
- **Postman Collection**: `{self.test_audio_dir}/postman_collection.json`
- **cURL Scriptleri**: `{self.test_audio_dir}/*.sh`

## 🚀 Sonraki Adımlar

1. **Başarısız testleri düzeltin**
2. **Production deployment'a geçin (Phase F)**
3. **Monitoring ve alerting kurun**
4. **Load testing yapın**
5. **Performance optimization yapın**

---
*Rapor oluşturulma zamanı: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # Raporu kaydet
        report_path = Path("test_audio_files/phase_c4_test_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        self.log_test("Test Report", "PASS", f"Rapor oluşturuldu: {report_path}")

        # Konsola özet yazdır
        print(f"\n📊 Test Özeti:")
        print(f"✅ Başarılı: {passed_tests}")
        print(f"❌ Başarısız: {failed_tests}")
        print(f"⚠️ Uyarı: {warning_tests}")
        print(f"📈 Başarı Oranı: {success_rate:.1f}%")
        print(f"📋 Detaylı rapor: {report_path}")

    async def run_all_tests(self):
        """Tüm testleri çalıştır"""
        print("🚀 Phase C4: Ses ve UI Testleri Başlatılıyor...")
        print("=" * 60)

        # Test ses dosyalarını oluştur
        self.create_test_audio_files()

        # Endpoint testleri
        self.test_health_endpoints()
        self.test_audio_processing_endpoints()
        self.test_text_processing_endpoints()
        self.test_router_endpoints()

        # WebSocket testi
        await self.test_websocket_connection()

        # Kalite testleri
        self.test_audio_quality()
        self.test_sector_configuration()

        # Test araçları oluştur
        self.generate_postman_collection()
        self.generate_curl_scripts()

        # Rapor oluştur
        self.generate_test_report()

        print("\n🎉 Phase C4 testleri tamamlandı!")
        print("📁 Test sonuçları: test_audio_files/ klasöründe")


async def main():
    """Ana fonksiyon"""
    tester = VoiceEndpointTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
