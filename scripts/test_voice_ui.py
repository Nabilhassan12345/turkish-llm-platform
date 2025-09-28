#!/usr/bin/env python3
"""
Phase C4: Yerel Ses ve UI Duman Testleri
Voice Chat ve UI bileşenlerinin kapsamlı testleri
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


class VoiceUITester:
    """Voice Chat ve UI bileşenlerini test eden sınıf"""

    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8765"
        self.test_results = []

    def log_test(self, test_name, status, details=""):
        """Test sonucunu kaydet"""
        result = {
            "test": test_name,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "details": details,
        }
        self.test_results.append(result)
        print(f"🧪 {test_name}: {'✅' if status == 'PASS' else '❌'} {details}")

    def create_test_audio(self, duration=3, sample_rate=16000):
        """Test için sahte ses dosyası oluştur"""
        # Basit sinüs dalgası oluştur
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.3  # 440 Hz, 0.3 amplitude

        # WAV dosyası olarak kaydet
        test_audio_path = Path("test_audio.wav")
        with wave.open(str(test_audio_path), "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        return test_audio_path

    def test_voice_orchestrator_endpoints(self):
        """Voice orchestrator API endpoint'lerini test et"""
        print("\n🎯 Voice Orchestrator API Testleri")
        print("=" * 50)

        # Test 1: Health check
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                self.log_test("Health Check", "PASS", f"Status: {response.status_code}")
            else:
                self.log_test("Health Check", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Health Check", "FAIL", f"Error: {str(e)}")

        # Test 2: Audio processing endpoint
        try:
            test_audio = self.create_test_audio()
            with open(test_audio, "rb") as f:
                files = {"audio": ("test.wav", f, "audio/wav")}
                data = {"sector": "healthcare"}
                response = requests.post(
                    f"{self.base_url}/process-audio", files=files, data=data
                )

            if response.status_code == 200:
                result = response.json()
                self.log_test(
                    "Audio Processing",
                    "PASS",
                    f"Response: {result.get('status', 'unknown')}",
                )
            else:
                self.log_test(
                    "Audio Processing", "FAIL", f"Status: {response.status_code}"
                )

            # Test dosyasını temizle
            test_audio.unlink()

        except Exception as e:
            self.log_test("Audio Processing", "FAIL", f"Error: {str(e)}")

        # Test 3: Text processing endpoint
        try:
            data = {
                "text": "Merhaba, sağlık sektöründe çalışıyorum",
                "sector": "healthcare",
            }
            response = requests.post(f"{self.base_url}/process-text", json=data)

            if response.status_code == 200:
                result = response.json()
                self.log_test(
                    "Text Processing",
                    "PASS",
                    f"Response: {result.get('status', 'unknown')}",
                )
            else:
                self.log_test(
                    "Text Processing", "FAIL", f"Status: {response.status_code}"
                )

        except Exception as e:
            self.log_test("Text Processing", "FAIL", f"Error: {str(e)}")

    async def test_websocket_connection(self):
        """WebSocket bağlantısını test et"""
        print("\n🔌 WebSocket Bağlantı Testleri")
        print("=" * 50)

        try:
            async with websockets.connect(self.ws_url) as websocket:
                self.log_test("WebSocket Connection", "PASS", "Bağlantı başarılı")

                # Test mesajı gönder
                test_message = {
                    "type": "test_message",
                    "text": "Bu bir test mesajıdır",
                    "sector": "general",
                }

                await websocket.send(json.dumps(test_message))
                self.log_test("WebSocket Send", "PASS", "Mesaj gönderildi")

                # Yanıt bekle (timeout ile)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    self.log_test(
                        "WebSocket Receive",
                        "PASS",
                        f"Yanıt alındı: {response[:100]}...",
                    )
                except asyncio.TimeoutError:
                    self.log_test("WebSocket Receive", "FAIL", "Yanıt timeout")

        except Exception as e:
            self.log_test("WebSocket Connection", "FAIL", f"Error: {str(e)}")

    def test_ui_components(self):
        """UI bileşenlerini test et"""
        print("\n🎨 UI Bileşen Testleri")
        print("=" * 50)

        # React uygulamasının çalışıp çalışmadığını kontrol et
        try:
            response = requests.get("http://localhost:3000")
            if response.status_code == 200:
                self.log_test("React App", "PASS", "Port 3000'de çalışıyor")
            else:
                self.log_test("React App", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("React App", "FAIL", f"Port 3000'de erişilemiyor: {str(e)}")

        # UI dosyalarının varlığını kontrol et
        ui_files = [
            "ui/src/App.tsx",
            "ui/src/components/VoiceChat.tsx",
            "ui/src/components/SectorSelector.tsx",
            "ui/src/components/ChatHistory.tsx",
            "ui/package.json",
            "ui/tailwind.config.js",
        ]

        for file_path in ui_files:
            if Path(file_path).exists():
                self.log_test(f"UI File: {file_path}", "PASS", "Dosya mevcut")
            else:
                self.log_test(f"UI File: {file_path}", "FAIL", "Dosya bulunamadı")

    def test_audio_flow(self):
        """Ses akışını test et"""
        print("\n🎵 Ses Akış Testleri")
        print("=" * 50)

        # Test 1: Ses dosyası oluşturma
        try:
            test_audio = self.create_test_audio()
            if test_audio.exists():
                self.log_test(
                    "Audio Generation", "PASS", f"Dosya oluşturuldu: {test_audio}"
                )
            else:
                self.log_test("Audio Generation", "FAIL", "Ses dosyası oluşturulamadı")

            # Test 2: Ses dosyası formatı
            with wave.open(str(test_audio), "r") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()

                if channels == 1 and sample_width == 2 and frame_rate == 16000:
                    self.log_test(
                        "Audio Format",
                        "PASS",
                        f"Format: {channels}ch, {sample_width*8}bit, {frame_rate}Hz",
                    )
                else:
                    self.log_test(
                        "Audio Format",
                        "FAIL",
                        f"Beklenen format değil: {channels}ch, {sample_width*8}bit, {frame_rate}Hz",
                    )

            # Test 3: Ses dosyası boyutu
            file_size = test_audio.stat().st_size
            if file_size > 1000:  # En az 1KB
                self.log_test("Audio File Size", "PASS", f"Boyut: {file_size} bytes")
            else:
                self.log_test(
                    "Audio File Size", "FAIL", f"Çok küçük dosya: {file_size} bytes"
                )

            # Temizlik
            test_audio.unlink()

        except Exception as e:
            self.log_test("Audio Flow", "FAIL", f"Error: {str(e)}")

    def test_sector_configuration(self):
        """Sektör konfigürasyonunu test et"""
        print("\n🏢 Sektör Konfigürasyon Testleri")
        print("=" * 50)

        # Sektör config dosyasını kontrol et
        sectors_config = Path("configs/sectors.yaml")
        if sectors_config.exists():
            self.log_test("Sectors Config", "PASS", "Sektör config dosyası mevcut")

            # YAML içeriğini kontrol et
            try:
                import yaml

                with open(sectors_config, "r", encoding="utf-8") as f:
                    sectors = yaml.safe_load(f)

                if sectors and isinstance(sectors, dict):
                    sector_count = len(sectors)
                    self.log_test(
                        "Sectors Content", "PASS", f"{sector_count} sektör tanımlı"
                    )
                else:
                    self.log_test("Sectors Content", "FAIL", "Geçersiz YAML formatı")

            except Exception as e:
                self.log_test("Sectors Content", "FAIL", f"YAML parse error: {str(e)}")
        else:
            self.log_test("Sectors Config", "FAIL", "Sektör config dosyası bulunamadı")

    def run_postman_tests(self):
        """Postman benzeri testler çalıştır"""
        print("\n📮 Postman/Curl Testleri")
        print("=" * 50)

        # Test 1: GET /health
        curl_cmd = f'curl -s -o /dev/null -w "%{{http_code}}" {self.base_url}/health'
        try:
            result = subprocess.run(
                curl_cmd, shell=True, capture_output=True, text=True
            )
            if result.stdout.strip() == "200":
                self.log_test("Curl Health Check", "PASS", "HTTP 200 OK")
            else:
                self.log_test(
                    "Curl Health Check", "FAIL", f"HTTP {result.stdout.strip()}"
                )
        except Exception as e:
            self.log_test("Curl Health Check", "FAIL", f"Error: {str(e)}")

        # Test 2: POST /process-text
        curl_cmd = f'''curl -s -X POST {self.base_url}/process-text \\
            -H "Content-Type: application/json" \\
            -d '{{"text": "Test mesajı", "sector": "general"}}' \\
            -w "%{{http_code}}"'''

        try:
            result = subprocess.run(
                curl_cmd, shell=True, capture_output=True, text=True
            )
            if result.stdout.strip() in ["200", "201"]:
                self.log_test(
                    "Curl Text Processing", "PASS", f"HTTP {result.stdout.strip()}"
                )
            else:
                self.log_test(
                    "Curl Text Processing", "FAIL", f"HTTP {result.stdout.strip()}"
                )
        except Exception as e:
            self.log_test("Curl Text Processing", "FAIL", f"Error: {str(e)}")

    def generate_test_report(self):
        """Test raporu oluştur"""
        print("\n📊 Test Raporu")
        print("=" * 50)

        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASS"])
        failed_tests = total_tests - passed_tests

        print(f"Toplam Test: {total_tests}")
        print(f"Başarılı: {passed_tests} ✅")
        print(f"Başarısız: {failed_tests} ❌")
        print(f"Başarı Oranı: {(passed_tests/total_tests)*100:.1f}%")

        # Başarısız testleri listele
        if failed_tests > 0:
            print("\n❌ Başarısız Testler:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  - {result['test']}: {result['details']}")

        # Raporu dosyaya kaydet
        report_path = Path("test_report_voice_ui.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "summary": {
                        "total": total_tests,
                        "passed": passed_tests,
                        "failed": failed_tests,
                        "success_rate": (passed_tests / total_tests) * 100,
                    },
                    "results": self.test_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\n📄 Detaylı rapor kaydedildi: {report_path}")

        return passed_tests == total_tests

    async def run_all_tests(self):
        """Tüm testleri çalıştır"""
        print("🚀 Voice Chat ve UI Testleri Başlatılıyor...")
        print("=" * 60)

        # Test 1: Voice Orchestrator endpoints
        self.test_voice_orchestrator_endpoints()

        # Test 2: WebSocket connection
        await self.test_websocket_connection()

        # Test 3: UI components
        self.test_ui_components()

        # Test 4: Audio flow
        self.test_audio_flow()

        # Test 5: Sector configuration
        self.test_sector_configuration()

        # Test 6: Postman/curl tests
        self.run_postman_tests()

        # Rapor oluştur
        success = self.generate_test_report()

        if success:
            print("\n🎉 Tüm testler başarılı! Voice Chat ve UI hazır.")
        else:
            print("\n⚠️  Bazı testler başarısız. Lütfen hataları kontrol edin.")

        return success


async def main():
    """Ana test fonksiyonu"""
    tester = VoiceUITester()

    try:
        await tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n⏹️  Testler kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n💥 Test çalıştırılırken hata oluştu: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
