#!/usr/bin/env python3
"""
Turkish LLM Web Interface
Simple Streamlit UI for testing the Turkish LLM system.
"""

import streamlit as st
import requests
import json
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def main():
    st.set_page_config(page_title="Turkish LLM System", page_icon="🇹🇷", layout="wide")

    st.title("🇹🇷 Turkish LLM System")
    st.subheader("22 Sektörlü Türk İş Dünyası AI Sistemi")

    # Sidebar
    st.sidebar.header("Sistem Durumu")

    # Check if inference service is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            st.sidebar.success("✅ Inference Service Aktif")
            st.sidebar.info(f"Yüklü Adapter: {health_data.get('adapters_loaded', 0)}")
            st.sidebar.info(f"Cihaz: {health_data.get('device', 'Unknown')}")
        else:
            st.sidebar.error("❌ Inference Service Bağlantı Hatası")
    except:
        st.sidebar.warning("⚠️ Inference Service Çalışmıyor")
        st.sidebar.info("Servisi başlatmak için: python services/inference_service.py")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Sohbet")

        # Text input
        user_input = st.text_area(
            "Mesajınızı yazın:",
            placeholder="Örnek: Banka kredisi almak istiyorum, nasıl başvurabilirim?",
            height=100,
        )

        # Generation parameters
        col_params1, col_params2, col_params3 = st.columns(3)

        with col_params1:
            max_length = st.slider("Maksimum Uzunluk", 50, 200, 100)

        with col_params2:
            temperature = st.slider("Sıcaklık", 0.1, 1.0, 0.7)

        with col_params3:
            top_p = st.slider("Top P", 0.1, 1.0, 0.9)

        # Send button
        if st.button("🚀 Gönder", type="primary"):
            if user_input.strip():
                with st.spinner("Yanıt oluşturuluyor..."):
                    try:
                        # Send request to inference service
                        response = requests.post(
                            "http://localhost:8000/infer",
                            json={
                                "text": user_input,
                                "max_length": max_length,
                                "temperature": temperature,
                                "top_p": top_p,
                            },
                            timeout=30,
                        )

                        if response.status_code == 200:
                            result = response.json()

                            # Display result
                            st.success("✅ Yanıt alındı!")

                            # Response
                            st.subheader("🤖 AI Yanıtı:")
                            st.write(result["response"])

                            # Metadata
                            st.subheader("📊 Detaylar:")
                            col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)

                            with col_meta1:
                                st.metric(
                                    "Sektör", result["sector"].replace("_", " ").title()
                                )

                            with col_meta2:
                                st.metric("Güven", f"{result['confidence']:.2%}")

                            with col_meta3:
                                st.metric("Süre", f"{result['inference_time']:.2f}s")

                            with col_meta4:
                                st.metric("Adapter", len(result["adapters_used"]))

                        else:
                            st.error(f"❌ Hata: {response.status_code}")
                            st.code(response.text)

                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Bağlantı hatası: {e}")
                        st.info("Inference service'in çalıştığından emin olun.")
            else:
                st.warning("⚠️ Lütfen bir mesaj yazın.")

    with col2:
        st.header("🎯 Sektör Testleri")

        # Quick test buttons
        test_queries = {
            "🏦 Finans": "Banka kredisi almak istiyorum, faiz oranları nedir?",
            "🏥 Sağlık": "Hastane randevusu almak istiyorum, hangi doktorlar müsait?",
            "🎓 Eğitim": "Üniversite sınavına hazırlanıyorum, hangi kurslar önerilir?",
            "📰 Medya": "Gazete için muhabir arıyoruz, başvuru süreci nasıl?",
            "⚖️ Hukuk": "Hukuki danışmanlık için avukat arıyorum",
            "🏛️ Kamu": "Belediye hizmetleri hakkında bilgi istiyorum",
            "🏭 Üretim": "Fabrikada üretim süreçlerini optimize etmek istiyorum",
            "📦 Lojistik": "Lojistik süreçlerini optimize etmek istiyorum",
            "🛒 E-ticaret": "E-ticaret sitesi kurulumu için adımlar neler?",
            "⚡ Enerji": "Güneş enerjisi sistemi kurmak istiyorum",
        }

        for label, query in test_queries.items():
            if st.button(label, key=label):
                st.session_state.test_query = query
                st.rerun()

        # Auto-fill if test query selected
        if "test_query" in st.session_state:
            st.text_area("Test Sorgusu:", st.session_state.test_query, disabled=True)
            if st.button("Bu sorguyu kullan"):
                user_input = st.session_state.test_query
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("### 📋 Sistem Bilgileri")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.subheader("🏢 Desteklenen Sektörler")
        sectors = [
            "Finans ve Bankacılık",
            "Sağlık",
            "Eğitim",
            "Medya ve Yayıncılık",
            "Hukuk",
            "Kamu Yönetimi",
            "İmalat Endüstrisi",
            "Varlık Takibi",
            "Sigortacılık",
            "Turizm ve Otelcilik",
            "E-ticaret",
            "Enerji",
            "Enerji Üretimi, Dağıtımı ve İletimi",
            "Tarım",
            "Ulaşım",
            "Lojistik",
            "Telekomünikasyon",
            "İnşaat ve Mimarlık",
            "Akıllı Şehirler, Kentleşme ve Altyapı",
            "Mobilite",
            "Savunma ve Güvenlik",
            "Acil Durum İletişimi ve Afet Yönetimi",
        ]

        for sector in sectors:
            st.write(f"• {sector}")

    with col_info2:
        st.subheader("🔧 Teknik Özellikler")
        st.write("• **Router**: Akıllı sektör sınıflandırma")
        st.write("• **Adapters**: Sektör-spesifik modeller")
        st.write("• **Load Balancing**: Yük dengeleme")
        st.write("• **Multi-Expert**: Çoklu uzman desteği")
        st.write("• **Turkish NLP**: Türkçe dil optimizasyonu")
        st.write("• **Real-time**: Gerçek zamanlı yanıt")

    st.markdown("---")
    st.markdown(
        "*Phase A4: Scalability Benchmarks & Router - 22 Turkish Business Sectors*"
    )


if __name__ == "__main__":
    main()
