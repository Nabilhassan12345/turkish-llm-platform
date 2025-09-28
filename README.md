# 🇹🇷 Turkish AI Agent: Enterprise-Grade AI Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal)](https://fastapi.tiangolo.com/)
[![CI/CD](https://img.shields.io/badge/GitHub-Actions-purple)](https://github.com/features/actions)
[![CPU Friendly](https://img.shields.io/badge/CPU-Friendly-lightgrey)](#)

⭐ If you find this project useful, consider giving it a **star** on GitHub — it helps showcase my work to recruiters and collaborators!

---

## 🚀 Project Overview

**Turkish AI Agent** is an enterprise-grade AI platform tailored for Turkish business sectors, combining advanced AI/ML engineering, full-stack development, and production-ready infrastructure.  

💼 **Portfolio-ready:** Designed for recruiters and technical showcases. The demo runs fully on **8GB VRAM (RTX 4060)** or **CPU-only mode**, making it accessible to anyone.

- **22 sector-specific AI models** (Healthcare, Education, Finance, and more)  
- **Real-time voice interaction** with WebSocket, STT/TTS, and SSML support  
- **Modern React frontend** with TypeScript and Tailwind CSS  
- **RAG (Retrieval-Augmented Generation)** for context-aware responses  
- **Production infrastructure** with Docker, Prometheus & Grafana monitoring, and CI/CD pipelines  
- **Sector routing / Mixture-of-Experts** (MoE) architecture for intelligent adapter selection  

---

## 🎯 Key Technical Achievements

### 🤖 AI / ML Engineering
- **Gemma-2B Fine-Tuned Models** (4-bit QLoRA, LoRA adapters, rank 8 / alpha 16)  
  - Healthcare: 0.0781 loss, ~2h 49m training  
  - Education: 0.2571 loss, ~9h 50m training  
  - Finance: ~0.25–0.30 loss, ~10h training  
- **GPT-Neo-1.3B Models** for experimentation (loss initially high, dropped to ~10)  
- **RTX 4060 Optimized** for 8GB VRAM using 4-bit quantization  
- **RAG Integration**: FAISS vector DB, top-k chunk retrieval, context-aware inference  

### 🏗 Enterprise Architecture & Services
- **Voice Orchestrator**: WebSocket real-time communication  
- **Inference Service**: FastAPI endpoints with intelligent sector routing  
- **Sector Router / MoE**: 22 sector adapters, confidence-based expert selection  

### 🎨 Frontend
- React 18 + TypeScript + Tailwind CSS  
- Voice chat interface with sector selection  
- Responsive UI with Framer Motion animations  

### 📡 Infrastructure & Monitoring
- Dockerized multi-service deployment  
- Prometheus + Grafana dashboards for real-time metrics  
- GitHub Actions CI/CD with automated testing and security scans  

---

## 📊 Demo & Portfolio Showcase

Everything for the demo can be bootstrapped automatically with one script:

```bash
bash bootstrap_project.sh

This script generates all required demo files (requirements_demo.txt, demo/demo_inference.py, MODEL_CARD.md, etc.).

⚡ Quick Start (1-Minute Demo)
🖥 CPU Mode (Minimal Resources
python demo/demo_inference.py

🚀 FastAPI Service
uvicorn app.inference_service:app --reload --port 8000

🐳 Docker Deployment
docker-compose up --build

🎤 Demo Features (Why It Matters)
• 	Voice interaction with sector-aware AI → realistic enterprise scenarios
• 	RAG integration → demonstrates real-world context-aware intelligence
• 	Sector selection (Healthcare, Education, Finance) → shows adaptability across industries
• 	CPU-friendly mode → recruiters can test without GPU hardware

Perfect — thanks for clarifying, Nabil. I’ll now give you a single, polished, recruiter‑ready README canvas in one big Markdown block. It will include:
• 	✅ Extra badges (Docker, FastAPI, CI/CD, CPU‑friendly)
• 	✅ Portfolio emphasis in the top summary
• 	✅ “Star this repo ⭐” call‑to‑action (tasteful, not pushy)
• 	✅ Expanded demo features with “why it matters” notes
• 	✅ Clear Quick Start (CPU, FastAPI, Docker)
• 	✅ Stronger disclaimer (“for demonstration only”)
• 	✅ Technical Highlights table (model sizes, VRAM, training time)
• 	✅ Contact section with your LinkedIn link
• 	✅ Placeholders for visuals (so you can add screenshots/diagrams later)
Here’s your ready‑to‑paste canvas:

This script generates all required demo files (, , , etc.).

⚡ Quick Start (1-Minute Demo)
🖥 CPU Mode (Minimal Resources)

🚀 FastAPI Service

🐳 Docker Deployment


🎤 Demo Features (Why It Matters)
• 	Voice interaction with sector-aware AI → realistic enterprise scenarios
• 	RAG integration → demonstrates real-world context-aware intelligence
• 	Sector selection (Healthcare, Education, Finance) → shows adaptability across industries
• 	CPU-friendly mode → recruiters can test without GPU hardware

📊 Technical Highlights


🖼 Visual Enhancements


⚠️ Responsible Use Disclaimer
For demonstration purposes only.
• 	Healthcare / Finance models: Do NOT use outputs as final medical, legal, or financial advice.
• 	Always include a human-in-the-loop for high-risk decisions.
• 	Personal data (PII/PHI) must be sanitized before any model usage.

📞 Contact & Portfolio
• 	Email: nabilhassanmohamedali@gmail.com
• 	GitHub: Turkish LLM Platform
• 	LinkedIn: linkedin.com/in/nabil-h-003751299

📝 Notes for Portfolio
• 	All essential components are preserved for demonstration.
• 	Full models, datasets, and training logs are summarized in .
• 	Optimized for portfolio presentation, clean structure, and easy demo execution.
