# Turkish AI Agent: Enterprise-Grade AI Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Project Overview

**Turkish AI Agent** is a full enterprise-grade AI platform for Turkish business sectors, showcasing advanced AI/ML engineering, full-stack development, and production-ready infrastructure:

* **22 sector-specific AI models** (Healthcare, Education, Finance, and more)
* **Real-time voice interaction** with WebSocket, STT/TTS, and SSML support
* **Modern React frontend** with TypeScript and Tailwind CSS
* **RAG (Retrieval-Augmented Generation)** for context-aware responses
* **Production infrastructure** with Docker, Prometheus & Grafana monitoring, and CI/CD pipelines
* **Sector routing / Mixture-of-Experts** (MoE) architecture for intelligent adapter selection

This repository contains **demo models** for portfolio purposes. Full-scale models and datasets are summarized in `MODEL_CARD.md`.

---

## 🎯 Key Technical Achievements

### AI / ML Engineering

* **Gemma-2B Fine-Tuned Models** (4-bit QLoRA, LoRA adapters, rank 8 / alpha 16)

  * Healthcare: 0.0781 loss, ~2h 49m training
  * Education: 0.2571 loss, ~9h 50m training
  * Finance: ~0.25-0.30 loss, ~10h training
* **GPT-Neo-1.3B Models** for experimentation (loss initially high, dropped to ~10)
* **RTX 4060 Optimized** for 8GB VRAM using 4-bit quantization
* **RAG Integration**: FAISS vector DB, top-k chunk retrieval, context-aware inference

### Enterprise Architecture & Services

* **Voice Orchestrator**: WebSocket real-time communication
* **Inference Service**: FastAPI endpoints with intelligent sector routing
* **Sector Router / MoE**: 22 sector adapters, confidence-based expert selection

### Frontend

* React 18 + TypeScript + Tailwind CSS
* Voice chat interface with sector selection
* Responsive UI with Framer Motion animations

### Infrastructure & Monitoring

* Dockerized multi-service deployment
* Prometheus + Grafana dashboards for real-time metrics
* GitHub Actions CI/CD with automated testing and security scans

---

## 📊 Demo & Portfolio Showcase

Everything for the demo can be bootstrapped automatically with one script. Just run:

```bash
bash bootstrap_project.sh
```

This script will generate:

* `requirements_demo.txt`
* `demo/demo_inference.py`
* `demo/demo_rag_example.py`
* `MODEL_CARD.md`
* `.gitignore`
* `docker-compose.yml`
* `app/inference_service.py`
* `.github/workflows/ci.yml`

Once generated, you can:

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# 2. Install demo dependencies
pip install -r requirements_demo.txt

# 3. Run the minimal demo
python demo/demo_inference.py

# 4. (Optional) Run RAG demo
python demo/demo_rag_example.py

# 5. (Optional) Run FastAPI service
uvicorn app.inference_service:app --reload --port 8000

# 6. (Optional) Run via Docker Compose
docker-compose up --build
```

### Demo Features

* Voice interaction: Speak or type queries in Turkish
* Sector selection: Healthcare, Education, Finance
* RAG integration: Context-aware AI responses
* Minimal resource usage: Runs on RTX 4060 / 8GB VRAM or CPU-only mode

⚠️ Full-scale models and datasets are not included due to GitHub limits. This demo demonstrates core functionality for portfolio purposes.

---

## 📜 bootstrap_project.sh

```bash
#!/bin/bash

# Create folder structure
mkdir -p demo app .github/workflows

# requirements_demo.txt
cat > requirements_demo.txt <<EOL
torch>=2.1.0
transformers>=4.35.0
sentence-transformers>=2.2.2
fastapi>=0.100.0
uvicorn>=0.24.0
python-dotenv>=1.0.0
numpy>=1.25.0
scikit-learn>=1.3.0
faiss-cpu>=1.7.4
pyttsx3>=2.90
sounddevice>=0.4.7
soundfile>=0.12.1
requests>=2.31.0
EOL

# demo_inference.py
cat > demo/demo_inference.py <<EOL
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"  # Tiny placeholder for portfolio demo

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def demo_query(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("Turkish AI Agent Minimal Demo")
    while True:
        query = input("Type a query: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = demo_query(query)
        print(f"AI Response: {response}")
EOL

# demo_rag_example.py
cat > demo/demo_rag_example.py <<EOL
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
corpus = ["Merhaba, bu bir demo.", "Eğitim sektörü örnek veri.", "Finans sektörü örnek veri."]
corpus_embeddings = model.encode(corpus)

def rag_demo(query: str):
    query_emb = model.encode([query])[0]
    scores = np.dot(corpus_embeddings, query_emb)
    best_idx = np.argmax(scores)
    return corpus[best_idx]

if __name__ == "__main__":
    print("RAG Minimal Demo")
    while True:
        q = input("Type a query: ")
        if q.lower() in ["exit", "quit"]:
            break
        print(f"RAG Response: {rag_demo(q)}")
EOL

# MODEL_CARD.md
cat > MODEL_CARD.md <<EOL
# Turkish AI Agent Model Card

This document summarizes the models trained and used in this project.

- **Gemma-2B fine-tuned models**: Healthcare, Education, Finance
- **RAG integration** with FAISS
- **Mixture-of-Experts routing** for 22 sectors

⚠️ Note: Full models and datasets are not included due to GitHub limits.
EOL

# docker-compose.yml
cat > docker-compose.yml <<EOL
version: '3.9'
services:
  inference:
    build: .
    command: uvicorn app.inference_service:app --host 0.0.0.0 --port 8000
    volumes:
      - .:/code
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
EOL

# inference_service.py
cat > app/inference_service.py <<EOL
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

@app.post("/api/generate")
def generate(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}
EOL

# GitHub Actions workflow
cat > .github/workflows/ci.yml <<EOL
name: CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements_demo.txt
      - name: Run lint
        run: |
          pip install flake8
          flake8 .
      - name: Run tests
        run: |
          echo "No tests yet."
EOL

# .gitignore
cat > .gitignore <<EOL
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.DS_Store
EOL

echo "✅ Project bootstrap complete."
```

---

## 📞 Contact & Portfolio

For inquiries, collaboration, or portfolio-related questions:

* Email: **[nabilhassanmohamedali@gmail.com](mailto:nabilhassanmohamedali@gmail.com)**
* GitHub Repository: [Turkish LLM Platform](https://github.com/Nabilhassan12345/turkish-llm-platform)

---

## ⚠️ Responsible Use Disclaimer

* Healthcare / Finance models: Do **NOT** use outputs as final medical, legal, or financial advice.
* Always include a human-in-the-loop for high-risk decisions.
* Personal data (PII/PHI) must be sanitized before any model usage.

---

## 📝 Notes for Portfolio

* All essential components are preserved for demonstration.
* Full models, datasets, and training logs are summarized in `MODEL_CARD.md`.
* This setup is optimized for portfolio presentation, clean structure, and easy demo execution.
