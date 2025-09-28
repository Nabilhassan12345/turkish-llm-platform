#!/usr/bin/env python3
"""
Demo FastAPI service for Turkish LLM
Provides a simple API endpoint for demo purposes
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Turkish LLM Demo API",
    description="Demo API for Turkish Language Model",
    version="1.0.0",
)

class InferenceRequest(BaseModel):
    """Request model for inference"""
    text: str
    sector: str = "healthcare"
    max_length: int = 100
    temperature: float = 0.7

class InferenceResponse(BaseModel):
    """Response model for inference"""
    response: str
    model_used: str
    processing_time_ms: float
    sector: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Turkish LLM Demo API"}

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """Inference endpoint"""
    start_time = time.time()
    
    # Check if model exists
    model_path = Path(f"demo_models/{request.sector}-small")
    if not model_path.exists() and request.sector != "healthcare":
        # Fall back to healthcare model
        logger.warning(f"Model for sector {request.sector} not found, using healthcare model")
        request.sector = "healthcare"
        model_path = Path("demo_models/healthcare-small")
    
    if not model_path.exists():
        raise HTTPException(status_code=500, detail="Demo model not found")
    
    # Simulate inference
    try:
        # Try to import transformers
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Check if we're using the placeholder
            placeholder = model_path / "DEMO_PLACEHOLDER_README.txt"
            
            if placeholder.exists():
                # Simulate response
                response = simulate_response(request.text, request.sector)
            else:
                # Use real model
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                
                inputs = tokenizer(request.text, return_tensors="pt")
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=0.9,
                    do_sample=True
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
        except (ImportError, Exception) as e:
            logger.warning(f"Error using transformers: {e}")
            # Fall back to simulated response
            response = simulate_response(request.text, request.sector)
            
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return InferenceResponse(
            response=response,
            model_used=f"{request.sector}-small",
            processing_time_ms=processing_time,
            sector=request.sector
        )
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

def simulate_response(text: str, sector: str) -> str:
    """Simulate a response for demo purposes"""
    responses = {
        "healthcare": {
            "default": "Bu sağlık sorunuyla ilgili bir doktora başvurmanızı öneririm.",
            "aile hekimi": "Aile hekiminizi değiştirmek için MHRS üzerinden veya e-Nabız uygulamasından işlem yapabilirsiniz.",
            "randevu": "Randevu almak için MHRS uygulamasını kullanabilir veya ALO 182'yi arayabilirsiniz.",
            "covid": "COVID-19 aşısı için size en yakın aşı merkezine başvurabilirsiniz.",
        },
        "finance": {
            "default": "Bu finansal konuyla ilgili bankanızla iletişime geçmenizi öneririm.",
            "kredi": "Kredi başvurusu için bankanızın internet şubesini kullanabilir veya şubeye gidebilirsiniz.",
            "kart": "Kredi kartı başvurusu için bankanızın internet şubesini kullanabilir veya şubeye gidebilirsiniz.",
            "borç": "Kredi kartı borcunuzu yapılandırmak için bankanızın müşteri hizmetlerini aramalı veya internet bankacılığı üzerinden başvuru yapmalısınız.",
        },
        "education": {
            "default": "Bu eğitim sorunuyla ilgili okulunuzla iletişime geçmenizi öneririm.",
            "üniversite": "Üniversite başvurusu için YKS sonuç belgesi, lise diploması ve nüfus cüzdanı fotokopisi gereklidir.",
            "burs": "Burs başvurusu için gerekli belgeler: transkript, gelir belgesi ve ikametgah belgesidir.",
            "sınav": "Sınav sonuçları e-Devlet üzerinden veya ilgili kurumun web sitesinden öğrenilebilir.",
        }
    }
    
    # Default to healthcare if sector not found
    sector_responses = responses.get(sector, responses["healthcare"])
    
    # Check for keywords in the text
    for keyword, response in sector_responses.items():
        if keyword in text.lower() and keyword != "default":
            return response
    
    # Return default response if no keyword matches
    return sector_responses["default"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)