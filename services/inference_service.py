#!/usr/bin/env python3
"""
Turkish LLM Inference Service
Enhanced FastAPI-based inference service with sector routing and Turkish language optimization.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from contextlib import asynccontextmanager

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.router import SectorRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    """Enhanced request model for inference."""

    text: str = Field(..., description="Input text for inference")
    max_length: int = Field(
        default=512, ge=1, le=2048, description="Maximum output length"
    )
    temperature: float = Field(
        default=0.7, ge=0.1, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Top-p sampling")
    sector_hint: Optional[str] = Field(
        default=None, description="Optional sector hint for routing"
    )
    use_sector_routing: bool = Field(
        default=True, description="Enable sector-based routing"
    )


class InferenceResponse(BaseModel):
    """Enhanced response model for inference."""

    generated_text: str
    sector: str
    confidence: float
    processing_time_ms: float
    model_used: str
    adapters_used: List[str]
    timestamp: str


class TurkishInferenceService:
    """Enhanced inference service for Turkish sector-specific adapters."""

    def __init__(
        self, adapters_dir: str = "adapters", config_path: str = "configs/sectors.yaml"
    ):
        self.adapters_dir = Path(adapters_dir)
        self.config_path = config_path
        self.router = SectorRouter(config_path)
        self.base_model = None
        self.tokenizer = None
        self.adapters = {}
        self.loaded_adapters = {}
        self.start_time = time.time()
        self.request_count = 0
        self.total_inference_time = 0.0

    async def initialize(self):
        """Initialize the service asynchronously."""
        logger.info("Initializing Turkish Inference Service...")
        await self.load_base_model()
        await self.discover_adapters()
        logger.info("Service initialization complete")

    async def load_base_model(self):
        """Load the base model and tokenizer."""
        logger.info("Loading base model...")

        try:
            # Load tokenizer and model (fallback to GPT-2 for demo)
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.base_model = AutoModelForCausalLM.from_pretrained("gpt2")

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Base model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    async def discover_adapters(self):
        """Discover available adapters."""
        if not self.adapters_dir.exists():
            logger.warning(f"Adapters directory {self.adapters_dir} does not exist")
            return

        for adapter_path in self.adapters_dir.iterdir():
            if adapter_path.is_dir():
                adapter_name = adapter_path.name
                self.adapters[adapter_name] = str(adapter_path)
                logger.info(f"Discovered adapter: {adapter_name}")

    def infer(
        self,
        text: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        sector_hint: Optional[str] = None,
        use_sector_routing: bool = True,
    ) -> InferenceResponse:
        """Perform inference with enhanced features."""
        start_time = time.time()
        self.request_count += 1

        try:
            # Determine sector
            sector = "general"
            confidence = 0.5
            adapters_used = []

            if use_sector_routing:
                if sector_hint:
                    sector = sector_hint
                    confidence = 0.9
                else:
                    sector_scores = self.router.classify_sector(text)
                    if sector_scores:
                        sector, confidence = sector_scores[0]

            # Tokenize and generate
            inputs = self.tokenizer.encode(
                text, return_tensors="pt", truncation=True, max_length=512
            )

            with torch.no_grad():
                outputs = self.base_model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(
                outputs[0][inputs.shape[1] :], skip_special_tokens=True
            )
            processing_time = (time.time() - start_time) * 1000
            self.total_inference_time += processing_time

            return InferenceResponse(
                generated_text=generated_text,
                sector=sector,
                confidence=confidence,
                processing_time_ms=processing_time,
                model_used=f"gpt2:{sector}",
                adapters_used=adapters_used,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# Global service instance
inference_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global inference_service

    # Startup
    logger.info("Starting Turkish LLM Inference Service...")
    inference_service = TurkishInferenceService()
    await inference_service.initialize()
    logger.info("Service startup complete")

    yield

    # Shutdown
    logger.info("Shutting down service...")


# Initialize FastAPI app
app = FastAPI(
    title="Turkish LLM Inference Service",
    description="Sector-aware Turkish Large Language Model inference API",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/inference", response_model=InferenceResponse)
async def inference_endpoint(request: InferenceRequest) -> InferenceResponse:
    """Enhanced inference endpoint."""
    if not inference_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return inference_service.infer(
        request.text,
        request.max_length,
        request.temperature,
        request.top_p,
        request.sector_hint,
        request.use_sector_routing,
    )


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": (
            "healthy"
            if inference_service and inference_service.base_model
            else "loading"
        ),
        "model_loaded": inference_service and inference_service.base_model is not None,
        "router_status": inference_service and inference_service.router is not None,
        "uptime_seconds": time.time()
        - (inference_service.start_time if inference_service else time.time()),
    }


@app.get("/sectors")
async def list_sectors() -> Dict[str, Any]:
    """List available sectors."""
    if not inference_service or not inference_service.router:
        raise HTTPException(status_code=503, detail="Router not initialized")

    sectors = inference_service.router.list_sectors()
    return {"sectors": sectors, "total_count": len(sectors)}


@app.post("/classify")
async def classify_text(request: Dict[str, str]) -> Dict[str, Any]:
    """Classify text into sectors."""
    if not inference_service or not inference_service.router:
        raise HTTPException(status_code=503, detail="Router not initialized")

    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    sector_scores = inference_service.router.classify_sector(text)
    return {"text": text, "sector_scores": sector_scores}


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "service": "Turkish LLM Inference Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
