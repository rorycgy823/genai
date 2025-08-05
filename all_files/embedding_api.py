#!/usr/bin/env python3
"""
Embedding API Service for Virtual Machine
========================================

A standalone embedding service to be deployed on Ubuntu VM
- Provides text embedding generation for ChromaDB
- Supports multiple embedding models with fallbacks
- Designed for deployment on Ubuntu-22.04-x64 VM

Server: 1.32.228.33:8001
Author: Rory Chen
"""

import os
import sys
import json
import logging
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/embedding_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try to import embedding libraries with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("SentenceTransformers available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available")

try:
    import openai
    OPENAI_AVAILABLE = True
    logger.info("OpenAI available")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available")

# Pydantic models for API requests
class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = "default"
    normalize: Optional[bool] = True

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model_used: str
    dimension: int
    processing_time: float

class EmbeddingService:
    """Embedding service manager for VM deployment"""
    
    def __init__(self):
        self.models = {}
        self.default_model = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize available embedding models"""
        try:
            # Try to load SentenceTransformers models
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self._load_sentence_transformer_models()
            
            # Set default model
            if self.models:
                self.default_model = list(self.models.keys())[0]
                logger.info(f"Default model set to: {self.default_model}")
            else:
                logger.warning("No embedding models available - using fallback")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {str(e)}")
    
    def _load_sentence_transformer_models(self):
        """Load SentenceTransformers models with fallbacks"""
        models_to_try = [
            "all-MiniLM-L6-v2",  # Fast and efficient
            "all-mpnet-base-v2",  # High quality
            "paraphrase-MiniLM-L6-v2",  # Lightweight
        ]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Loading SentenceTransformer model: {model_name}")
                model = SentenceTransformer(model_name)
                self.models[model_name] = {
                    "model": model,
                    "type": "sentence_transformer",
                    "dimension": model.get_sentence_embedding_dimension()
                }
                logger.info(f"Successfully loaded {model_name} (dim: {model.get_sentence_embedding_dimension()})")
                break  # Use the first successfully loaded model
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
                continue
    
    def get_embeddings(self, texts: List[str], model_name: str = None, 
                      normalize: bool = True) -> Dict[str, Any]:
        """Generate embeddings for input texts"""
        start_time = datetime.now()
        
        try:
            # Use default model if none specified
            if model_name is None or model_name == "default":
                model_name = self.default_model
            
            if not model_name or model_name not in self.models:
                # Fallback to simple hash-based embeddings
                return self._generate_fallback_embeddings(texts, start_time)
            
            model_info = self.models[model_name]
            model = model_info["model"]
            model_type = model_info["type"]
            
            if model_type == "sentence_transformer":
                embeddings = model.encode(texts, normalize_embeddings=normalize)
                embeddings = embeddings.tolist()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Generated embeddings for {len(texts)} texts using {model_name} in {processing_time:.3f}s")
            
            return {
                "embeddings": embeddings,
                "model_used": model_name,
                "dimension": len(embeddings[0]) if embeddings else 0,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Fallback to simple embeddings
            return self._generate_fallback_embeddings(texts, start_time)
    
    def _generate_fallback_embeddings(self, texts: List[str], start_time: datetime) -> Dict[str, Any]:
        """Generate simple hash-based embeddings as fallback"""
        try:
            embeddings = []
            dimension = 384  # Standard dimension for fallback
            
            for text in texts:
                # Simple hash-based embedding
                text_hash = hash(text.lower())
                
                # Convert hash to embedding vector
                embedding = []
                for i in range(dimension):
                    # Use different hash functions for each dimension
                    val = hash(f"{text_hash}_{i}") % 1000000
                    embedding.append(val / 1000000.0 - 0.5)  # Normalize to [-0.5, 0.5]
                
                # Normalize the embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = [x / norm for x in embedding]
                
                embeddings.append(embedding)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.warning(f"Using fallback embeddings for {len(texts)} texts")
            
            return {
                "embeddings": embeddings,
                "model_used": "fallback_hash",
                "dimension": dimension,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Fallback embedding generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models"""
        models_info = {}
        
        for name, info in self.models.items():
            models_info[name] = {
                "type": info["type"],
                "dimension": info["dimension"],
                "status": "available"
            }
        
        # Add fallback model info
        models_info["fallback_hash"] = {
            "type": "hash_based",
            "dimension": 384,
            "status": "available"
        }
        
        return {
            "available_models": models_info,
            "default_model": self.default_model or "fallback_hash",
            "total_models": len(models_info)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        try:
            # Test embedding generation
            test_texts = ["Hello world", "Test embedding"]
            result = self.get_embeddings(test_texts)
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "models_available": len(self.models),
                "default_model": self.default_model,
                "test_embedding_dimension": result["dimension"],
                "libraries": {
                    "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
                    "openai": OPENAI_AVAILABLE
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Initialize embedding service
embedding_service = EmbeddingService()

# Initialize FastAPI app
app = FastAPI(
    title="Embedding API Service for Rory's AI Assistant",
    description="Text embedding generation service for vector database operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Embedding API Service for Rory's AI Assistant",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "embeddings": "/embeddings",
            "models": "/models",
            "batch_embeddings": "/embeddings/batch"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return embedding_service.health_check()

@app.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for input texts"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if len(request.texts) > 1000:
        raise HTTPException(status_code=400, detail="Too many texts (max 1000)")
    
    result = embedding_service.get_embeddings(
        request.texts,
        request.model,
        request.normalize
    )
    
    return EmbeddingResponse(**result)

@app.post("/embeddings/batch")
async def generate_batch_embeddings(request: EmbeddingRequest):
    """Generate embeddings in batches for large inputs"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    batch_size = 100
    all_embeddings = []
    total_processing_time = 0
    model_used = None
    dimension = 0
    
    # Process in batches
    for i in range(0, len(request.texts), batch_size):
        batch_texts = request.texts[i:i + batch_size]
        batch_result = embedding_service.get_embeddings(
            batch_texts,
            request.model,
            request.normalize
        )
        
        all_embeddings.extend(batch_result["embeddings"])
        total_processing_time += batch_result["processing_time"]
        model_used = batch_result["model_used"]
        dimension = batch_result["dimension"]
    
    return {
        "embeddings": all_embeddings,
        "model_used": model_used,
        "dimension": dimension,
        "processing_time": total_processing_time,
        "batch_size": batch_size,
        "total_texts": len(request.texts)
    }

@app.get("/models")
async def get_available_models():
    """Get information about available embedding models"""
    return embedding_service.get_available_models()

@app.post("/embeddings/similarity")
async def compute_similarity(texts1: List[str], texts2: List[str], model: str = "default"):
    """Compute similarity between two sets of texts"""
    try:
        # Generate embeddings for both sets
        result1 = embedding_service.get_embeddings(texts1, model)
        result2 = embedding_service.get_embeddings(texts2, model)
        
        embeddings1 = np.array(result1["embeddings"])
        embeddings2 = np.array(result2["embeddings"])
        
        # Compute cosine similarity
        similarities = []
        for emb1 in embeddings1:
            row_similarities = []
            for emb2 in embeddings2:
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                row_similarities.append(float(similarity))
            similarities.append(row_similarities)
        
        return {
            "similarities": similarities,
            "model_used": result1["model_used"],
            "texts1_count": len(texts1),
            "texts2_count": len(texts2)
        }
        
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task for model management
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Embedding API Service starting up...")
    logger.info(f"Service available at: http://0.0.0.0:8001")
    logger.info(f"Available models: {list(embedding_service.models.keys())}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Embedding API Service shutting down...")

if __name__ == "__main__":
    # Configuration
    HOST = "0.0.0.0"  # Listen on all interfaces
    PORT = 8001
    
    logger.info(f"Starting Embedding API Service on {HOST}:{PORT}")
    
    # Run the service
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True
    )
