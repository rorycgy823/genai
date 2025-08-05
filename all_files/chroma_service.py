#!/usr/bin/env python3
"""
ChromaDB Service for Virtual Machine
===================================

A standalone ChromaDB server service to be deployed on Ubuntu VM
- Hosts vector database for Rory's AI Assistant
- Provides HTTP API endpoints for document storage and retrieval
- Designed for deployment on Ubuntu-22.04-x64 VM

Server: 1.32.228.33:8000
Author: Rory Chen
"""

import os
import sys
import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/chroma_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for API requests
class DocumentRequest(BaseModel):
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    ids: List[str]

class QueryRequest(BaseModel):
    query_texts: List[str]
    n_results: int = 5
    where: Optional[Dict[str, Any]] = None
    where_document: Optional[Dict[str, Any]] = None

class ConversationRequest(BaseModel):
    user_query: str
    ai_response: str
    chunks_used: int = 0

class ChromaDBService:
    """ChromaDB service manager for VM deployment"""
    
    def __init__(self, persist_directory: str = "/root/data/chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collections = {}
        self.initialize_service()
    
    def initialize_service(self):
        """Initialize ChromaDB client and collections"""
        try:
            # Ensure persist directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize collections
            self.collections['knowledge_base'] = self._get_or_create_collection(
                "rory_knowledge_base",
                {"description": "Rory's professional knowledge base"}
            )
            
            self.collections['conversations'] = self._get_or_create_collection(
                "conversation_logs",
                {"description": "User conversations for model improvement"}
            )
            
            logger.info("ChromaDB service initialized successfully")
            logger.info(f"Persist directory: {self.persist_directory}")
            logger.info(f"Collections: {list(self.collections.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB service: {str(e)}")
            raise
    
    def _get_or_create_collection(self, name: str, metadata: Dict[str, Any]):
        """Get existing collection or create new one"""
        try:
            return self.client.get_collection(name=name)
        except Exception:
            logger.info(f"Creating new collection: {name}")
            return self.client.create_collection(name=name, metadata=metadata)
    
    def add_documents(self, collection_name: str, documents: List[str], 
                     metadatas: List[Dict], ids: List[str]) -> Dict[str, Any]:
        """Add documents to specified collection"""
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection = self.collections[collection_name]
            
            # Add documents to collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            doc_count = collection.count()
            logger.info(f"Added {len(documents)} documents to {collection_name}. Total: {doc_count}")
            
            return {
                "status": "success",
                "added_count": len(documents),
                "total_count": doc_count,
                "collection": collection_name
            }
            
        except Exception as e:
            logger.error(f"Error adding documents to {collection_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def query_documents(self, collection_name: str, query_texts: List[str], 
                       n_results: int = 5, where: Dict = None, 
                       where_document: Dict = None) -> Dict[str, Any]:
        """Query documents from specified collection"""
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection = self.collections[collection_name]
            
            # Query collection
            results = collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            logger.info(f"Queried {collection_name} with {len(query_texts)} queries, returned {len(results.get('documents', [[]]))} result sets")
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying {collection_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection"""
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection = self.collections[collection_name]
            count = collection.count()
            
            return {
                "name": collection_name,
                "count": count,
                "metadata": collection.metadata if hasattr(collection, 'metadata') else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting info for {collection_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def log_conversation(self, user_query: str, ai_response: str, chunks_used: int = 0) -> Dict[str, Any]:
        """Log conversation to conversations collection"""
        try:
            conversation_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            conversation_text = f"User: {user_query}\nAI: {ai_response}"
            
            metadata = {
                "timestamp": timestamp,
                "user_query": user_query,
                "ai_response": ai_response,
                "chunks_used": chunks_used,
                "type": "conversation"
            }
            
            return self.add_documents(
                "conversations",
                [conversation_text],
                [metadata],
                [conversation_id]
            )
            
        except Exception as e:
            logger.error(f"Error logging conversation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        try:
            status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "collections": {}
            }
            
            for name, collection in self.collections.items():
                try:
                    count = collection.count()
                    status["collections"][name] = {
                        "count": count,
                        "status": "healthy"
                    }
                except Exception as e:
                    status["collections"][name] = {
                        "count": 0,
                        "status": f"error: {str(e)}"
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Initialize ChromaDB service
chroma_service = ChromaDBService()

# Initialize FastAPI app
app = FastAPI(
    title="ChromaDB Service for Rory's AI Assistant",
    description="Vector database service for knowledge base and conversation logging",
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
        "service": "ChromaDB Service for Rory's AI Assistant",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "add_documents": "/collections/{collection_name}/add",
            "query_documents": "/collections/{collection_name}/query",
            "collection_info": "/collections/{collection_name}/info",
            "log_conversation": "/conversations/log"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return chroma_service.health_check()

@app.post("/collections/{collection_name}/add")
async def add_documents(collection_name: str, request: DocumentRequest):
    """Add documents to collection"""
    return chroma_service.add_documents(
        collection_name,
        request.documents,
        request.metadatas,
        request.ids
    )

@app.post("/collections/{collection_name}/query")
async def query_documents(collection_name: str, request: QueryRequest):
    """Query documents from collection"""
    return chroma_service.query_documents(
        collection_name,
        request.query_texts,
        request.n_results,
        request.where,
        request.where_document
    )

@app.get("/collections/{collection_name}/info")
async def get_collection_info(collection_name: str):
    """Get collection information"""
    return chroma_service.get_collection_info(collection_name)

@app.post("/conversations/log")
async def log_conversation(request: ConversationRequest):
    """Log conversation"""
    return chroma_service.log_conversation(
        request.user_query,
        request.ai_response,
        request.chunks_used
    )

@app.get("/collections")
async def list_collections():
    """List all collections"""
    collections_info = {}
    for name in chroma_service.collections.keys():
        collections_info[name] = chroma_service.get_collection_info(name)
    return collections_info

# Background task for periodic maintenance
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("ChromaDB Service starting up...")
    logger.info(f"Service available at: http://0.0.0.0:8000")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("ChromaDB Service shutting down...")

if __name__ == "__main__":
    # Configuration
    HOST = "0.0.0.0"  # Listen on all interfaces
    PORT = 8000
    
    logger.info(f"Starting ChromaDB Service on {HOST}:{PORT}")
    
    # Run the service
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True
    )
