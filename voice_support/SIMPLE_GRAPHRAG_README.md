# Simple GraphRAG AI Assistant

A streamlined AI assistant system that meets your requirements:

## ✅ What You Asked For

1. **Keep previous webpage frontend design** - Clean, professional Streamlit interface
2. **GenAI using GraphRAG** - Implemented GraphRAG for intelligent knowledge extraction
3. **ChromaDB knowledge base** - All documents stored in persistent ChromaDB
4. **Powered by Qwen Max API** - All responses generated exclusively through Qwen Max API

## 🚀 Features

- **ChromaDB Integration**: Persistent knowledge base storage
- **GraphRAG Technology**: Enhanced retrieval with entity relationship mapping
- **Qwen Max API**: Professional AI responses
- **Document Processing**: Supports PDF, DOCX, TXT, and MD files with OCR fallback
- **Clean UI**: Simple, professional interface
- **Real-time Chat**: Interactive conversation with AI assistant

## 📁 Key Files

- `simple_graphrag_assistant.py` - Main application
- `simple_requirements.txt` - Required packages
- `chroma_db/` - Knowledge base storage (auto-created)

## 🔧 How It Works

1. **Document Processing**: Automatically processes all documents in the directory
2. **ChromaDB Storage**: Documents are embedded and stored in ChromaDB
3. **GraphRAG Enhancement**: Builds knowledge graph from extracted entities
4. **Query Processing**: 
   - User query → ChromaDB similarity search
   - GraphRAG entity extraction and relationship mapping
   - Context building with related information
5. **Response Generation**: Qwen Max API generates professional responses

## 🎯 System Architecture

```
User Query
    ↓
ChromaDB Query (similarity search)
    ↓
GraphRAG Processing (entity relationships)
    ↓
Context Building (combine results)
    ↓
Qwen Max API (response generation)
    ↓
Professional Response
```

## 📊 Current Status

- ✅ Running on http://localhost:8504
- ✅ ChromaDB initialized and ready
- ✅ GraphRAG system active
- ✅ Qwen Max API connected
- ✅ Documents processed and indexed

## 🔑 Configuration

The system uses your Qwen API key: `sk-015ea57c8b254c4181d30b2de4259d8b`

## 📝 Usage

1. Open http://localhost:8504 in your browser
2. Chat with the AI assistant about Rory's professional background
3. The system will use GraphRAG to provide intelligent, contextual responses
4. All responses are generated through Qwen Max API

## 🎨 Design Philosophy

- **Simplicity**: Clean, focused implementation
- **Efficiency**: Direct ChromaDB + GraphRAG + Qwen Max pipeline
- **Reliability**: Robust error handling and fallbacks
- **Professional**: Clean UI matching previous design aesthetic

This implementation gives you exactly what you requested: a simple, effective AI assistant using GraphRAG with ChromaDB and Qwen Max API, without the complexity of the previous versions.
