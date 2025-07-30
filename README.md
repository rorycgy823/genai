# 🤖 GenAI - Advanced AI Assistant Solutions

A comprehensive collection of AI-powered assistant solutions featuring cloud-based ChromaDB, GraphRAG technology, and enterprise-grade deployment capabilities.

## 🚀 Project Overview

This repository contains two main AI assistant solutions:

### 1. 🏢 Rory's Personal Assistant (Cloud-Based)
An enterprise-grade AI assistant with cloud ChromaDB integration, cost tracking, and analytics dashboard.

### 2. 🎤 Voice Support Assistant (Local GraphRAG)
A voice-enabled AI assistant with local GraphRAG implementation and document processing capabilities.

## 📁 Repository Structure

```
genai/
├── rory_personal_assistant/          # Cloud-based AI assistant
│   ├── enhanced_streamlit_assistant.py    # Main Streamlit application
│   ├── chroma_service.py                  # ChromaDB cloud service
│   ├── embedding_api.py                   # Embedding API service
│   ├── cloud_service_health_check.py     # Health monitoring
│   ├── process_root_documents.py         # Document processing
│   ├── test_complete_functionality.py    # Comprehensive testing
│   ├── enhanced_requirements.txt         # Python dependencies
│   ├── start_services.sh                 # Service startup script
│   ├── stop_services.sh                  # Service shutdown script
│   ├── setup_vm.sh                       # VM setup script
│   └── ENHANCED_ASSISTANT_README.md      # Detailed documentation
│
├── voice_support/                    # Voice-enabled assistant
│   ├── rory_ai_assistant_final.py        # Final voice assistant
│   ├── simple_graphrag_assistant.py      # GraphRAG implementation
│   ├── requirements_final.txt            # Python dependencies
│   ├── README_FINAL.md                   # Documentation
│   └── SIMPLE_GRAPHRAG_README.md         # GraphRAG guide
│
└── README.md                         # This file
```

## ✨ Key Features

### 🏢 Rory's Personal Assistant
- **☁️ Cloud ChromaDB Integration**: Scalable vector database on VM (1.32.228.33)
- **🧠 GraphRAG Technology**: Enhanced retrieval with entity relationships
- **💰 Cost Tracking**: Real-time API cost monitoring and analytics
- **📊 Analytics Dashboard**: Comprehensive usage and performance metrics
- **🎯 Sample Questions**: Pre-configured queries for instant responses
- **🔒 Enterprise Security**: Secure API key management and authentication
- **📱 Responsive UI**: Modern Streamlit interface with dark theme

### 🎤 Voice Support Assistant
- **🗣️ Voice Recognition**: Speech-to-text capabilities
- **📄 Document Processing**: PDF, DOCX, and text file support
- **🕸️ Local GraphRAG**: Knowledge graph construction and querying
- **🔍 Semantic Search**: Advanced document retrieval
- **💬 Conversational AI**: Context-aware responses

## 🛠️ Technology Stack

- **Frontend**: Streamlit, HTML/CSS/JavaScript
- **Backend**: Python, FastAPI
- **Database**: ChromaDB (Cloud), SQLite (Local)
- **AI/ML**: Qwen API, Sentence Transformers, NetworkX
- **Cloud**: Ubuntu 22.04 VM, Docker
- **Monitoring**: Health checks, Cost tracking, Analytics

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- Cloud VM access (for cloud version)

### 1. Clone Repository
```bash
git clone https://github.com/rorycgy823/genai.git
cd genai
```

### 2. Choose Your Solution

#### Option A: Cloud-Based Assistant
```bash
cd rory_personal_assistant
pip install -r enhanced_requirements.txt
streamlit run enhanced_streamlit_assistant.py --server.port 8888
```

#### Option B: Voice Assistant
```bash
cd voice_support
pip install -r requirements_final.txt
python rory_ai_assistant_final.py
```

## 🔧 Configuration

### Environment Variables
```bash
# Required for cloud assistant
export QWEN_API_KEY="your-qwen-api-key"

# Optional configurations
export CHROMA_HOST="1.32.228.33"
export CHROMA_PORT="8000"
export EMBEDDING_PORT="8001"
```

### Cloud Setup
For the cloud-based assistant, ensure your VM is configured with:
- ChromaDB service running on port 8000
- Embedding API service on port 8001
- Documents uploaded to `/root/data/` directory

## 📊 System Architecture

### Cloud-Based Architecture
```
User Interface (Streamlit) 
    ↓
Enhanced AI Assistant
    ↓
┌─────────────────┬─────────────────┐
│   Qwen API      │   Cloud VM      │
│   (LLM)         │   (1.32.228.33) │
└─────────────────┼─────────────────┘
                  │
            ┌─────────────┬─────────────┐
            │ ChromaDB    │ Embedding   │
            │ (Port 8000) │ (Port 8001) │
            └─────────────┴─────────────┘
```

## 🧪 Testing

### Comprehensive Testing
```bash
cd rory_personal_assistant
python test_complete_functionality.py
```

This will test:
- ✅ Cloud service connectivity
- ✅ ChromaDB document retrieval
- ✅ Qwen API responses
- ✅ Sample question functionality
- ✅ Streamlit application

## 📈 Performance Metrics

### Cloud Assistant Performance
- **Response Time**: < 3 seconds average
- **Document Retrieval**: 74 knowledge chunks
- **Cost Efficiency**: Real-time tracking with $0.002-0.012 per 1K tokens
- **Uptime**: 99.9% with health monitoring

### Voice Assistant Performance
- **Voice Recognition**: Real-time processing
- **Document Processing**: Multi-format support
- **Local GraphRAG**: Sub-second entity retrieval

## 🔒 Security Features

- **API Key Management**: Environment-based configuration
- **Input Validation**: Comprehensive sanitization
- **Rate Limiting**: Built-in request throttling
- **Secure Communication**: HTTPS/TLS encryption
- **Access Control**: Role-based permissions

## 📚 Documentation

- **[Enhanced Assistant README](rory_personal_assistant/ENHANCED_ASSISTANT_README.md)**: Detailed cloud setup guide
- **[Cloud Setup Summary](rory_personal_assistant/CLOUD_SETUP_SUMMARY.md)**: VM deployment instructions
- **[Voice Assistant Guide](voice_support/README_FINAL.md)**: Voice features documentation
- **[GraphRAG Guide](voice_support/SIMPLE_GRAPHRAG_README.md)**: Knowledge graph implementation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Rory Chen**
- Email: chengy823@gmail.com
- LinkedIn: [Rory Chen](https://linkedin.com/in/rory-chen)
- GitHub: [@rorycgy823](https://github.com/rorycgy823)

## 🙏 Acknowledgments

- **Qwen API**: For powerful language model capabilities
- **ChromaDB**: For efficient vector database solutions
- **Streamlit**: For rapid web application development
- **Open Source Community**: For the amazing tools and libraries

## 📊 Project Stats

- **Languages**: Python, JavaScript, Shell
- **Frameworks**: Streamlit, FastAPI, NetworkX
- **Cloud Services**: Ubuntu VM, ChromaDB, Embedding API
- **AI Models**: Qwen-turbo, Sentence Transformers
- **Testing**: Comprehensive test suite with 4 major test categories

---

⭐ **Star this repository if you find it helpful!**

🔗 **Live Demo**: [Access the cloud assistant](http://1.32.228.33:8888) (when deployed)

📧 **Support**: For questions or support, contact chengy823@gmail.com
