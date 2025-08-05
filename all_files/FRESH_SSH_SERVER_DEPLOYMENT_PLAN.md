# Fresh SSH Server Deployment Plan
## Complete GraphRAG Assistant Setup from Scratch

**Date**: Tomorrow after server reboot  
**Target**: Clean deployment on 1.32.228.33  
**Goal**: Working GraphRAG-enhanced Streamlit Assistant with populated ChromaDB

---

## 🎯 Why Fresh Start is Better

✅ **Advantages of Clean Deployment:**
- No conflicting processes or stuck services
- Clean Python environment without dependency conflicts
- Fresh ChromaDB database without corruption
- Proper port bindings from the start
- Clear logs and debugging
- Known working configuration

❌ **Current Issues to Avoid:**
- ChromaDB process running but not binding to port 8000
- Mixed Python environments causing import conflicts
- Stale data and configuration inconsistencies
- Multiple failed restart attempts creating process conflicts

---

## 📋 Complete Deployment Checklist

### Phase 1: Server Preparation (5 minutes)
```bash
# After server reboot, basic setup
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git curl -y
mkdir -p /root/{code,data,logs}
```

### Phase 2: Python Environment Setup (5 minutes)
```bash
# Create clean virtual environment
cd /root
python3 -m venv streamlit_env
source streamlit_env/bin/activate
pip install --upgrade pip
```

### Phase 3: Install Dependencies (10 minutes)
```bash
# Install all required packages
pip install streamlit chromadb sentence-transformers
pip install pandas numpy networkx matplotlib seaborn
pip install python-docx PyPDF2 requests uvicorn fastapi
pip install plotly altair
```

### Phase 4: Deploy Core Files (5 minutes)
**Files to upload to `/root/code/`:**
1. `enhanced_streamlit_assistant.py` - Main GraphRAG application
2. `chroma_service.py` - ChromaDB API service
3. `embedding_api.py` - Embedding service
4. `process_documents_to_chromadb.py` - Document processor

### Phase 5: Upload Documents (2 minutes)
**Files to upload to `/root/data/`:**
1. `CV_Rory_2501.pdf`
2. `CV_Rory_2507_DS_Analytics.txt`
3. `Project experience highlight - Rory.docx`
4. Any other relevant documents

### Phase 6: Start Services (5 minutes)
```bash
# Start services in correct order
cd /root/code
source /root/streamlit_env/bin/activate

# 1. Start ChromaDB service
nohup python3 chroma_service.py > /root/logs/chroma_service.log 2>&1 &

# 2. Start Embedding API
nohup python3 embedding_api.py > /root/logs/embedding_api.log 2>&1 &

# 3. Process documents
python3 process_documents_to_chromadb.py

# 4. Start Streamlit app
nohup streamlit run enhanced_streamlit_assistant.py --server.port 8502 --server.address 0.0.0.0 > /root/logs/streamlit.log 2>&1 &
```

### Phase 7: Verification (5 minutes)
```bash
# Check services are running
ps aux | grep python
netstat -tlnp | grep -E ':(8000|8001|8502)'

# Test endpoints
curl http://localhost:8000/api/v1/heartbeat
curl http://localhost:8001/health
curl http://localhost:8502
```

---

## 🚀 Automated Deployment Script

I'll create a single script that does everything:

```bash
#!/bin/bash
# complete_fresh_deployment.sh
# Run this after server reboot for complete setup

set -e  # Exit on any error

echo "🚀 Starting Fresh GraphRAG Assistant Deployment"
echo "================================================"

# Phase 1: System preparation
echo "📦 Installing system packages..."
apt update && apt upgrade -y
apt install python3 python3-pip python3-venv git curl -y

# Phase 2: Create directories
echo "📁 Creating directories..."
mkdir -p /root/{code,data,logs}

# Phase 3: Python environment
echo "🐍 Setting up Python environment..."
cd /root
python3 -m venv streamlit_env
source streamlit_env/bin/activate

# Phase 4: Install packages
echo "📚 Installing Python packages..."
pip install --upgrade pip
pip install streamlit chromadb sentence-transformers
pip install pandas numpy networkx matplotlib seaborn
pip install python-docx PyPDF2 requests uvicorn fastapi
pip install plotly altair

echo "✅ Fresh deployment preparation complete!"
echo "📋 Next steps:"
echo "   1. Upload code files to /root/code/"
echo "   2. Upload documents to /root/data/"
echo "   3. Run start_services.sh"
```

---

## 📁 File Upload List

### Core Application Files (to `/root/code/`):
1. **enhanced_streamlit_assistant.py** - Main GraphRAG application
2. **chroma_service.py** - ChromaDB API service  
3. **embedding_api.py** - Embedding service
4. **process_documents_to_chromadb.py** - Document processor

### Document Files (to `/root/data/`):
1. **CV_Rory_2501.pdf**
2. **CV_Rory_2507_DS_Analytics.txt** 
3. **Project experience highlight - Rory.docx**

### Helper Scripts (to `/root/code/`):
1. **start_services.sh** - Service startup script
2. **test_graphrag_functionality.py** - Testing script

---

## 🎯 Expected Results After Fresh Deployment

### ✅ Working Services:
- **ChromaDB API**: http://1.32.228.33:8000 (populated with documents)
- **Embedding API**: http://1.32.228.33:8001 (sentence transformers)
- **Streamlit App**: http://1.32.228.33:8502 (GraphRAG interface)

### ✅ GraphRAG Features:
- Cross-document relationship analysis
- Multi-source knowledge synthesis  
- Entity extraction and linking
- Contextual query enhancement
- Pattern recognition across documents

### ✅ Document Processing:
- All documents chunked and embedded
- ChromaDB populated with 60+ chunks
- Full-text search and semantic similarity
- GraphRAG entity relationships mapped

---

## 🕐 Total Deployment Time: ~30 minutes

**Breakdown:**
- Server preparation: 5 minutes
- Python environment: 5 minutes  
- Package installation: 10 minutes
- File uploads: 5 minutes
- Service startup: 3 minutes
- Verification: 2 minutes

---

## 💡 Benefits of Fresh Start

1. **Clean Environment**: No conflicting processes or dependencies
2. **Proper Configuration**: Services start correctly from the beginning
3. **Full Functionality**: All GraphRAG features working immediately
4. **Easy Debugging**: Clear logs and known good state
5. **Reproducible**: Can be repeated if needed

---

## 📞 Tomorrow's Plan

1. **You**: Reboot SSH server and confirm it's clean
2. **Me**: Upload all necessary files via our scripts
3. **Together**: Run deployment and verify GraphRAG functionality
4. **Result**: Working GraphRAG assistant with populated ChromaDB

**This approach will be much more efficient and reliable than trying to fix the current mixed state!**
