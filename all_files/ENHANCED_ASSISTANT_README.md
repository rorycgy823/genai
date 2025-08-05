# Enhanced Streamlit AI Assistant 🚀

A comprehensive AI assistant with advanced features including cost tracking, ChromaDB analytics, and multi-dashboard interface.

## 🌟 New Features

### ✅ **What's Been Added:**

1. **💰 Cost Calculation Dashboard**
   - Real-time token usage tracking
   - Cost estimation for Qwen API calls
   - Cumulative cost analytics
   - Per-conversation cost breakdown
   - Interactive cost visualization charts

2. **🔒 Secure API Configuration**
   - Removed API key from sidebar (security improvement)
   - Environment variable-based configuration
   - Secure credential handling

3. **📊 ChromaDB Visualization Dashboard**
   - Document analytics and statistics
   - Source file breakdown
   - Document type distribution
   - Chunk analysis and visualization
   - Interactive charts and graphs
   - Sample document preview

4. **🎛️ Multi-Page Dashboard Interface**
   - Tab-based navigation
   - Chat Assistant tab
   - Cost Analytics tab
   - ChromaDB Dashboard tab
   - Enhanced user experience

## 📁 File Structure

```
rory_personal_assistant/
├── enhanced_streamlit_assistant.py    # Main enhanced application
├── enhanced_requirements.txt          # Dependencies for enhanced version
├── streamlit_compatible_assistant.py  # Original version (rollback option)
├── inspect_chromadb.py               # ChromaDB inspection tool
├── ENHANCED_ASSISTANT_README.md      # This file
├── CHROMADB_INSPECTION_GUIDE.md      # ChromaDB inspection guide
└── ... (other files)
```

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
cd rory_personal_assistant
pip install -r enhanced_requirements.txt
```

### 2. **Set Environment Variables (Optional)**
```bash
# Set your Qwen API key (optional - has default for demo)
export QWEN_API_KEY="your-api-key-here"
```

### 3. **Launch Enhanced Assistant**
```bash
streamlit run enhanced_streamlit_assistant.py
```

### 4. **Access the Application**
- Open your browser to `http://localhost:8501`
- Navigate between the three main tabs:
  - 💬 **Chat Assistant**: Main chat interface
  - 📊 **Cost Analytics**: Token usage and cost tracking
  - 🗄️ **ChromaDB Dashboard**: Database analytics and visualization

## 🎯 Key Features

### **💬 Chat Assistant Tab**
- **Enhanced Chat Interface**: Professional AI assistant for Rory's career information
- **Real-time Cost Display**: Shows cost per query in chat
- **GraphRAG Integration**: Enhanced retrieval with entity relationships
- **Chunk Analysis**: Shows which knowledge chunks were used
- **Source Attribution**: Displays document sources for responses

### **📊 Cost Analytics Tab**
- **Cost Metrics**: Total cost, tokens used, conversation count
- **Interactive Charts**: 
  - Cumulative cost over time
  - Cost per conversation
  - Token usage distribution
  - Input vs output token analysis
- **Detailed Breakdown**: Timestamped cost table with full details
- **Cost Reset**: Option to reset tracking for new sessions

### **🗄️ ChromaDB Dashboard Tab**
- **Document Statistics**: Total docs, characters, sources, average length
- **Visual Analytics**:
  - Pie chart of documents by source
  - Bar chart of document types
  - Histogram of document length distribution
- **Chunk Analysis**: Total chunks and indexing information
- **Sample Documents**: Preview of stored documents with metadata
- **Real-time Data**: Live connection to cloud ChromaDB

## 🔧 Configuration Options

### **Sidebar Settings**
- **Text Chunking**: Adjust chunk size, overlap, and minimum size
- **Retrieval Settings**: Configure max chunks retrieved per query
- **System Status**: View library compatibility and current configuration
- **Actions**: Refresh system and reset cost tracking

### **Security Improvements**
- ✅ API key removed from sidebar
- ✅ Environment variable configuration
- ✅ Secure credential handling
- ✅ No sensitive data exposure in UI

## 📊 Analytics Features

### **Cost Tracking**
```python
# Automatic cost calculation for each API call
- Input tokens: Estimated from prompt length
- Output tokens: Estimated from response length
- Cost calculation: Based on Qwen API pricing
- Session tracking: Cumulative costs and statistics
```

### **ChromaDB Analytics**
```python
# Real-time database analysis
- Document count and character statistics
- Source file distribution
- Document type breakdown
- Chunk analysis and indexing
- Sample document preview
```

### **Visualization Charts**
- **Plotly Integration**: Interactive charts and graphs
- **Dark Theme**: Consistent with application design
- **Responsive Design**: Works on different screen sizes
- **Export Capability**: Charts can be downloaded

## 🛠️ Technical Architecture

### **Enhanced Components**
1. **CostCalculator**: Tracks API usage and calculates costs
2. **CloudChromaDBManager**: Enhanced with analytics methods
3. **QwenAPIClient**: Integrated with cost tracking
4. **Multi-tab Interface**: Streamlit tabs for different dashboards

### **Data Flow**
```
User Query → ChromaDB Retrieval → GraphRAG Enhancement → 
Qwen API Call → Cost Calculation → Response Display → 
Analytics Update
```

### **Storage**
- **Cloud ChromaDB**: Primary vector database (1.32.228.33:8000)
- **Session State**: Cost tracking and conversation history
- **In-Memory Fallback**: Automatic fallback if cloud unavailable

## 🔍 Monitoring & Analytics

### **Real-time Metrics**
- Session cost tracking
- Token usage monitoring
- Database statistics
- Performance metrics

### **Historical Analysis**
- Cost trends over time
- Usage patterns
- Document analytics
- Query performance

## 🚨 Troubleshooting

### **Common Issues**

1. **Cost Tracking Not Working**
   ```bash
   # Reset
