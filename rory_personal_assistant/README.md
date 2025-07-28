# Rory's Personal AI Assistant

A sophisticated AI assistant powered by GraphRAG technology, ChromaDB knowledge base, and Qwen Max API.

## 🚀 Features

- **GraphRAG Technology**: Intelligent knowledge extraction with entity relationship mapping
- **ChromaDB Integration**: Persistent knowledge base storage with automatic document processing
- **Qwen Max API**: Professional AI responses with comprehensive reasoning
- **Dark Theme UI**: Modern, professional interface with excellent visibility
- **Conversation Logging**: All interactions stored for model improvement
- **OCR Support**: Enhanced PDF processing with EasyOCR and Tesseract fallback for image-based text
- **Real-time Analytics**: Query analysis with chunk and entity information

## 📁 Project Structure

```
rory_personal_assistant/
├── simple_graphrag_assistant.py    # Main application
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
└── chroma_db/                      # Knowledge base (auto-created)
```

## 🔧 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Set your Qwen API key in the sidebar or as an environment variable:
```bash
export QWEN_API_KEY="your-api-key-here"
```

### 3. Run the Application
```bash
streamlit run simple_graphrag_assistant.py --server.port 8504
```

### 4. Access the Interface
Open your browser and navigate to: `http://localhost:8504`

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
Professional Response + Analytics
```

## 🧠 How It Works

### **Document Processing**
1. Automatically scans directory for documents (PDF, DOCX, TXT, MD)
2. Extracts text using multiple methods including OCR fallback
3. **OCR Fallback**: When regular PDF text extraction fails, automatically uses:
   - **EasyOCR**: Advanced OCR with multi-language support (English, Chinese)
   - **Tesseract**: Backup OCR engine for reliable text recognition
   - **Image Processing**: Converts PDF pages to images for OCR analysis
4. Stores documents in ChromaDB with metadata

### **GraphRAG Enhancement**
1. Extracts entities (companies, skills, positions, years, industries)
2. Builds knowledge graph with entity relationships
3. Provides related entities for enhanced context

### **Query Processing**
1. **ChromaDB Search**: Finds relevant document chunks
2. **GraphRAG Analysis**: Identifies related entities and relationships
3. **Context Building**: Combines search results with graph insights
4. **Response Generation**: Qwen Max API creates professional responses

### **Analytics & Logging**
1. **Real-time Metrics**: Shows chunks used, entities found, sources
2. **Conversation Storage**: All Q&A pairs stored in ChromaDB
3. **Privacy Transparency**: Clear notice about data collection

## 🎨 Interface Features

### **Dark Theme Design**
- Professional dark color scheme
- High contrast for excellent readability
- Styled components with proper shadows and borders

### **System Status Display**
- Knowledge chunks count
- Conversations logged
- GraphRAG status
- API connectivity

### **Query Analytics**
- Number of knowledge chunks used
- GraphRAG entities identified
- Source documents referenced
- Technical processing details

## 📊 Technical Specifications

### **Core Technologies**
- **Streamlit**: Web interface framework
- **ChromaDB**: Vector database for knowledge storage
- **NetworkX**: Graph processing for entity relationships
- **PyMuPDF/PyPDF2**: PDF text extraction with OCR fallback
- **python-docx**: DOCX document processing
- **Requests**: HTTP client for API calls

### **AI Components**
- **Qwen Max API**: Large language model for response generation
- **Sentence Transformers**: Text embeddings (optional enhancement)
- **GraphRAG**: Custom implementation for knowledge graph reasoning

### **Data Storage**
- **Knowledge Base**: ChromaDB collection for documents
- **Conversations**: Separate ChromaDB collection for Q&A logging
- **Metadata**: Document sources, types, timestamps, analytics

## 🔒 Privacy & Data Handling

### **Data Collection**
- User queries and AI responses are stored in ChromaDB
- Metadata includes timestamps, chunks used, and processing details
- All data used for model improvement and learning

### **Privacy Notice**
- Clear disclosure in the interface
- Users informed about data collection purposes
- Transparent about storage and usage

### **Data Security**
- Local ChromaDB storage
- No external data transmission except to Qwen API
- User control over data through clear chat functionality

## 🚀 Usage Examples

### **Professional Queries**
- "What is Rory's experience in machine learning?"
- "Tell me about Rory's career progression"
- "What projects has Rory worked on at CITIC Bank?"

### **Technical Questions**
- "What programming languages does Rory use?"
- "Describe Rory's cloud platform experience"
- "What achievements has Rory made in data science?"

### **Industry Insights**
- "Why did Rory transition from market research to banking?"
- "What industries has Rory worked in?"
- "How has Rory's role evolved over time?"

## 📈 Performance Metrics

### **System Capabilities**
- **Response Time**: Typically 3-10 seconds depending on query complexity
- **Knowledge Base**: Processes unlimited documents automatically
- **Accuracy**: Enhanced by GraphRAG entity relationship mapping
- **Scalability**: ChromaDB handles large document collections efficiently

### **Analytics Tracking**
- **Chunks Used**: Number of document segments retrieved per query
- **GraphRAG Entities**: Related entities identified and processed
- **Source Attribution**: Original documents providing context
- **Conversation History**: Complete interaction logging

## 🛠️ Troubleshooting

### **Common Issues**

**API Connection Problems**
- Verify Qwen API key is correct
- Check internet connectivity
- Ensure API quota is available

**Document Processing Issues**
- Verify document formats are supported (PDF, DOCX, TXT, MD)
- Check file permissions and accessibility
- Review ChromaDB initialization logs

**Performance Optimization**
- Reduce query complexity for faster responses
- Clear chat history periodically
- Monitor ChromaDB storage usage

## 📞 Contact Information

**Rory Chen**
- **Email**: chengy823@gmail.com
- **Phone**: 68746551
- **Position**: AVP of Data Science, China CITIC Bank International
- **Location**: Hong Kong SAR

## 📝 License & Usage

This project is designed for professional demonstration and personal use. The AI assistant provides information about Rory Chen's professional background and capabilities.

---

**Built with ❤️ using GraphRAG, ChromaDB, and Qwen Max API**
