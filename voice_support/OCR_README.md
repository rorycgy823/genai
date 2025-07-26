# Rory's AI Assistant with OCR Support

An enhanced AI-powered assistant that can process both text documents and PDF files using OCR (Optical Character Recognition) technology. This application is specifically designed to answer questions about Rory Chen's professional background, experience, and expertise in data science and analytics.

## 🚀 Key Features

### Enhanced Document Processing
- **Text Files**: Direct processing of `.txt` and `.md` files
- **PDF Files**: Advanced PDF text extraction with OCR fallback
- **OCR Technology**: Supports both English and Chinese text recognition
- **Multi-Method Extraction**: Uses PyMuPDF, PyPDF2, and Tesseract OCR for maximum compatibility

### Intelligent AI System
- **GraphRAG Implementation**: Advanced retrieval-augmented generation
- **Intent Analysis**: Sophisticated query understanding and response generation
- **Factual Reasoning**: Can estimate age, calculate experience duration, and extract specific information
- **Caching System**: Optimized performance with intelligent response caching

### User Interface
- **Streamlit Web App**: Professional, responsive design
- **Interactive Chat**: Real-time conversation with the AI assistant
- **Sample Questions**: Pre-defined questions for quick exploration
- **Professional CV Display**: Comprehensive overview of Rory's background

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- Windows/Linux/macOS
- At least 4GB RAM (8GB recommended)
- Internet connection for AI model downloads

### Tesseract OCR Installation

#### Windows
1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location: `C:\Program Files\Tesseract-OCR\`
3. The application will automatically detect the installation

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-chi-sim  # For Chinese support
```

#### macOS
```bash
brew install tesseract
brew install tesseract-lang  # For additional language support
```

## 🛠️ Installation

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd voice_support
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r ocr_requirements.txt
```

### 4. Verify Tesseract Installation
```python
import pytesseract
print(pytesseract.get_tesseract_version())
```

## 🚀 Usage

### Running the Application
```bash
streamlit run rory_ai_assistant.py
```

The application will start and be available at: `http://localhost:8501`

### Document Processing
1. **Automatic Processing**: The app automatically processes all `.txt`, `.md`, and `.pdf` files in the current directory
2. **OCR Processing**: PDF files are processed using multiple extraction methods:
   - Text-based extraction (fastest)
   - OCR fallback for image-based PDFs
   - Chinese and English text recognition

### Interacting with the AI
1. **Direct Questions**: Type questions in the input box
2. **Sample Questions**: Click on pre-defined questions for quick answers
3. **Chat Interface**: Engage in ongoing conversations with the AI

## 📁 File Structure

```
voice_support/
├── rory_ai_assistant.py          # Main application with OCR support
├── ocr_requirements.txt          # Dependencies for OCR functionality
├── OCR_README.md                # This documentation
├── flomo文档.txt                # Chinese document (example)
├── CV_Rory_2507_DS_Analytics.txt # Rory's CV
├── kfs_mtg.txt                  # Meeting notes
└── sample_documents/            # Additional documents
```

## 🔧 Technical Architecture

### Document Processing Pipeline
1. **Text Files**: Direct UTF-8 encoding reading
2. **PDF Files**: Multi-stage extraction:
   - PyMuPDF for text-based PDFs
   - PyPDF2 as fallback
   - Tesseract OCR for image-based content

### AI Processing
1. **Vector Store**: Sentence-transformers for document embeddings
2. **Retrieval**: Cosine similarity search for relevant content
3. **Generation**: Qwen-Plus model for intelligent responses
4. **Fallback**: Local reasoning when API is unavailable

### Key Classes
- `EnhancedDocumentProcessor`: Handles OCR and document processing
- `OptimizedVectorStore`: Manages document embeddings and search
- `IntelligentGraphRAGChain`: Orchestrates AI reasoning and response generation

## 🎯 Supported Query Types

### Factual Information
- Age estimation based on education timeline
- Experience duration calculations
- Contact information extraction
- Educational background details

### Analytical Queries
- Career progression analysis
- Technical skills assessment
- Industry experience evaluation
- Unique value proposition identification

### Conversational Queries
- General questions about Rory's background
- Project and achievement discussions
- Professional insights and perspectives

## 🔍 OCR Capabilities

### Supported Languages
- English (primary)
- Chinese Simplified (chi_sim)
- Additional languages can be added via Tesseract language packs

### PDF Processing Features
- **Text-based PDFs**: Direct text extraction
- **Image-based PDFs**: OCR conversion to text
- **Mixed PDFs**: Combination of both methods
- **Multi-page Support**: Processes entire documents
- **Error Handling**: Graceful fallback for problematic files

## 🚨 Troubleshooting

### Common Issues

#### Tesseract Not Found
```
Error: Tesseract not found
```
**Solution**: Install Tesseract OCR and ensure it's in your system PATH

#### PDF Processing Fails
```
Error: Failed to process PDF
```
**Solutions**:
1. Check if the PDF is corrupted
2. Ensure sufficient memory is available
3. Try with a smaller PDF file first

#### Chinese Text Not Recognized
```
Warning: OCR failed for Chinese text
```
**Solution**: Install Chinese language pack for Tesseract:
```bash
# Linux
sudo apt install tesseract-ocr-chi-sim

# Windows: Download from Tesseract GitHub releases
```

#### Memory Issues
```
Error: Out of memory during processing
```
**Solutions**:
1. Process fewer documents at once
2. Increase system RAM
3. Close other applications

## 📊 Performance Optimization

### Document Processing
- **Chunking**: Documents split into 300-word chunks with 30-word overlap
- **Filtering**: Very short chunks (<50 words) are filtered out
- **Batch Processing**: Embeddings generated in batches of 32

### OCR Optimization
- **Page Limiting**: OCR limited to first 10 pages for very large PDFs
- **Resolution Control**: Optimized image resolution for OCR accuracy
- **Language Detection**: Automatic language detection for better results

## 🔐 Security Considerations

### API Keys
- API keys are embedded in the code for demo purposes
- For production use, store API keys in environment variables:
```python
import os
api_key = os.getenv('QWEN_API_KEY')
```

### Data Privacy
- All document processing happens locally
- Only query text is sent to the AI API
- No document content is transmitted externally

## 🤝 Contributing

### Adding New Document Types
1. Extend `EnhancedDocumentProcessor.process_documents()`
2. Add new file extension handling
3. Implement appropriate text extraction method

### Improving OCR Accuracy
1. Adjust image preprocessing parameters
2. Add language-specific optimizations
3. Implement confidence scoring

### Enhancing AI Responses
1. Modify prompt templates in `_generate_intelligent_response()`
2. Add new intent types in `_analyze_query_intent()`
3. Extend factual reasoning methods

## 📞 Support

For technical support or questions about Rory's background:
- **Email**: chengy823@gmail.com
- **Location**: Hong Kong SAR
- **LinkedIn**: Available through the AI assistant

## 📄 License

This project is created for demonstration purposes. Please respect privacy and use responsibly.

---

**Note**: This AI assistant is designed to provide information about Rory Chen's professional background based on available documents. All responses are generated based on the processed document content and should be verified for accuracy in professional contexts.
