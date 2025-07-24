# GraphRAG Voice Support Assistant (Citic智能助手)

A sophisticated Chinese language voice support assistant powered by GraphRAG (Graph-based Retrieval Augmented Generation) technology. This system builds a knowledge graph from PDF documents and provides intelligent question-answering capabilities.

## Features

### 🧠 Advanced GraphRAG Technology
- **Knowledge Graph Construction**: Automatically builds a comprehensive knowledge graph from PDF documents
- **Entity Extraction**: Identifies Chinese organizations, financial terms, dates, and numerical values
- **Graph Walking Algorithm**: Uses breadth-first search to discover contextually related information
- **Hybrid Retrieval**: Combines graph-based retrieval with traditional similarity search

### 🇨🇳 Chinese Language Optimization
- **Specialized Entity Patterns**: Custom regex patterns for Chinese financial documents
- **Multilingual Embeddings**: Uses paraphrase-multilingual-MiniLM-L12-v2 for semantic understanding
- **Chinese Text Processing**: Optimized text splitting with Chinese punctuation support
- **Financial Terminology**: Recognizes key Chinese financial terms and concepts

### 🚀 Performance Benefits
- **Improved Search Efficiency**: Graph structure reduces search space through intelligent entity relationships
- **Better Contextual Understanding**: Entity relationships provide deeper semantic understanding
- **Enhanced Accuracy**: Graph-based retrieval finds more contextually relevant information
- **Scalable Design**: Architecture scales well with increasing document size and complexity

## System Architecture

### Core Components
1. **GraphRAG Class**: Document processing, entity extraction, and graph construction
2. **SimpleGraphRAGRetriever**: Clean retriever implementation without inheritance issues
3. **SimpleQAChain**: Custom QA chain with multiple method interfaces
4. **Streamlit Interface**: User-friendly chat interface with loading states

### GraphRAG Workflow
1. **Document Processing**: PDF text split into chunks with Chinese-optimized separators
2. **Entity Extraction**: Regex patterns identify Chinese financial entities, dates, amounts
3. **Graph Construction**: NetworkX graph with entity and chunk nodes
4. **Query Processing**: Semantic similarity to find relevant entities
5. **Graph Walking**: BFS algorithm traverses graph to find connected chunks
6. **Response Generation**: Tongyi LLM generates answers based on top-ranked chunks

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
1. Clone the repository:
```bash
git clone <your-repo-url>
cd voice_support
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API key:
   - Update the Tongyi API key in `fixed_main.py`
   - Replace `"sk-015ea57c8b254c4181d30b2de4259d8b"` with your actual API key

4. Add your PDF document:
   - Place your PDF file in the project directory
   - Update the file path in `fixed_main.py`

## Usage

### Running the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Using the Assistant
1. Open the web interface
2. Type your question in Chinese in the chat input
3. The system will:
   - Extract relevant entities from your query
   - Walk the knowledge graph to find related information
   - Generate a comprehensive answer with source attribution

## File Structure

```
voice_support/
├── app.py                 # Streamlit web interface
├── fixed_main.py         # Main GraphRAG implementation (recommended)
├── main.py               # Original GraphRAG implementation
├── simple_main.py        # Simple RAG fallback
├── debug_test.py         # Debug utilities
├── test_app.py           # Test interface
├── kfs_mtg.pdf          # Sample PDF document
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Technical Details

### Entity Extraction Patterns
- **Organizations**: 公司, 银行, 保险, 基金, 证券, 投资, 集团, 有限, 股份
- **Financial Amounts**: 元, 万, 亿, 千, 百, %
- **Dates**: 年月日 format variations
- **Financial Terms**: 资产, 负债, 收入, 支出, 利润, 亏损, 投资, 融资, 贷款, 存款, 利率, 汇率

### Graph Structure
- **Nodes**: Entities and document chunks
- **Edges**: Contains relationships (entity-chunk) and co-occurrence relationships (entity-entity)
- **Weights**: Based on co-occurrence frequency

### Retrieval Process
1. Query → Relevant Entities (top 10 by semantic similarity)
2. Entities → Graph Walk (BFS, max depth 2) → Connected Chunks
3. Chunks → Similarity Ranking → Top 5 Results
4. Fallback to traditional similarity search if needed

## Configuration

### Model Settings
- **Embedding Model**: paraphrase-multilingual-MiniLM-L12-v2
- **LLM**: Tongyi Qwen-max
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Retrieval Count**: 5 documents

### Customization
- Modify entity extraction patterns in `_extract_entities()` method
- Adjust graph walking parameters (max_depth, top_k)
- Update prompt template for different use cases
- Configure different embedding models or LLMs

## Troubleshooting

### Common Issues
1. **Recursion Errors**: Use `fixed_main.py` instead of `main.py`
2. **API Key Issues**: Ensure valid Tongyi API key is configured
3. **PDF Path Issues**: Verify PDF file path is correct
4. **Memory Issues**: Reduce chunk size or document size for large files

### Performance Optimization
- Use GPU acceleration for embeddings if available
- Implement caching for frequently accessed entities
- Optimize graph structure for specific document types

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the RAG framework
- NetworkX for graph operations
- Streamlit for the web interface
- SentenceTransformers for multilingual embeddings
- Tongyi for Chinese language model support
