"""
Streamlit Cloud Compatible AI Assistant
=====================================

A simplified version optimized for Streamlit Cloud deployment:
- Minimal dependencies for better compatibility
- Optional imports with fallbacks
- Cloud ChromaDB integration
- Cost tracking and analytics

Author: Rory Chen
Server: 1.32.228.33 (Ubuntu-22.04-x64)
"""

import streamlit as st
import os
import requests
import json
from typing import List, Dict, Any
import uuid
from datetime import datetime, timedelta
import re
from collections import defaultdict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional imports with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    # Simple fallback for graph functionality
    class SimpleGraph:
        def __init__(self):
            self.nodes_dict = {}
            self.edges_dict = defaultdict(list)
        
        def add_node(self, node, **attrs):
            self.nodes_dict[node] = attrs
        
        def add_edge(self, node1, node2, **attrs):
            self.edges_dict[node1].append((node2, attrs))
            self.edges_dict[node2].append((node1, attrs))
        
        def has_node(self, node):
            return node in self.nodes_dict
        
        def has_edge(self, node1, node2):
            return any(n[0] == node2 for n in self.edges_dict.get(node1, []))
        
        def neighbors(self, node):
            return [n[0] for n in self.edges_dict.get(node, [])]
        
        def nodes(self):
            return self.nodes_dict.keys()
    
    nx = type('nx', (), {'Graph': SimpleGraph})()

# Try to import ChromaDB with fallback
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except Exception as e:
    CHROMADB_AVAILABLE = False
    CHROMADB_ERROR = str(e)

# Document processing libraries with fallbacks
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except:
    PYPDF2_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except:
    DOCX_AVAILABLE = False

class CostCalculator:
    """Calculate API costs and token usage"""
    
    def __init__(self):
        # Qwen API pricing
        self.pricing = {
            "qwen-turbo": {
                "input_tokens": 0.0015,  # per 1K tokens
                "output_tokens": 0.002,  # per 1K tokens
            },
            "qwen-plus": {
                "input_tokens": 0.008,
                "output_tokens": 0.012,
            }
        }
        
        # Initialize session state for cost tracking
        if "total_cost" not in st.session_state:
            st.session_state.total_cost = 0.0
        if "total_input_tokens" not in st.session_state:
            st.session_state.total_input_tokens = 0
        if "total_output_tokens" not in st.session_state:
            st.session_state.total_output_tokens = 0
        if "conversation_costs" not in st.session_state:
            st.session_state.conversation_costs = []
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return len(text) // 4
    
    def calculate_cost(self, input_text: str, output_text: str, model: str = "qwen-turbo") -> Dict[str, Any]:
        """Calculate cost for a single API call"""
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        
        if model not in self.pricing:
            model = "qwen-turbo"
        
        input_cost = (input_tokens / 1000) * self.pricing[model]["input_tokens"]
        output_cost = (output_tokens / 1000) * self.pricing[model]["output_tokens"]
        total_cost = input_cost + output_cost
        
        # Update session totals
        st.session_state.total_cost += total_cost
        st.session_state.total_input_tokens += input_tokens
        st.session_state.total_output_tokens += output_tokens
        
        # Log this conversation cost
        cost_record = {
            "timestamp": datetime.now(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "model": model
        }
        st.session_state.conversation_costs.append(cost_record)
        
        return cost_record
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary"""
        return {
            "total_cost": st.session_state.total_cost,
            "total_input_tokens": st.session_state.total_input_tokens,
            "total_output_tokens": st.session_state.total_output_tokens,
            "total_tokens": st.session_state.total_input_tokens + st.session_state.total_output_tokens,
            "conversation_count": len(st.session_state.conversation_costs),
            "avg_cost_per_conversation": st.session_state.total_cost / max(1, len(st.session_state.conversation_costs))
        }
    
    def reset_costs(self):
        """Reset all cost tracking"""
        st.session_state.total_cost = 0.0
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.session_state.conversation_costs = []

class InMemoryStorage:
    """Fallback in-memory storage when ChromaDB is not available"""
    
    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.conversations = []
        
    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to in-memory storage"""
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        return True
    
    def query_documents(self, query: str, n_results: int = 5) -> Dict:
        """Simple text matching for document retrieval"""
        results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        if not self.documents:
            return results
        
        # Simple keyword matching
        query_lower = query.lower()
        matches = []
        
        for i, doc in enumerate(self.documents):
            if any(word in doc.lower() for word in query_lower.split()):
                matches.append((doc, self.metadatas[i], 0.5))
        
        # Sort by relevance
        matches.sort(key=lambda x: sum(word in x[0].lower() for word in query_lower.split()), reverse=True)
        
        # Return top results
        top_matches = matches[:n_results]
        if top_matches:
            results["documents"] = [[match[0] for match in top_matches]]
            results["metadatas"] = [[match[1] for match in top_matches]]
            results["distances"] = [[match[2] for match in top_matches]]
        
        return results
    
    def get_collection_count(self) -> int:
        return len(self.documents)
    
    def log_conversation(self, user_query: str, ai_response: str, chunks_used: int = 0):
        """Log conversation to in-memory storage"""
        conversation = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "ai_response": ai_response,
            "chunks_used": chunks_used
        }
        self.conversations.append(conversation)
        return True
    
    def get_conversation_count(self) -> int:
        return len(self.conversations)

class CloudChromaDBManager:
    """Manages ChromaDB operations with cloud server HTTP API connection"""
    
    def __init__(self, host: str = "1.32.228.33", chroma_port: int = 8000, embedding_port: int = 8001):
        self.host = host
        self.chroma_port = chroma_port
        self.embedding_port = embedding_port
        self.chroma_url = f"http://{host}:{chroma_port}"
        self.embedding_url = f"http://{host}:{embedding_port}"
        
        # Test connection to HTTP API services
        try:
            # Test ChromaDB service health
            chroma_response = requests.get(f"{self.chroma_url}/health", timeout=10)
            embedding_response = requests.get(f"{self.embedding_url}/health", timeout=10)
            
            if chroma_response.status_code == 200 and embedding_response.status_code == 200:
                self.storage_type = "Cloud HTTP API ✅"
                self.connection_status = "Connected to VM 1.32.228.33"
                self.use_fallback = False
            else:
                raise Exception(f"Service health check failed")
                
        except Exception as e:
            self.storage = InMemoryStorage()
            self.storage_type = "In-Memory (Fallback) ⚠️"
            self.connection_status = f"Cloud connection failed: {str(e)}"
            self.use_fallback = True
    
    def query_documents(self, query: str, n_results: int = 5) -> Dict:
        """Query documents from storage via HTTP API"""
        if not self.use_fallback:
            try:
                data = {
                    "query_texts": [query],
                    "n_results": n_results
                }
                
                response = requests.post(
                    f"{self.chroma_url}/collections/knowledge_base/query",
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
                    
            except Exception as e:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        else:
            return self.storage.query_documents(query, n_results)
    
    def get_collection_count(self) -> int:
        """Get total number of documents in collection via HTTP API"""
        if not self.use_fallback:
            try:
                response = requests.get(f"{self.chroma_url}/collections", timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    knowledge_base_info = result.get("knowledge_base", {})
                    return knowledge_base_info.get("count", 0)
                else:
                    return 0
                    
            except Exception as e:
                return 0
        else:
            return self.storage.get_collection_count()
    
    def log_conversation(self, user_query: str, ai_response: str, chunks_used: int = 0):
        """Log conversation for model improvement via HTTP API"""
        if not self.use_fallback:
            try:
                data = {
                    "user_query": user_query,
                    "ai_response": ai_response,
                    "chunks_used": chunks_used
                }
                
                response = requests.post(
                    f"{self.chroma_url}/conversations/log",
                    json=data,
                    timeout=15
                )
                
                return response.status_code == 200
                
            except Exception as e:
                return False
        else:
            return self.storage.log_conversation(user_query, ai_response, chunks_used)

class SimpleGraphRAGProcessor:
    """Simplified GraphRAG implementation without networkx dependency"""
    
    def __init__(self):
        if NETWORKX_AVAILABLE:
            self.knowledge_graph = nx.Graph()
        else:
            self.knowledge_graph = nx.Graph()  # Uses our simple fallback
        
        self.entity_patterns = {
            'companies': r'(?i)\b(?:China CITIC Bank|CITIC Bank|AXA|Cigna|Ipsos|City University|Education University|CITIC)\b',
            'skills': r'(?i)\b(?:Python|PySpark|SQL|Machine Learning|Deep Learning|NLP|Tableau|Power BI|Azure|AWS|AutoML|MLOps|Data Science|Analytics|AI|Artificial Intelligence)\b',
            'positions': r'(?i)\b(?:AVP|Assistant.*Manager|Data Science Analyst|Research Executive|Manager|Director|Analyst)\b',
            'years': r'\b(?:20\d{2})\b',
            'industries': r'(?i)\b(?:Insurance|Banking|Healthcare|Finance|Market Research|Medicare|ESG|Green|Environmental)\b',
            'projects': r'(?i)\b(?:Go-Green|Go Green|ESG|Dashboard|AutoML|Pipeline|Migration|Campaign)\b',
            'locations': r'(?i)\b(?:Hong Kong|China|Asia)\b'
        }
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text"""
        entities = []
        for category, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(f"{category}:{match}")
        return entities
    
    def get_related_entities(self, query: str) -> List[str]:
        """Get entities related to query"""
        query_entities = self.extract_entities(query)
        related = set(query_entities)
        
        # Add entities found in the query to related set
        for entity in query_entities:
            if self.knowledge_graph.has_node(entity):
                neighbors = list(self.knowledge_graph.neighbors(entity))
                related.update(neighbors[:5])
        
        # Also search for partial matches in graph nodes
        query_lower = query.lower()
        for node in self.knowledge_graph.nodes():
            node_value = node.split(':')[-1].lower() if ':' in node else node.lower()
            for word in query_lower.split():
                if len(word) > 2 and word in node_value:
                    related.add(node)
                    neighbors = list(self.knowledge_graph.neighbors(node))
                    related.update(neighbors[:3])
                    break
        
        return list(related)

class QwenAPIClient:
    """Qwen API client for response generation with cost tracking"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.available = self._test_connection()
        self.cost_calculator = CostCalculator()
    
    def _test_connection(self) -> bool:
        """Test API connection"""
        if not self.api_key:
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "qwen-turbo",
                "input": {"prompt": "Hello"},
                "parameters": {"max_tokens": 10, "temperature": 0.1}
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, query: str, context: str, graph_entities: List[str] = None) -> tuple:
        """Generate response using Qwen API with cost tracking"""
        if not self.available:
            return "❌ Qwen API is not available. Please contact Rory at chengy823@gmail.com", None
        
        # Build comprehensive prompt
        system_prompt = """You are Rory Chen's professional AI assistant with access to his comprehensive career information.

CORE INFORMATION ABOUT RORY:
- Current Role: AVP of Data Science at China CITIC Bank International (Nov 2022 - Current)
- 8 years of professional experience (2017-2025)
- Career progression: Research Executive → Data Science Analyst → Assistant Data Science Manager → AVP
- Industries: Market Research, Healthcare (Medicare), Insurance, Banking
- Education: Master's in Quantitative Analysis (2016-2017), Master's in Public Policy (2014-2015), Bachelor's (2011-2014)
- Location: Hong Kong SAR
- Contact: chengy823@gmail.com

TECHNICAL EXPERTISE:
- Programming: Python, Pyspark, SQL (Advanced proficiency)
- Machine Learning: Deep Learning, NLP, Computer Vision, MLOps, AutoML, LangChain
- Data Processing and ETL
- Pega Customer Decision Hub
- Analytics: Predictive Modeling, Customer Analytics, Time Series Analysis
- Cloud Platforms: Azure, Google Cloud, Databricks, Cloudera CDSW
- Visualization: Tableau, Power BI, Dashboard Development

KEY ACHIEVEMENTS:
- Developed 20+ ML models achieving 1.5x business uplift vs control groups
- Created AutoML pipeline reducing coding effort by 80%
- Designed AI+BI framework when develop models in China Citic Bank Int'l
- Created ONE AXA Dashboard which consolidated all LoBs customer and product data into a single dasboard 
- Led cloud migration from on-premise to Azure infrastructure

Provide professional, accurate responses based on the context."""

        # Add GraphRAG context if available
        graph_context = ""
        if graph_entities:
            graph_context = f"\n\nGraphRAG Analysis - Related entities: {', '.join(graph_entities[:10])}"
        
        user_prompt = f"""Question: {query}

Context from cloud vector database:
{context}
{graph_context}

Please provide a comprehensive, professional response."""

        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "qwen-turbo",
                "input": {"prompt": combined_prompt},
                "parameters": {
                    "max_tokens": 8000,
                    "temperature": 0.7,
                    "top_p": 0.8
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=45)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['output']['text'].strip()
                
                if graph_entities:
                    ai_response += "\n\n*Response enhanced with GraphRAG technology and cloud vector database.*"
                
                # Calculate cost
                cost_info = self.cost_calculator.calculate_cost(combined_prompt, ai_response, "qwen-turbo")
                
                return ai_response, cost_info
            else:
                return "❌ Qwen API failed. Please contact Rory at chengy823@gmail.com", None
                
        except Exception as e:
            return "❌ Qwen API failed. Please contact Rory at chengy823@gmail.com", None

# Streamlit App Configuration
st.set_page_config(
    page_title="Rory's AI Assistant - Cloud Compatible",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize cost calculator
if "cost_calculator" not in st.session_state:
    st.session_state.cost_calculator = CostCalculator()

# Custom CSS for enhanced design
st.markdown("""
<style>
    .main .block-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1400px;
        padding-top: 1rem;
    }
    
    .header-container {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .status-container {
        background: #f8f9fa;
        color: #333;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .cost-container {
        background: #e8f5e8;
        color: #333;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-container {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    .chunk-info {
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-top: 0.5rem;
        border: 1px solid #bbdefb;
    }
    
    .cost-info {
        background: #e8f5e8;
        color: #2e7d32;
        padding: 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-top: 0.5rem;
        border: 1px solid #c8e6c9;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    max_chunks = st.slider(
        "Max Chunks Retrieved:",
        min_value=1,
        max_value=10,
        value=5,
        help="Maximum number of chunks to retrieve per query"
    )
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    st.markdown("**Library Status:**")
    st.write(f"NetworkX: {'✅' if NETWORKX_AVAILABLE else '❌ (using fallback)'}")
    st.write(f"ChromaDB: {'✅' if CHROMADB_AVAILABLE else '❌'}")
    st.write(f"PyPDF2: {'✅' if PYPDF2_AVAILABLE else '❌'}")
    st.write(f"python-docx: {'✅' if DOCX_AVAILABLE else '❌'}")
    
    st.markdown("---")
    st.markdown("### 🔄 Actions")
    
    if st.button("🔄 Refresh System"):
        st.cache_resource.clear()
        st.rerun()
    
    if st.button("💰 Reset Cost Tracking"):
        st.session_state.cost_calculator.reset_costs()
        st.success("Cost tracking reset!")
        st.rerun()

# Main header
st.markdown("""
<div class="header-container">
    <h1>🤖 Rory's AI Assistant - Streamlit Cloud Compatible</h1>
    <p>Cloud ChromaDB • GraphRAG • Cost Tracking • Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Initialize system
@st.cache_resource
def initialize_system():
    """Initialize the complete system with cloud ChromaDB"""
    try:
        # Get API key from environment or use default
        api_key = os.getenv("QWEN_API_KEY", "sk-015ea57c8b254c4181d30b2de4259d8b")
        os.environ["QWEN_API_KEY"] = api_key
        
        # Initialize components
        chroma_db = CloudChromaDBManager()
        graph_rag = SimpleGraphRAGProcessor()
        qwen_client = QwenAPIClient()
        
        # Get document count from cloud ChromaDB
        doc_count = chroma_db.get_collection_count()
        
        return chroma_db, graph_rag, qwen_client, doc_count
    
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return None, None, None, 0

# Initialize system
with st.spinner("🚀 Initializing AI System..."):
    chroma_db, graph_rag, qwen_client, doc_count = initialize_system()

if not chroma_db:
    st.error("❌ System initialization failed. Please check the configuration.")
    st.stop()

# System status
api_status = "✅ Connected" if qwen_client.available else "❌ Not Available"
storage_type = getattr(chroma_db, 'storage_type', 'Unknown')

# Cost summary
cost_summary = st.session_state.cost_calculator.get_cost_summary()

# Status and cost display
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div class="status-container">
        <strong>🧠 System Status:</strong> {doc_count} knowledge chunks in {storage_type} | 
        GraphRAG: {'✅ NetworkX' if NETWORKX_AVAILABLE else '✅ Fallback'} | Qwen API: {api_status}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="cost-container">
        <strong>💰 Session Cost:</strong> ${cost_summary['total_cost']:.4f} | 
        <strong>Tokens:</strong> {cost_summary['total_tokens']:,} | 
        <strong>Queries:</strong> {cost_summary['conversation_count']}
    </div>
    """, unsafe_allow_html=True)

# Main dashboard with tabs
tab1, tab2 = st.tabs(["💬 Chat Assistant", "📊 Cost Analytics"])

with tab1:
    # Chat interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        ### 👨‍💼 Rory Chen
        **AVP of Data Science**  
        China CITIC Bank International
        
        **Experience:** 8 years (2017-2025)
        
        **Industries:**
        - Banking (China CITIC Bank Int'l)
        - Insurance (AXA)
        - Healthcare (Cigna)
        - Market Research (Ipsos)
                                      
        **Key Skills:**
        - Python, Pyspark, SQL
        - Machine Learning & AI, LangChain
        - Cloud Platforms (Azure, Google Cloud)
        - Pega Customer Decision Hub
        - Data Visualization
        
        **Email:** chengy823@gmail.com
        **Phone:** 68746551
        """)
    
    with col2:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown("### 💬 Chat with AI Assistant")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": "👋 Hello! I'm Rory's AI assistant with cost tracking and analytics. Ask me anything about Rory's professional background!"
            })
        
        # Quick sample questions
        st.markdown("#### 🎯 Quick Sample Questions:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏢 Current Role", key="q1"):
                sample_question = "Tell me about Rory's current role at China CITIC Bank International"
                st.session_state.messages.append({"role": "user", "content": sample_question})
                st.rerun()
        
        with col2:
            if st.button("🚀 Technical Skills", key="q2"):
                sample_question = "What are Rory's key technical skills and achievements?"
                st.session_state.messages.append({"role": "user", "content": sample_question})
                st.rerun()
        
        with col3:
            if st.button("📈 Career Path", key="q3"):
                sample_question = "Describe Rory's career progression and experience"
                st.session_state.messages.append({"role": "user", "content": sample_question})
                st.rerun()
        
        st.markdown("---")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about Rory's experience, skills, or achievements..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Processing..."):
                    try:
                        # Query ChromaDB
                        chroma_results = chroma_db.query_documents(prompt, n_results=max_chunks)
                        
                        # Get GraphRAG entities
                        graph_entities = graph_rag.get_related_entities(prompt)
                        
                        # Build context
                        context = ""
                        chunks_used = 0
                        if chroma_results['documents'] and chroma_results['documents'][0]:
                            context = "\n\n".join(chroma_results['documents'][0])
                            chunks_used = len(chroma_results['documents'][0])
                        
                        # Generate response with cost tracking
                        response, cost_info = qwen_client.generate_response(prompt, context, graph_entities)
                        
                        # Display response
                        st.markdown(response)
                        
                        # Display chunk information
                        if chunks_used > 0:
                            sources = []
                            if chroma_results.get('metadatas') and chroma_results['metadatas'][0]:
                                sources = [meta.get('source', 'Unknown') for meta in chroma_results['metadatas'][0][:3]]
                            
                            st.markdown(f"""
                            <div class="chunk-info">
                                📊 <strong>Query Analysis:</strong> Used {chunks_used} knowledge chunks | 
                                GraphRAG entities: {len(graph_entities)} | 
                                Sources: {', '.join(set(sources)) if sources else 'Default knowledge'}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display cost information
                        if cost_info:
                            st.markdown(f"""
                            <div class="cost-info">
                                💰 <strong>Cost:</strong> ${cost_info['total_cost']:.4f} | 
                                <strong>Input:</strong> {cost_info['input_tokens']} tokens | 
                                <strong>Output:</strong> {cost_info['output_tokens']} tokens
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Log conversation to database
                        chroma_db.log_conversation(prompt, response, chunks_used)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # Cost Analytics Dashboard
    st.markdown("### 📊 Cost Analytics Dashboard")
    
    cost_summary = st.session_state.cost_calculator.get_cost_summary()
    
    # Cost metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>${cost_summary['total_cost']:.4f}</h3>
            <p>Total Cost</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{cost_summary['total_tokens']:,}</h3>
            <p>Total Tokens</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{cost_summary['conversation_count']}</h3>
            <p>Conversations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>${cost_summary['avg_cost_per_conversation']:.4f}</h3>
            <p>Avg Cost/Query</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cost visualization
    if st.session_state.conversation_costs:
        st.markdown("### 📈 Cost Trends")
        
        # Create DataFrame for visualization
        df_costs = pd.DataFrame(st.session_state.conversation_costs)
        df_costs['cumulative_cost'] = df_costs['total_cost'].cumsum()
        
        # Cost over time chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Cost Over Time', 'Cost per Conversation', 
                          'Token Usage Distribution', 'Input vs Output Tokens'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Cumulative cost
        fig.add_trace(
            go.Scatter(x=df_costs.index, y=df_costs['cumulative_cost'],
                      mode='lines+markers', name='Cumulative Cost',
                      line=dict(color='#28a745')),
            row=1, col=1
        )
        
        # Cost per conversation
        fig.add_trace(
            go.Bar(x=df_costs.index, y=df_costs['total_cost'],
                   name='Cost per Query', marker_color='#3498db'),
            row=1, col=2
        )
        
        # Token distribution
        fig.add_trace(
            go.Histogram(x=df_costs['input_tokens'] + df_costs['output_tokens'],
                        name='Total Tokens', marker_color='#f39c12'),
            row=2, col=1
        )
        
        # Input vs Output tokens
        fig.add_trace(
            go.Scatter(x=df_costs['input_tokens'], y=df_costs['output_tokens'],
                      mode='markers', name='Input vs Output',
                      marker=dict(color='#e74c3c', size=8)),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed cost table
        st.markdown("### 📋 Detailed Cost Breakdown")
        
        # Format the data for display
        display_df = df_costs.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
        display_df['total_cost'] = display_df['total_cost'].apply(lambda x: f"${x:.4f}")
        display_df['input_cost'] = display_df['input_cost'].apply(lambda x: f"${x:.4f}")
        display_df['output_cost'] = display_df['output_cost'].apply(lambda x: f"${x:.4f}")
        
        st.dataframe(
            display_df[['timestamp', 'input_tokens', 'output_tokens', 
                       'input_cost', 'output_cost', 'total_cost', 'model']],
            use_container_width=True
        )
    else:
        st.info("💡 Start chatting to see cost analytics!")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>🤖 <strong>Streamlit Cloud Compatible AI Assistant</strong></p>
    <p>GraphRAG • Cost Tracking • Analytics Dashboard • {storage_type} Storage</p>
    <p>📧 Contact: chengy823@gmail.com</p>
</div>
""", unsafe_allow_html=True)
