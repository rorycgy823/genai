"""
Enhanced Cloud ChromaDB Compatible AI Assistant
==============================================

A Streamlit compatible version with enhanced features:
- Cost calculation and token tracking
- ChromaDB visualization dashboard
- Secure API configuration
- Multi-page dashboard interface

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
import networkx as nx
from collections import defaultdict
import re
import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import ChromaDB with fallback
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except Exception as e:
    CHROMADB_AVAILABLE = False
    CHROMADB_ERROR = str(e)

# Try to import document processing libraries with fallbacks
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except:
    PYPDF2_AVAILABLE = False

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except:
    PYMUPDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except:
    DOCX_AVAILABLE = False

# OCR libraries for enhanced PDF processing
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except:
    EASYOCR_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class CostCalculator:
    """Calculate API costs and token usage"""
    
    def __init__(self):
        # Qwen API pricing (example rates - adjust based on actual pricing)
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
        # Rough estimation: 1 token ≈ 4 characters for English
        # For Chinese/mixed content, might be different
        return len(text) // 4
    
    def calculate_cost(self, input_text: str, output_text: str, model: str = "qwen-turbo") -> Dict[str, Any]:
        """Calculate cost for a single API call with accurate token tracking"""
        # More accurate token estimation for Chinese/English mixed content
        input_tokens = max(1, len(input_text.split()) + len(input_text) // 6)  # Account for Chinese characters
        output_tokens = max(1, len(output_text.split()) + len(output_text) // 6)
        
        if model not in self.pricing:
            model = "qwen-turbo"  # fallback
        
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
        
        # Sort by relevance (simple scoring)
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
                raise Exception(f"Service health check failed: ChromaDB {chroma_response.status_code}, Embedding {embedding_response.status_code}")
                
        except Exception as e:
            self.storage = InMemoryStorage()
            self.storage_type = "In-Memory (Fallback) ⚠️"
            self.connection_status = f"Cloud connection failed: {str(e)}"
            self.use_fallback = True
    
    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to storage via HTTP API"""
        if not self.use_fallback:
            try:
                # Use HTTP API to add documents to knowledge_base collection
                data = {
                    "documents": documents,
                    "metadatas": metadatas,
                    "ids": ids
                }
                
                response = requests.post(
                    f"{self.chroma_url}/collections/knowledge_base/add",
                    json=data,
                    timeout=30
                )
                
                return response.status_code == 200
                    
            except Exception as e:
                return False
        else:
            return self.storage.add_documents(documents, metadatas, ids)
    
    def query_documents(self, query: str, n_results: int = 5) -> Dict:
        """Query documents from storage via HTTP API"""
        if not self.use_fallback:
            try:
                # Use HTTP API to query knowledge_base collection
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
                response = requests.get(
                    f"{self.chroma_url}/collections", 
                    timeout=10
                )
                
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
    
    def get_all_documents(self, limit: int = 1000) -> Dict[str, Any]:
        """Get all documents from knowledge base for analysis"""
        if not self.use_fallback:
            try:
                # Use multiple broad queries to get all documents
                queries = ["data science", "experience", "skills", "projects", "work", "education", "CV", "resume"]
                all_documents = []
                all_metadatas = []
                all_distances = []
                seen_ids = set()
                
                for query in queries:
                    data = {
                        "query_texts": [query],
                        "n_results": min(50, limit),
                        "include": ["documents", "metadatas", "distances"]
                    }
                    
                    response = requests.post(
                        f"{self.chroma_url}/collections/knowledge_base/query",
                        json=data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        docs = result.get("documents", [[]])[0] if result.get("documents") else []
                        metas = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
                        dists = result.get("distances", [[]])[0] if result.get("distances") else []
                        
                        # Avoid duplicates by checking document content
                        for i, doc in enumerate(docs):
                            doc_id = hash(doc[:100])  # Use first 100 chars as ID
                            if doc_id not in seen_ids:
                                seen_ids.add(doc_id)
                                all_documents.append(doc)
                                all_metadatas.append(metas[i] if i < len(metas) else {})
                                all_distances.append(dists[i] if i < len(dists) else 0.0)
                
                return {
                    "documents": [all_documents[:limit]],
                    "metadatas": [all_metadatas[:limit]],
                    "distances": [all_distances[:limit]]
                }
                    
            except Exception as e:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        else:
            return {
                "documents": [self.storage.documents],
                "metadatas": [self.storage.metadatas],
                "distances": [[0.0] * len(self.storage.documents)]
            }
    
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
                # Don't show error for conversation logging to avoid cluttering UI
                return False
        else:
            return self.storage.log_conversation(user_query, ai_response, chunks_used)
    
    def get_conversation_count(self) -> int:
        """Get total number of conversations logged via HTTP API"""
        if not self.use_fallback:
            try:
                response = requests.get(
                    f"{self.chroma_url}/collections",
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    conversations_info = result.get("conversations", {})
                    return conversations_info.get("count", 0)
                else:
                    return 0
                    
            except Exception as e:
                return 0
        else:
            return self.storage.get_conversation_count()

class GraphRAGProcessor:
    """Simple GraphRAG implementation for enhanced retrieval"""
    
    def __init__(self):
        self.knowledge_graph = nx.Graph()
        self.entity_patterns = {
            'companies': r'(?i)\b(?:China CITIC Bank|CITIC Bank|AXA|Cigna|Ipsos|City University|Education University|CITIC)\b',
            'skills': r'(?i)\b(?:Python|PySpark|SQL|Machine Learning|Deep Learning|NLP|Tableau|Power BI|Azure|AWS|AutoML|MLOps|Data Science|Analytics|AI|Artificial Intelligence|CRM)\b',
            'positions': r'(?i)\b(?:AVP|Assistant.*Manager|Data Science Analyst|Research Executive|Manager|Director|Analyst)\b',
            'years': r'\b(?:20\d{2})\b',
            'industries': r'(?i)\b(?:Insurance|Banking|Healthcare|Finance|Market Research|Medicare|ESG|Green|Environmental)\b',
            'projects': r'(?i)\b(?:Go-Green|Go Green|ESG|Dashboard|AutoML|Pipeline|Migration|Campaign|Pega|CRM)\b',
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
    
    def build_graph_from_documents(self, documents: List[str], metadatas: List[Dict]):
        """Build knowledge graph from documents"""
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            entities = self.extract_entities(doc)
            doc_id = f"doc_{i}"
            
            # Add entities as nodes
            for entity in entities:
                if not self.knowledge_graph.has_node(entity):
                    self.knowledge_graph.add_node(entity, documents=[doc_id])
                else:
                    self.knowledge_graph.nodes[entity]['documents'].append(doc_id)
            
            # Connect related entities
            for j, entity1 in enumerate(entities):
                for entity2 in entities[j+1:]:
                    if self.knowledge_graph.has_edge(entity1, entity2):
                        self.knowledge_graph[entity1][entity2]['weight'] += 1
                    else:
                        self.knowledge_graph.add_edge(entity1, entity2, weight=1)
    
    def get_related_entities(self, query: str) -> List[str]:
        """Get entities related to query"""
        # Extract entities from the query
        query_entities = self.extract_entities(query)
        related = set(query_entities)
        
        # Add entities found in the query to related set
        for entity in query_entities:
            if entity in self.knowledge_graph:
                neighbors = list(self.knowledge_graph.neighbors(entity))
                related.update(neighbors[:5])  # Limit to top 5 neighbors
        
        # Also search for partial matches in graph nodes
        query_lower = query.lower()
        for node in self.knowledge_graph.nodes():
            node_value = node.split(':')[-1].lower() if ':' in node else node.lower()
            # Check if any word in query matches the entity
            for word in query_lower.split():
                if len(word) > 2 and word in node_value:
                    related.add(node)
                    # Add neighbors of matched entities
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
        system_prompt = """You are Rory Chen's professional AI interview assistant with access to his comprehensive career information through GraphRAG technology and cloud-hosted vector database. As the interview assistant, your role is to help Rory to win a job from the user enquires.
You will provide accurate, professional responses based on the following context and demonstrate reasoning, and graph analysis when relevant. Please do not make up information or provide speculative answers.
If the users asking about Rory's GenAI exp or advanced technologies, you may explain that current company Rory working in does not have cloud or genAI capabilities, but he has extensive experience in these areas through self learning and practices. Recently he is exploring to use the genAI technologies to improve UAT efforts during their use case developments. This application is also a sample to demonstrate his experience to leverage genAI technogies to certain extent.
Rory has some people management experience since his project management experience in Ipsos though at that moment he was not a people manager. After moving to China CITIC, he has been leading a taskforce with 3-4 people for machine learning and advanced analytics. 
Rory also has quite a lot of project management experience in his career. In China Citic, the bank fully engaged with agile scrum approach so Rory has around 30-40% time spending as a business analyst to support the team agile scrum management.
Rory is a person with strong curiosity and resilience, always eager to learn new technologies and apply them in practice. He has been actively exploring and self-learning about genAI. 
As a sociologist in the life, Rory has a strong interest in understanding human behavior and social dynamics, which he applies into his work and analytics. He likes to explore the future of human society and how techonology can shape it.
For some funny questions such as do you know how old is Rory or if Rory married etc, please respond with some humourous answers like "Rory is forever 18 years old in his heart" or "Rory is married to his wife but he is opening for new relationships with AI technologies", or "贫穷有时会限制我的想象" etc.
All the description on the top can be used to understand Rory's background and experience and personality. 
If asking why Rory left AXA to China Citic, please explain that Rory left AXA to China Citic because he wanted to explore new challenges and opportunities in the banking industry, and to apply his data science and analytics skills in a different sector. Also, the covid may also be a reason that trigger some family considerations so that have to make a decision to leave AXA and join China Citic Bank International.
Rory has best team, management, colleagues, friends in AXA and Ipsos, though he is also grateful for the opportunties from other companies such as China Citic and Cigna.
The AutoML package and AI+BI framework was developed in China Citic Bank not AXA, so please do not mention it in AXA context.
Does not include any rude or inappropriate content and do not make up any information

CORE INFORMATION ABOUT RORY:
- Current Role: AVP of Data Science at China CITIC Bank International (Nov 2022 - Current)
- 8 years of professional experience (2017-2025)
- Career progression: 2017, Research Executive (Ipsos, and then promote to Senior Research Executive in 2018) → Data Science Analyst (cigna, in 2018 to 2019) → Senior Data Analyst (AXA, 2019) → Assistant Data Science Manager (AXA., in 2022) → Senior Manager(Citic, in 2022) → AVP (Citic, in 2024)
- Industries: Market Research, Healthcare (Medicare), Insurance, Banking
- Education: Master's in Quantitative Analysis (2016-2017), Master's in Public Policy (2014-2015), Bachelor's (2011-2014)
- Location: Hong Kong SAR
- Contact: chengy823@gmail.com

TECHNICAL EXPERTISE:
- Programming: Python, Pyspark, SQL, SAS
- Machine Learning: LightGBM, SKLearn, Deep Learning, NLP, Computer Vision, MLOps, AutoML, LangChain, GraphRAG, LLMs, Context Prompting
- Data Processing and ETL
- Pega Customer Decision Hub
- Analytics: Predictive Modeling, Customer Analytics, Time Series Analysis
- Cloud Platforms: Azure, Google Cloud, Databricks, Cloudera CDSW, Aliyun,
- Visualization: Tableau, Power BI, Dashboard Development

KEY ACHIEVEMENTS:
China Citic Bank International:
- Developed a digital engagement platform on Pega CDH platform and monetized the CRM platform for the bank to enable next best action framework and omni channel marketing including SMS, Email, push, app, etc.
- Used the Pega CDH platform to develop different types of marketing such as event trigger, batch campaign, sas campaign, real-time decisioning, etc.
- Led the development of 10+ ML models for different bank products to enable NBA framework in Pega CDH to achieving 1.5x business uplift vs control groups in China Citic Bank, integrated with AI+BI framework and test & learn framework
- Created AutoML pipeline reducing coding effort by 80%, even for non-technical users to quick develop ML models in China Citic Bank
- The platform enabled over 100 use cases including batch and real time event trigger with weekly blasting over 200K+ messages to customers, acheived 20M+ HKD revenue uplift from 2023 to 2024.
- Helped the team to win 2022 Best CNCBI, CAHK Star Award 2023 and The Asian Banker Hong Kong Excellence in Retail Finance and Technology Innovation Awards 2024

AXA:
- Business partner for GI side to support all sorts of data analytics and solutions. Especially the go-green project which won few awards in 2021 and 2022 including HKB Technology Excellence Awards 2022 – Big Data – Insurance, and ONE AXA Award – 2021 - second place finalist.
- Supported the development of Customer 360 and migration of on-prem DB to Azure cloud.
- Created a prospect customer database on azure cloud to consolidate prospect customers from all channels such as salesforce, one circle club, etc.
- Created ONE AXA Dashboard which consolidated all LoBs customer and product data into a single dasboard, and supported migration from Tableau to power bi.
- Supported cloud migration from on-premise to Azure infrastructure
- Supported campaign customer list generation and campaign performance analysis
- Developed a predictive disease model to predict the probability of diease for each customer based on their medical history and demographics

Cigna:
- Developed predictive models to identify high-risk patients for emergency room visits to reduce the costs
- Created dashboard to visualize patient data and analytics
- Developed customer segmentation model to identify high value customers

Ipsos:
- Led market research projects for clients in insurance and banking sectors
- Developed customer satisfaction and NPS score analytics and tracking insights
- Conducted marketing research and project management up to regional level, managed up to 10 countries fieldwork teams simultaneously
- Developed and implemented data collection methodologies and tools, designed surveys and questionnaires
- Supported the tracking dashboard development (EFM - Enterprise Feedback Managment) for clients to track their NPS and customer satisfaction scores

CAREER PROGRESSION REASONING:
- Ipsos (Market Research) → Cigna (Healthcare Analytics) → AXA (Insurance) → China CITIC Bank Int'l (Banking)
- Client relationships at Ipsos likely included insurance/banking companies, facilitating career transitions, demonstrated recognization from clients, 
- Progressive technical leadership from individual contributor to AVP level
- Cross-industry expertise demonstrates adaptability and broad business acumen

Provide professional, accurate responses based on the context and demonstrate reasoning."""

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
        
        # Single attempt with longer timeout - Qwen API only
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
    page_title="Rory's AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize cost calculator
if "cost_calculator" not in st.session_state:
    st.session_state.cost_calculator = CostCalculator()

# Custom CSS for maximum text visibility in both light and dark themes with completely hidden sidebar
st.markdown("""
<style>
    /* COMPLETE SIDEBAR REMOVAL - All possible selectors */
    .css-1d391kg, .css-1lcbmhc, .css-1y4p8pa, .css-17eq0hr, .css-1544g2n,
    .css-1rs6os, .css-1vencpc, .css-k1vhr4, .css-1cypcdb, .css-1avcm0n,
    .css-18e3th9, .css-1d391kg, .css-1lcbmhc, .css-1y4p8pa, .css-17eq0hr,
    .css-1544g2n, .css-1rs6os, .css-1vencpc, .css-k1vhr4, .css-1cypcdb,
    .css-1avcm0n, .css-18e3th9, section[data-testid="stSidebar"],
    .stSidebar, [data-testid="stSidebar"], .sidebar, .css-sidebar {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Hide sidebar toggle button completely */
    button[kind="header"], .css-1rs6os, .css-1vencpc, 
    [data-testid="collapsedControl"], .css-k1vhr4 {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* LIGHT THEME - Default styling with maximum contrast */
    .main .block-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1400px;
        padding-top: 1rem;
        background-color: #ffffff !important;
        margin-left: 0 !important;
        padding-left: 1rem !important;
        width: 100% !important;
        color: #000000 !important;
    }
    
    /* Root background for consistency */
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Header with maximum contrast */
    .header-container {
        background: linear-gradient(90deg, #1a237e 0%, #283593 100%) !important;
        color: #ffffff !important;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Status containers with maximum contrast */
    .status-container {
        background: #e1f5fe !important;
        color: #01579b !important;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0277bd !important;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        font-weight: 700 !important;
    }
    
    .cost-container {
        background: #e8f5e8 !important;
        color: #1b5e20 !important;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e7d32 !important;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        font-weight: 700 !important;
    }
    
    /* Chat container with maximum contrast */
    .chat-container {
        background: #ffffff !important;
        color: #000000 !important;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        border: 3px solid #bdbdbd !important;
    }
    
    /* Dashboard containers */
    .dashboard-container {
        background: #ffffff !important;
        color: #000000 !important;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        border: 2px solid #bdbdbd !important;
    }
    
    /* Metric cards with maximum contrast */
    .metric-card {
        background: #f5f5f5 !important;
        color: #000000 !important;
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        border: 2px solid #9e9e9e !important;
        font-weight: 700 !important;
    }
    
    /* Information boxes with maximum contrast */
    .chunk-info {
        background: #e1f5fe !important;
        color: #01579b !important;
        padding: 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        border: 2px solid #4fc3f7 !important;
        font-weight: 700 !important;
    }
    
    .cost-info {
        background: #e8f5e8 !important;
        color: #1b5e20 !important;
        padding: 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        border: 2px solid #66bb6a !important;
        font-weight: 700 !important;
    }
    
    /* Text inputs with maximum contrast */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 3px solid #757575 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1976d2 !important;
        box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.3) !important;
        color: #000000 !important;
    }
    
    /* Chat messages with maximum contrast */
    .stChatMessage[data-testid="user-message"] {
        background-color: #e3f2fd !important;
        color: #01579b !important;
        border: 3px solid #42a5f5 !important;
        font-weight: 600 !important;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 3px solid #757575 !important;
        font-weight: 600 !important;
    }
    
    /* General chat message styling */
    .stChatMessage {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 3px solid #757575 !important;
        border-radius: 8px !important;
        margin-bottom: 0.5rem !important;
        font-weight: 600 !important;
    }
    
    /* Chat message content with maximum contrast */
    .stChatMessage .stMarkdown {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    .stChatMessage[data-testid="user-message"] .stMarkdown {
        color: #01579b !important;
        font-weight: 600 !important;
    }
    
    /* Tab styling with maximum contrast */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f5f5f5 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: 3px solid #9e9e9e !important;
        font-weight: 700 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1976d2 !important;
        color: #ffffff !important;
        border: 3px solid #1976d2 !important;
        font-weight: 700 !important;
    }
    
    /* Button styling with maximum contrast */
    .stButton > button {
        background-color: #f5f5f5 !important;
        color: #000000 !important;
        border: 3px solid #757575 !important;
        border-radius: 4px;
        font-weight: 700 !important;
    }
    
    .stButton > button:hover {
        background-color: #e0e0e0 !important;
        border-color: #424242 !important;
        color: #000000 !important;
    }
    
    /* Chat input with maximum contrast */
    .stChatInput > div > div > div > div {
        background-color: #f8f9fa !important;
        color: #000000 !important;
        border: 3px solid #757575 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Ensure ALL text has maximum contrast - Light theme */
    .stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6, 
    .stSelectbox label, .stSlider label, .stCheckbox label,
    .stRadio label, .stTextArea label, .stNumberInput label,
    .stDateInput label, .stTimeInput label, .stFileUploader label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Special styling for headers with maximum contrast */
    h1, h2, h3 {
        color: #1a237e !important;
        font-weight: 800 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Streamlit specific elements */
    .stMetric-label {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    .stMetric-value {
        color: #000000 !important;
        font-weight: 800 !important;
    }
    
    /* DARK MODE - Light colored boxes with white text */
    @media (prefers-color-scheme: dark) {
        .main .block-container, .stApp {
            background-color: #121212 !important;
            color: #ffffff !important;
        }
        
        .header-container {
            background: linear-gradient(90deg, #64b5f6 0%, #90caf9 100%) !important;
            color: #000000 !important;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.3);
            font-weight: 800 !important;
        }
        
        .status-container {
            background: #81c784 !important;
            color: #000000 !important;
            border-left-color: #4caf50 !important;
            font-weight: 800 !important;
        }
        
        .cost-container {
            background: #ffb74d !important;
            color: #000000 !important;
            border-left-color: #ff9800 !important;
            font-weight: 800 !important;
        }
        
        .chat-container, .dashboard-container {
            background: #e0e0e0 !important;
            color: #000000 !important;
            border-color: #bdbdbd !important;
        }
        
        .metric-card {
            background: #f5f5f5 !important;
            color: #000000 !important;
            border-color: #9e9e9e !important;
            font-weight: 800 !important;
        }
        
        .chunk-info {
            background: #81c784 !important;
            color: #000000 !important;
            border-color: #4caf50 !important;
            font-weight: 800 !important;
        }
        
        .cost-info {
            background: #ffb74d !important;
            color: #000000 !important;
            border-color: #ff9800 !important;
            font-weight: 800 !important;
        }
        
        .stChatMessage[data-testid="assistant-message"] {
            background-color: #f0f0f0 !important;
            color: #000000 !important;
            border-color: #9e9e9e !important;
        }
        
        .stChatMessage[data-testid="user-message"] {
            background-color: #bbdefb !important;
            color: #000000 !important;
            border-color: #2196f3 !important;
        }
        
        .stChatMessage .stMarkdown {
            color: #000000 !important;
        }
        
        .stChatMessage[data-testid="user-message"] .stMarkdown {
            color: #000000 !important;
        }
        
        .stTextInput > div > div > input, .stChatInput > div > div > div > div {
            background-color: #f8f9fa !important;
            color: #000000 !important;
            border-color: #9e9e9e !important;
        }
        
        .stTextInput > div > div > input:focus, .stChatInput > div > div > div > div:focus {
            border-color: #2196f3 !important;
            color: #000000 !important;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background-color: #e0e0e0 !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f5f5f5 !important;
            color: #000000 !important;
            border-color: #9e9e9e !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #2196f3 !important;
            color: #ffffff !important;
        }
        
        .stButton > button {
            background-color: #e0e0e0 !important;
            color: #000000 !important;
            border-color: #9e9e9e !important;
        }
        
        .stButton > button:hover {
            background-color: #d0d0d0 !important;
            border-color: #757575 !important;
            color: #000000 !important;
        }
        
        /* Ensure ALL text has maximum contrast - Dark theme */
        .stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6,
        .stSelectbox label, .stSlider label, .stCheckbox label,
        .stRadio label, .stTextArea label, .stNumberInput label,
        .stDateInput label, .stTimeInput label, .stFileUploader label {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 800 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        }
        
        .stMetric-label, .stMetric-value {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
    }
    
    /* Light theme - Black text on light colored boxes */
    @media (prefers-color-scheme: light), (prefers-color-scheme: no-preference) {
        .main .block-container, .stApp {
            background-color: #fafafa !important;
        }
        
        .chat-container, .dashboard-container {
            background: #f8f9fa !important;
            color: #000000 !important;
        }
        
        .stChatMessage[data-testid="assistant-message"] {
            background-color: #f5f5f5 !important;
            color: #000000 !important;
        }
        
        .stTextInput > div > div > input {
            background-color: #f8f9fa !important;
            color: #000000 !important;
        }
        
        /* Ensure all text is black in light mode */
        .stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6,
        .stSelectbox label, .stSlider label, .stCheckbox label,
        .stRadio label, .stTextArea label, .stNumberInput label,
        .stDateInput label, .stTimeInput label, .stFileUploader label {
            color: #000000 !important;
        }
        
        /* Special containers keep their specific colors */
        .header-container *, .stTabs [aria-selected="true"] * {
            color: #ffffff !important;
        }
        
        .status-container * {
            color: #01579b !important;
        }
        
        .cost-container * {
            color: #1b5e20 !important;
        }
        
        .chunk-info * {
            color: #01579b !important;
        }
        
        .cost-info * {
            color: #1b5e20 !important;
        }
        
        .stChatMessage[data-testid="user-message"] * {
            color: #01579b !important;
        }
    }
    
    /* Dark theme - White text on light colored boxes */
    @media (prefers-color-scheme: dark) {
        /* Main text should be white */
        .stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6,
        .stSelectbox label, .stSlider label, .stCheckbox label,
        .stRadio label, .stTextArea label, .stNumberInput label,
        .stDateInput label, .stTimeInput label, .stFileUploader label {
            color: #ffffff !important;
        }
        
        /* But text inside light colored boxes should be black */
        .header-container *, .status-container *, .cost-container *, 
        .chunk-info *, .cost-info *, .chat-container *, .dashboard-container *,
        .metric-card *, .stChatMessage *, .stTextInput *, .stChatInput *,
        .stTabs [data-baseweb="tab"] *, .stButton > button * {
            color: #000000 !important;
        }
        
        /* Selected tab and header keep white text */
        .stTabs [aria-selected="true"] * {
            color: #ffffff !important;
        }
    }
    
    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        
        .header-container {
            padding: 1rem;
            font-size: 0.9rem;
        }
        
        .chat-container {
            padding: 0.5rem;
        }
        
        .stChatMessage {
            margin-bottom: 0.25rem !important;
            font-size: 0.9rem !important;
        }
        
        .metric-card {
            padding: 0.75rem;
            font-size: 0.85rem;
        }
    }
    
    /* Tablet responsive adjustments */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        
        .stChatMessage {
            font-size: 0.95rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for settings (removed API key for security)
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Text Chunking Settings
    st.markdown("**Text Chunking:**")
    chunk_size = st.slider(
        "Chunk Size (characters):",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="Maximum size of each text chunk"
    )
    
    overlap = st.slider(
        "Chunk Overlap (characters):",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="Overlap between consecutive chunks"
    )
    
    min_chunk_size = st.slider(
        "Minimum Chunk Size:",
        min_value=50,
        max_value=500,
        value=100,
        step=25,
        help="Minimum size for a chunk to be stored"
    )
    
    max_chunks = st.slider(
        "Max Chunks Retrieved:",
        min_value=1,
        max_value=10,
        value=5,
        help="Maximum number of chunks to retrieve per query"
    )
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    # Show compatibility status
    st.markdown("**Library Status:**")
    st.write(f"ChromaDB: {'✅' if CHROMADB_AVAILABLE else '❌'}")
    st.write(f"PyMuPDF: {'✅' if PYMUPDF_AVAILABLE else '❌'}")
    st.write(f"PyPDF2: {'✅' if PYPDF2_AVAILABLE else '❌'}")
    st.write(f"python-docx: {'✅' if DOCX_AVAILABLE else '❌'}")
    
    st.markdown("**Current Config:**")
    st.write(f"Chunk Size: {chunk_size} chars")
    st.write(f"Overlap: {overlap} chars")
    st.write(f"Min Size: {min_chunk_size} chars")
    st.write(f"Max Retrieved: {max_chunks} chunks")
    
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
    <h1>🤖 Rory's AI Assistant</h1>
    <p>Cloud ChromaDB • GraphRAG • Cost Tracking • Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Initialize system with secure API key handling
@st.cache_resource
def initialize_system(chunk_size=1000, overlap=200, min_chunk_size=100):
    """Initialize the complete system with cloud ChromaDB - ONLY use cloud data"""
    try:
        # Get API key from environment or use default (for demo purposes)
        api_key = os.getenv("QWEN_API_KEY", "sk-015ea57c8b254c4181d30b2de4259d8b")
        os.environ["QWEN_API_KEY"] = api_key
        
        # Initialize components
        chroma_db = CloudChromaDBManager()
        graph_rag = GraphRAGProcessor()
        qwen_client = QwenAPIClient()
        
        # Get document count from cloud ChromaDB
        doc_count = chroma_db.get_collection_count()
        
        # Build GraphRAG from existing cloud documents (if any)
        if doc_count > 0:
            try:
                # Get existing documents from cloud ChromaDB for GraphRAG
                all_docs = chroma_db.get_all_documents(limit=1000)
                documents = all_docs.get('documents', [[]])[0] if all_docs.get('documents') else []
                metadatas = all_docs.get('metadatas', [[]])[0] if all_docs.get('metadatas') else []
                
                if documents and metadatas:
                    # Build GraphRAG from cloud documents
                    graph_rag.build_graph_from_documents(documents, metadatas)
                    st.success(f"✅ Connected to cloud ChromaDB with {doc_count} documents from /root/data/")
                else:
                    st.info("📭 Cloud ChromaDB connected but no documents found. Please ensure documents are uploaded to /root/data/ on the cloud server.")
            except Exception as e:
                st.warning(f"Could not build GraphRAG from cloud documents: {str(e)}")
        else:
            st.info("📭 Cloud ChromaDB connected but empty. Please upload documents to /root/data/ folder on the cloud server (1.32.228.33).")
        
        return chroma_db, graph_rag, qwen_client, doc_count
    
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return None, None, None, 0

# Initialize system
with st.spinner("🚀 Initializing Enhanced AI System..."):
    chroma_db, graph_rag, qwen_client, doc_count = initialize_system()

if not chroma_db:
    st.error("❌ System initialization failed. Please check the configuration.")
    st.stop()

# System status
api_status = "✅ Connected" if qwen_client.available else "❌ Not Available"
conversation_count = chroma_db.get_conversation_count()
storage_type = getattr(chroma_db, 'storage_type', 'Unknown')

# Cost summary
cost_summary = st.session_state.cost_calculator.get_cost_summary()

# Status and cost display
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div class="status-container">
        <strong>🧠 System Status:</strong> {doc_count} knowledge chunks in {storage_type} | 
        {conversation_count} conversations logged | GraphRAG: ✅ Active | Qwen API: {api_status}
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
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📊 Cost Analytics", "🗄️ ChromaDB Dashboard"])

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
        - Machine Learning & AI, LangChain, LLMs, Context Prompting
        - Cloud Platforms (Azure, Google Cloud, Aliyun)
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
                "content": "👋 Hello! I'm Rory's AI assistant with cost tracking and analytics. Ask me anything about Rory's professional background, or try one of the sample questions below!"
            })
        
        # Initialize processing flag
        if "processing_sample_question" not in st.session_state:
            st.session_state.processing_sample_question = False
        
        # Quick sample questions
        st.markdown("#### 🎯 Quick Sample Questions:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏢 Tell me about Rory's current role at China CITIC Bank", key="q1"):
                sample_question = "Tell me about Rory's current role at China CITIC Bank International and his key responsibilities"
                st.session_state.messages.append({"role": "user", "content": sample_question})
                st.session_state.processing_sample_question = True
                st.rerun()
        
        with col2:
            if st.button("🚀 What are Rory's key technical skills and achievements?", key="q2"):
                sample_question = "What are Rory's key technical skills in data science and machine learning, and what major achievements has he accomplished?"
                st.session_state.messages.append({"role": "user", "content": sample_question})
                st.session_state.processing_sample_question = True
                st.rerun()
        
        with col3:
            if st.button("📈 Describe Rory's career progression and experience", key="q3"):
                sample_question = "Can you describe Rory's career progression from his early roles to becoming AVP of Data Science? What industries has he worked in?"
                st.session_state.messages.append({"role": "user", "content": sample_question})
                st.session_state.processing_sample_question = True
                st.rerun()
        
        st.markdown("---")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Process sample question if needed
        if st.session_state.processing_sample_question and len(st.session_state.messages) > 0:
            last_message = st.session_state.messages[-1]
            if last_message["role"] == "user":
                prompt = last_message["content"]
                
                # Generate response for sample question
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Processing with GraphRAG..."):
                        try:
                            # Query ChromaDB with configurable max chunks
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
                            st.session_state.processing_sample_question = False
                            
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            st.session_state.processing_sample_question = False
        
        # Chat input
        if prompt := st.chat_input("Ask about Rory's experience, skills, or achievements..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Processing with GraphRAG..."):
                    try:
                        # Query ChromaDB with configurable max chunks
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
        
        fig.update_layout(height=600, showlegend=False, 
                         plot_bgcolor='rgba(0,0,0,0)',
                         paper_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(gridcolor='#404040')
        fig.update_yaxes(gridcolor='#404040')
        
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

with tab3:
    # ChromaDB Dashboard
    st.markdown("### 🗄️ ChromaDB Analytics Dashboard")
    
    try:
        # Get all documents for analysis
        all_docs = chroma_db.get_all_documents()
        documents = all_docs.get('documents', [[]])[0] if all_docs.get('documents') else []
        metadatas = all_docs.get('metadatas', [[]])[0] if all_docs.get('metadatas') else []
        
        if documents:
            # Document statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_docs = len(documents)
            total_chars = sum(len(doc) for doc in documents)
            avg_doc_length = total_chars / total_docs if total_docs > 0 else 0
            
            # Count sources
            sources = {}
            doc_types = {}
            chunk_info = {"total_chunks": 0, "max_chunk_index": 0}
            
            for metadata in metadatas:
                if isinstance(metadata, dict):
                    source = metadata.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                    
                    doc_type = metadata.get('type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    
                    if 'chunk_index' in metadata:
                        chunk_info['total_chunks'] += 1
                        chunk_info['max_chunk_index'] = max(
                            chunk_info['max_chunk_index'], 
                            metadata.get('chunk_index', 0)
                        )
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{total_docs}</h3>
                    <p>Total Documents</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{total_chars:,}</h3>
                    <p>Total Characters</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(sources)}</h3>
                    <p>Unique Sources</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{avg_doc_length:.0f}</h3>
                    <p>Avg Doc Length</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("#### 📁 Documents by Source")
            if sources:
                fig_sources = px.pie(
                    values=list(sources.values()),
                    names=list(sources.keys()),
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_sources.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_sources, use_container_width=True)
            
            # Document length distribution
            st.markdown("#### 📊 Document Length Distribution")
            doc_lengths = [len(doc) for doc in documents]
            
            fig_lengths = px.histogram(
                x=doc_lengths,
                nbins=20,
                title="Distribution of Document Lengths",
                color_discrete_sequence=['#3498db']
            )
            fig_lengths.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis=dict(gridcolor='#404040', title='Document Length (characters)'),
                yaxis=dict(gridcolor='#404040', title='Count')
            )
            st.plotly_chart(fig_lengths, use_container_width=True)
            
            # Chunk analysis
            if chunk_info['total_chunks'] > 0:
                st.markdown("#### 🧩 Chunk Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Chunks", chunk_info['total_chunks'])
                
                with col2:
                    st.metric("Max Chunk Index", chunk_info['max_chunk_index'])
            
            # Sample documents
            st.markdown("#### 📄 Sample Documents")
            
            sample_size = min(3, len(documents))
            for i in range(sample_size):
                with st.expander(f"Document {i+1}: {metadatas[i].get('source', 'Unknown') if i < len(metadatas) else 'Unknown'}"):
                    st.text(documents[i][:500] + "..." if len(documents[i]) > 500 else documents[i])
                    if i < len(metadatas):
                        st.json(metadatas[i])
        
        else:
            st.info("📭 No documents found in ChromaDB. Add some documents to see analytics!")
    
    except Exception as e:
        st.error(f"Error loading ChromaDB analytics: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>🤖 <strong>Enhanced Cloud ChromaDB Compatible AI Assistant</strong></p>
    <p>GraphRAG • Cost Tracking • Analytics Dashboard • {storage_type} Storage</p>
    <p>📧 Contact: chengy823@gmail.com</p>
</div>
""".format(storage_type=storage_type), unsafe_allow_html=True)
