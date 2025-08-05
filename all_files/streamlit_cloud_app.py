"""
Streamlit Cloud Compatible AI Assistant
=====================================

Optimized for Streamlit Community Cloud deployment with:
- No external server dependencies
- Lightweight dependencies only
- Secure API key handling
- Built-in document processing

Author: Rory Chen
"""

import streamlit as st
import os
import requests
import json
from typing import List, Dict, Any
import uuid
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Lightweight in-memory storage (no ChromaDB dependency)
class InMemoryVectorStore:
    """Lightweight vector store for Streamlit Cloud"""
    
    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.conversations = []
        
        # Pre-load Rory's professional information
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize with Rory's professional information"""
        knowledge_chunks = [
            {
                "content": """Rory Chen - Current Role: AVP of Data Science at China CITIC Bank International (November 2022 - Present). 
                Leading data science initiatives, developing ML models for banking applications, managing team of data scientists. 
                Key achievements include developing 20+ ML models with 1.5x business uplift, creating AutoML pipeline reducing coding effort by 80%.""",
                "metadata": {"source": "current_role", "type": "professional", "category": "current_position"}
            },
            {
                "content": """Rory Chen - Career Progression: 8 years of professional experience (2017-2025). 
                Career path: Research Executive at Ipsos (2017-2019) → Data Science Analyst at Cigna (2019-2020) → 
                Assistant Data Science Manager at AXA (2020-2022) → AVP at China CITIC Bank International (2022-Present). 
                Cross-industry expertise in Market Research, Healthcare, Insurance, and Banking.""",
                "metadata": {"source": "career_progression", "type": "professional", "category": "experience"}
            },
            {
                "content": """Rory Chen - Technical Skills: Advanced proficiency in Python, PySpark, SQL. 
                Machine Learning expertise: Deep Learning, NLP, Computer Vision, MLOps, AutoML, LangChain. 
                Cloud Platforms: Azure, Google Cloud, Databricks, Cloudera CDSW. 
                Data Visualization: Tableau, Power BI, Dashboard Development. 
                Specialized in Pega Customer Decision Hub for banking applications.""",
                "metadata": {"source": "technical_skills", "type": "professional", "category": "skills"}
            },
            {
                "content": """Rory Chen - Education: Master's in Quantitative Analysis (2016-2017), 
                Master's in Public Policy (2014-2015), Bachelor's degree (2011-2014). 
                Located in Hong Kong SAR. Contact: chengy823@gmail.com, Phone: 68746551. 
                Strong analytical background with focus on data-driven decision making.""",
                "metadata": {"source": "education_contact", "type": "personal", "category": "background"}
            },
            {
                "content": """Rory Chen - Key Projects and Achievements: 
                • ONE AXA Dashboard: Consolidated all Lines of Business customer and product data into single dashboard
                • Go-Green ESG Initiative: Led environmental sustainability analytics project
                • Cloud Migration: Successfully migrated on-premise infrastructure to Azure
                • AI+BI Framework: Designed comprehensive framework for model development at China CITIC Bank
                • Customer Analytics: Developed predictive models for customer behavior and retention""",
                "metadata": {"source": "projects_achievements", "type": "professional", "category": "achievements"}
            },
            {
                "content": """Rory Chen - Industry Expertise and Business Impact:
                Banking: Risk modeling, customer segmentation, fraud detection, regulatory compliance
                Insurance: Claims prediction, underwriting automation, customer lifetime value
                Healthcare: Medicare analytics, patient outcome prediction, cost optimization
                Market Research: Survey analytics, consumer behavior modeling, market segmentation
                Demonstrated ability to translate business requirements into technical solutions with measurable ROI.""",
                "metadata": {"source": "industry_expertise", "type": "professional", "category": "domain_knowledge"}
            }
        ]
        
        for i, chunk in enumerate(knowledge_chunks):
            self.documents.append(chunk["content"])
            self.metadatas.append(chunk["metadata"])
            self.ids.append(f"knowledge_{i}")
    
    def query_documents(self, query: str, n_results: int = 3) -> Dict:
        """Simple keyword-based document retrieval"""
        query_lower = query.lower()
        matches = []
        
        for i, doc in enumerate(self.documents):
            score = 0
            doc_lower = doc.lower()
            
            # Simple scoring based on keyword matches
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 2:  # Skip very short words
                    score += doc_lower.count(word)
            
            if score > 0:
                matches.append((doc, self.metadatas[i], score))
        
        # Sort by relevance score
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Return top results
        top_matches = matches[:n_results]
        return {
            "documents": [[match[0] for match in top_matches]],
            "metadatas": [[match[1] for match in top_matches]],
            "distances": [[1.0/max(match[2], 1) for match in top_matches]]  # Convert score to distance
        }
    
    def get_collection_count(self) -> int:
        return len(self.documents)
    
    def log_conversation(self, user_query: str, ai_response: str, chunks_used: int = 0):
        """Log conversation for analytics"""
        conversation = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(),
            "user_query": user_query,
            "ai_response": ai_response,
            "chunks_used": chunks_used
        }
        self.conversations.append(conversation)
        return True
    
    def get_conversation_count(self) -> int:
        return len(self.conversations)

class CostCalculator:
    """Calculate API costs and token usage"""
    
    def __init__(self):
        self.pricing = {
            "qwen-turbo": {"input_tokens": 0.0015, "output_tokens": 0.002}
        }
        
        if "total_cost" not in st.session_state:
            st.session_state.total_cost = 0.0
        if "total_input_tokens" not in st.session_state:
            st.session_state.total_input_tokens = 0
        if "total_output_tokens" not in st.session_state:
            st.session_state.total_output_tokens = 0
        if "conversation_costs" not in st.session_state:
            st.session_state.conversation_costs = []
    
    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def calculate_cost(self, input_text: str, output_text: str, model: str = "qwen-turbo") -> Dict[str, Any]:
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        
        input_cost = (input_tokens / 1000) * self.pricing[model]["input_tokens"]
        output_cost = (output_tokens / 1000) * self.pricing[model]["output_tokens"]
        total_cost = input_cost + output_cost
        
        st.session_state.total_cost += total_cost
        st.session_state.total_input_tokens += input_tokens
        st.session_state.total_output_tokens += output_tokens
        
        cost_record = {
            "timestamp": datetime.now(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": total_cost,
            "model": model
        }
        st.session_state.conversation_costs.append(cost_record)
        return cost_record
    
    def get_cost_summary(self) -> Dict[str, Any]:
        return {
            "total_cost": st.session_state.total_cost,
            "total_tokens": st.session_state.total_input_tokens + st.session_state.total_output_tokens,
            "conversation_count": len(st.session_state.conversation_costs)
        }

class QwenAPIClient:
    """Qwen API client with secure key handling"""
    
    def __init__(self):
        # Get API key from Streamlit secrets (secure method)
        try:
            self.api_key = st.secrets["QWEN_API_KEY"]
            self.available = True
        except:
            self.api_key = None
            self.available = False
        
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.cost_calculator = CostCalculator()
    
    def generate_response(self, query: str, context: str) -> tuple:
        if not self.available:
            return "❌ API key not configured. Please add QWEN_API_KEY to Streamlit secrets.", None
        
        system_prompt = """You are Rory Chen's professional AI assistant. Provide accurate, professional responses based on the context provided.

CORE INFORMATION:
- Current: AVP of Data Science at China CITIC Bank International (Nov 2022 - Present)
- Experience: 8 years across Market Research, Healthcare, Insurance, Banking
- Education: Master's in Quantitative Analysis, Master's in Public Policy
- Location: Hong Kong SAR
- Contact: chengy823@gmail.com

Provide comprehensive, professional responses."""

        combined_prompt = f"{system_prompt}\n\nQuestion: {query}\n\nContext: {context}\n\nResponse:"
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "qwen-turbo",
                "input": {"prompt": combined_prompt},
                "parameters": {"max_tokens": 4000, "temperature": 0.7}
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['output']['text'].strip()
                cost_info = self.cost_calculator.calculate_cost(combined_prompt, ai_response)
                return ai_response, cost_info
            else:
                return "❌ API request failed. Please check your API key.", None
                
        except Exception as e:
            return f"❌ Error: {str(e)}", None

# Streamlit App Configuration
st.set_page_config(
    page_title="Rory's AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        max-width: 1200px;
        padding-top: 1rem;
    }
    .header-container {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize system
@st.cache_resource
def initialize_system():
    vector_store = InMemoryVectorStore()
    qwen_client = QwenAPIClient()
    return vector_store, qwen_client

# Header
st.markdown("""
<div class="header-container">
    <h1>🤖 Rory's AI Assistant</h1>
    <p>Streamlit Cloud Compatible • Professional AI Assistant</p>
</div>
""", unsafe_allow_html=True)

# Initialize
vector_store, qwen_client = initialize_system()

# Status
doc_count = vector_store.get_collection_count()
api_status = "✅ Connected" if qwen_client.available else "❌ Configure API Key"

st.markdown(f"""
<div class="status-container">
    <strong>📊 System Status:</strong> {doc_count} knowledge chunks loaded | 
    Qwen API: {api_status} | Ready for deployment
</div>
""", unsafe_allow_html=True)

# Main interface
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("""
    ### 👨‍💼 Rory Chen
    **AVP of Data Science**  
    China CITIC Bank International
    
    **Experience:** 8 years
    **Industries:** Banking, Insurance, Healthcare
    **Skills:** Python, ML, Cloud, Analytics
    **Email:** chengy823@gmail.com
    """)

with col2:
    st.markdown("### 💬 Chat Assistant")
    
    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hello! I'm Rory's AI assistant. Ask me about his professional background!"}
        ]
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about Rory's experience..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Query knowledge base
                results = vector_store.query_documents(prompt, n_results=3)
                context = "\n\n".join(results["documents"][0]) if results["documents"][0] else ""
                
                # Generate response
                response, cost_info = qwen_client.generate_response(prompt, context)
                st.markdown(response)
                
                # Show cost info
                if cost_info:
                    st.caption(f"💰 Cost: ${cost_info['total_cost']:.4f} | Tokens: {cost_info['input_tokens'] + cost_info['output_tokens']}")
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                vector_store.log_conversation(prompt, response, len(results["documents"][0]))

# Analytics section
if st.session_state.get("conversation_costs"):
    st.markdown("### 📊 Usage Analytics")
    
    df = pd.DataFrame(st.session_state.conversation_costs)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    with col2:
        st.metric("Total Tokens", f"{st.session_state.total_input_tokens + st.session_state.total_output_tokens:,}")
    with col3:
        st.metric("Conversations", len(st.session_state.conversation_costs))
    
    # Cost chart
    if len(df) > 1:
        fig = px.line(df.reset_index(), x='index', y='total_cost', title='Cost per Conversation')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("📧 **Contact:** chengy823@gmail.com | 🌐 **Deployed on Streamlit Cloud**")
