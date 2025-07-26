import streamlit as st
import os
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import glob
from typing import List, Dict, Tuple
import PyPDF2
import fitz  # PyMuPDF for better PDF handling
from PIL import Image
import io
import sqlite3
from datetime import datetime
import uuid
import networkx as nx
from collections import defaultdict, Counter
import re
import requests
import json

class GraphRAG:
    """Simplified Graph-based Retrieval Augmented Generation system"""
    
    def __init__(self):
        self.knowledge_graph = nx.Graph()
        self.entity_embeddings = {}
        self.document_entities = defaultdict(list)
        self.entity_documents = defaultdict(list)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_entities(self, text: str, doc_id: str) -> List[str]:
        """Extract entities from text using pattern matching"""
        entities = []
        
        # Extract key professional entities
        patterns = {
            'companies': r'\b(?:China CITIC Bank|AXA|Cigna|Ipsos|City University|Education University)\b',
            'positions': r'\b(?:AVP|Assistant.*Manager|Data Science Analyst|Research Executive)\b',
            'skills': r'\b(?:Python|R|SQL|Machine Learning|Deep Learning|NLP|Tableau|Power BI|Azure|AWS)\b',
            'years': r'\b(?:20\d{2})\b',
            'degrees': r'\b(?:Master|Bachelor|MBA|PhD).*?(?:Arts|Science|Business|Engineering)\b'
        }
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = match.strip()
                if entity and len(entity) > 2:
                    entities.append(f"{category}:{entity}")
                    self.document_entities[doc_id].append(entity)
                    self.entity_documents[entity].append(doc_id)
        
        return entities
    
    def build_knowledge_graph(self, documents_dict: Dict[str, List[Tuple[str, Dict]]]):
        """Build knowledge graph from documents"""
        for doc_type, docs in documents_dict.items():
            for i, (text, metadata) in enumerate(docs):
                doc_id = f"{doc_type}_{i}_{metadata.get('source', 'unknown')}"
                
                # Extract entities from document
                entities = self.extract_entities(text, doc_id)
                
                # Add entities to graph
                for entity in entities:
                    if not self.knowledge_graph.has_node(entity):
                        self.knowledge_graph.add_node(entity, 
                                                    type=entity.split(':')[0] if ':' in entity else 'general',
                                                    documents=[doc_id])
                    else:
                        self.knowledge_graph.nodes[entity]['documents'].append(doc_id)
                
                # Create edges between co-occurring entities
                for j, entity1 in enumerate(entities):
                    for entity2 in entities[j+1:]:
                        if self.knowledge_graph.has_edge(entity1, entity2):
                            self.knowledge_graph[entity1][entity2]['weight'] += 1
                        else:
                            self.knowledge_graph.add_edge(entity1, entity2, weight=1, documents=[doc_id])
        
        # Generate embeddings for entities
        entity_names = list(self.knowledge_graph.nodes())
        if entity_names:
            embeddings = self.model.encode(entity_names, show_progress_bar=False)
            for entity, embedding in zip(entity_names, embeddings):
                self.entity_embeddings[entity] = embedding
    
    def graph_based_retrieval(self, query: str, k: int = 5) -> List[Tuple[str, float, List[str]]]:
        """Retrieve relevant information using graph structure"""
        if not self.knowledge_graph.nodes():
            return []
        
        # Find relevant entities based on query
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        
        entity_scores = {}
        for entity, embedding in self.entity_embeddings.items():
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            entity_scores[entity] = similarity
        
        # Get top entities
        top_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Expand search using graph connections
        expanded_entities = set()
        for entity, score in top_entities:
            expanded_entities.add(entity)
            # Add connected entities with weighted scores
            for neighbor in self.knowledge_graph.neighbors(entity):
                edge_weight = self.knowledge_graph[entity][neighbor]['weight']
                expanded_entities.add(neighbor)
        
        # Collect relevant documents and sources
        relevant_docs = defaultdict(float)
        source_info = defaultdict(list)
        
        for entity in expanded_entities:
            if entity in self.knowledge_graph.nodes():
                docs = self.knowledge_graph.nodes[entity].get('documents', [])
                entity_score = entity_scores.get(entity, 0)
                
                for doc in docs:
                    relevant_docs[doc] += entity_score
                    source_info[doc].append(entity)
        
        # Return top documents with scores and sources
        top_docs = sorted(relevant_docs.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for doc_id, score in top_docs:
            sources = source_info[doc_id]
            results.append((doc_id, score, sources))
        
        return results

class ConversationDatabase:
    """Database manager for storing Q&A conversations"""
    
    def __init__(self, db_path: str = "conversation_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp DATETIME,
                user_question TEXT,
                ai_response TEXT,
                response_time_ms INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, session_id: str, question: str, response: str, response_time_ms: int = 0):
        """Save a Q&A pair to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            conversation_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            cursor.execute('''
                INSERT INTO conversations (id, session_id, timestamp, user_question, ai_response, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (conversation_id, session_id, timestamp, question, response, response_time_ms))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Database error: {str(e)}")
            return False
    
    def get_conversation_count(self) -> int:
        """Get total number of conversations stored"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except:
            return 0

# Initialize database
@st.cache_resource
def get_conversation_db():
    return ConversationDatabase()

# Page configuration
st.set_page_config(
    page_title="Rory's AI Assistant - Data Science & Analytics Expert",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class DocumentProcessor:
    """Simplified document processor with OCR support"""
    
    def __init__(self):
        self.ocr_available = False
        # Try to import OCR dependencies silently
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.ocr_available = True
        except Exception as e:
            self.ocr_available = False
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using available methods"""
        text = ""
        
        try:
            # Method 1: Try PyMuPDF first
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n"
                elif self.ocr_available:
                    # If OCR is available and no text found, try OCR
                    try:
                        import pytesseract
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        ocr_text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                        if ocr_text.strip():
                            text += f"[OCR Page {page_num + 1}]\n{ocr_text}\n"
                    except Exception as ocr_error:
                        st.warning(f"OCR failed for page {page_num + 1}: {str(ocr_error)}")
            doc.close()
            
            if len(text.strip()) > 100:
                return text
                
        except Exception as e:
            st.warning(f"PyMuPDF extraction failed: {str(e)}")
        
        try:
            # Method 2: Fallback to PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += page_text + "\n"
            
            if len(text.strip()) > 100:
                return text
                
        except Exception as e:
            st.warning(f"PyPDF2 extraction failed: {str(e)}")
        
        if self.ocr_available:
            try:
                # Method 3: Full OCR as last resort
                import pytesseract
                st.info(f"Attempting OCR extraction for {os.path.basename(pdf_path)}...")
                doc = fitz.open(pdf_path)
                for page_num in range(min(10, len(doc))):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    ocr_text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                    if ocr_text.strip():
                        text += f"[OCR Page {page_num + 1}]\n{ocr_text}\n"
                doc.close()
                
                if len(text.strip()) > 50:
                    return text
                    
            except Exception as e:
                st.error(f"OCR extraction failed: {str(e)}")
        
        return f"Failed to extract readable text from {os.path.basename(pdf_path)}."
    
    def process_documents(self, directory: str) -> Dict[str, List[Tuple[str, Dict]]]:
        """Process documents with type separation"""
        documents = {
            "professional": [],
            "mindset": [],
            "general": []
        }
        
        # Process text files
        text_extensions = ['*.txt', '*.md']
        for ext in text_extensions:
            files = glob.glob(os.path.join(directory, ext))
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    if text.strip() and len(text) > 100:
                        filename = os.path.basename(file_path).lower()
                        
                        if 'cv' in filename or 'resume' in filename:
                            doc_type = "professional"
                        elif 'flomo' in filename:
                            doc_type = "mindset"
                        else:
                            doc_type = "general"
                        
                        documents[doc_type].append((text, {
                            "source": os.path.basename(file_path), 
                            "type": "text",
                            "path": file_path,
                            "category": doc_type
                        }))
                        
                except Exception as e:
                    st.warning(f"Failed to read {file_path}: {str(e)}")
        
        # Process PDF files
        pdf_files = glob.glob(os.path.join(directory, '*.pdf'))
        for pdf_path in pdf_files:
            try:
                st.info(f"Processing PDF: {os.path.basename(pdf_path)}")
                text = self.extract_text_from_pdf(pdf_path)
                if text.strip() and len(text) > 50:
                    filename = os.path.basename(pdf_path).lower()
                    
                    if 'cv' in filename or 'resume' in filename:
                        doc_type = "professional"
                    elif 'flomo' in filename:
                        doc_type = "mindset"
                    else:
                        doc_type = "general"
                    
                    documents[doc_type].append((text, {
                        "source": os.path.basename(pdf_path), 
                        "type": "pdf",
                        "path": pdf_path,
                        "category": doc_type
                    }))
                    st.success(f"Successfully processed {os.path.basename(pdf_path)}")
                else:
                    st.warning(f"No readable text found in {os.path.basename(pdf_path)}")
            except Exception as e:
                st.error(f"Failed to process PDF {pdf_path}: {str(e)}")
        
        return documents

class VectorStore:
    """Simplified vector store with document type awareness"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = {
            "professional": [],
            "mindset": [],
            "general": []
        }
        self.embeddings = {
            "professional": [],
            "mindset": [],
            "general": []
        }
        self.metadata = {
            "professional": [],
            "mindset": [],
            "general": []
        }
    
    def add_documents_by_type(self, documents_dict: Dict[str, List[Tuple[str, Dict]]]):
        """Add documents categorized by type"""
        for doc_type, docs in documents_dict.items():
            for text, metadata in docs:
                if text.strip():
                    chunks = self._split_text(text, chunk_size=500, overlap=100)
                    for i, chunk in enumerate(chunks):
                        if chunk.strip() and len(chunk) > 50:
                            self.documents[doc_type].append(chunk)
                            chunk_metadata = metadata.copy()
                            chunk_metadata['chunk_index'] = i
                            self.metadata[doc_type].append(chunk_metadata)
        
        # Generate embeddings for each document type
        for doc_type in self.documents:
            if self.documents[doc_type]:
                self.embeddings[doc_type] = self.model.encode(
                    self.documents[doc_type], 
                    batch_size=32, 
                    show_progress_bar=False
                )
    
    def _split_text(self, text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
        """Split text into semantic chunks based on content sections"""
        # First, try to split by clear section markers
        sections = []
        
        # Split by common CV/resume section markers
        section_markers = [
            r'\n\s*(?:PROFESSIONAL EXPERIENCE|WORK EXPERIENCE|EXPERIENCE|EMPLOYMENT)\s*\n',
            r'\n\s*(?:EDUCATION|ACADEMIC BACKGROUND)\s*\n',
            r'\n\s*(?:TECHNICAL SKILLS|SKILLS|COMPETENCIES)\s*\n',
            r'\n\s*(?:ACHIEVEMENTS|ACCOMPLISHMENTS|KEY ACHIEVEMENTS)\s*\n',
            r'\n\s*(?:PROJECTS|KEY PROJECTS)\s*\n',
            r'\n\s*(?:CONTACT|CONTACT INFORMATION)\s*\n',
            r'\n\s*(?:SUMMARY|PROFILE|OBJECTIVE)\s*\n'
        ]
        
        import re
        current_text = text
        
        # Try to identify and separate different content types
        for marker in section_markers:
            if re.search(marker, current_text, re.IGNORECASE):
                parts = re.split(marker, current_text, flags=re.IGNORECASE)
                if len(parts) > 1:
                    sections.extend(parts)
                    break
        
        if not sections:
            # If no clear sections, split by job positions or companies
            job_patterns = [
                r'\n\s*(?:AVP|Assistant.*Manager|Data Science Analyst|Research Executive|Manager|Director|Senior|Lead)\s+[^\n]*\n',
                r'\n\s*(?:China CITIC Bank|AXA|Cigna|Ipsos)[^\n]*\n'
            ]
            
            for pattern in job_patterns:
                if re.search(pattern, current_text, re.IGNORECASE):
                    parts = re.split(pattern, current_text, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        sections.extend(parts)
                        break
        
        if not sections:
            # Fallback to paragraph-based splitting
            sections = text.split('\n\n')
        
        # Now chunk each section appropriately
        chunks = []
        for section in sections:
            section = section.strip()
            if not section or len(section) < 20:
                continue
            
            # If section is small enough, keep as one chunk
            if len(section.split()) <= chunk_size:
                chunks.append(section)
            else:
                # Split larger sections by sentences
                sentences = section.replace('\n', ' ').split('. ')
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    sentence_length = len(sentence.split())
                    
                    if current_length + sentence_length > chunk_size and current_chunk:
                        chunks.append('. '.join(current_chunk) + '.')
                        # Smaller overlap for better separation
                        overlap_sentences = current_chunk[-1:] if current_chunk else []
                        current_chunk = overlap_sentences + [sentence]
                        current_length = sum(len(s.split()) for s in current_chunk)
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
        
        return chunks if chunks else [text]
    
    def similarity_search_by_type(self, query: str, doc_type: str = "professional", k: int = 3) -> List[Tuple[str, Dict]]:
        """Search within specific document type"""
        if not self.documents[doc_type]:
            return []
        
        query_embedding = self.model.encode([query], show_progress_bar=False)
        similarities = cosine_similarity(query_embedding, self.embeddings[doc_type])[0]
        
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.2:  # Higher threshold for better quality
                results.append((self.documents[doc_type][idx], self.metadata[doc_type][idx]))
        
        return results

class QwenMaxAPI:
    """Qwen-Max API integration for enhanced response generation"""
    
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "sk-015ea57c8b254c4181d30b2de4259d8b")
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.model = "qwen-max"
    
    def generate_response(self, query: str, context: str, graph_sources: List = None, similarity_scores: List = None) -> str:
        """Generate response using Qwen-Max API with GraphRAG context"""
        try:
            # Prepare the prompt with GraphRAG context
            system_prompt = """You are Rory's AI assistant, an expert in data science and analytics. You have access to comprehensive information about Rory Chen's professional background, technical expertise, and career achievements through advanced GraphRAG (Graph-based Retrieval Augmented Generation) technology.

Your role is to:
1. Analyze and consolidate information from multiple GraphRAG sources
2. Provide accurate, helpful, and professional responses about Rory's career
3. Synthesize information from different documents and knowledge graph entities
4. Present information in a clear, organized manner without showing technical details like similarity scores

Focus areas include:
- Rory's 8-year career progression (2017-2025) from Research Executive to AVP level
- Technical expertise in Python, R, SQL, Machine Learning, Deep Learning, NLP, Computer Vision
- Experience across multiple industries: market research, healthcare, insurance, banking
- Leadership and management capabilities in data science teams
- Key achievements including 20+ ML models with 1.5x business uplift
- Contact information and professional networking

Always consolidate and digest the GraphRAG context before responding. Do not show raw context or similarity scores to users."""

            # Build comprehensive context description
            context_description = "Retrieved information from multiple knowledge sources"
            if graph_sources:
                unique_sources = set()
                for source_info in graph_sources:
                    if isinstance(source_info, tuple) and len(source_info) >= 3:
                        doc_id, score, entities = source_info
                        unique_sources.add(doc_id.split('_')[0])  # Get document type
                if unique_sources:
                    context_description = f"Retrieved from {len(unique_sources)} knowledge source types: {', '.join(unique_sources)}"

            user_prompt = f"""Based on the following context retrieved through GraphRAG technology, please consolidate and synthesize the information to answer the user's question comprehensively:

**User Question:** {query}

**GraphRAG Retrieved Context:**
{context}

**Context Source:** {context_description}

**Instructions:**
1. Analyze and consolidate all the provided context information
2. Synthesize the information into a coherent, professional response
3. Focus on directly answering the user's question
4. Organize the information logically and clearly
5. Do not mention similarity scores, raw context, or technical retrieval details
6. If multiple sources provide related information, combine them intelligently
7. Maintain professional tone and structure

Please provide a comprehensive response that demonstrates the power of GraphRAG analysis while being user-friendly."""

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "input": {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                },
                "parameters": {
                    "temperature": 0.3,
                    "max_tokens": 1500,
                    "top_p": 0.8
                }
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if "output" in result and "text" in result["output"]:
                    generated_response = result["output"]["text"].strip()
                    # Add GraphRAG attribution
                    if graph_sources:
                        generated_response += "\n\n*Response generated using GraphRAG technology with Qwen-Max.*"
                    return generated_response
                else:
                    return "Sorry, I received an unexpected response format from the AI service. Please check the API configuration."
            else:
                return f"Sorry, I encountered an API error (Status: {response.status_code}). Please check the API key and try again."
                
        except Exception as e:
            return f"Sorry, I encountered an error while generating the response: {str(e)}. Please check your network connection and API configuration."

class SimplifiedReasoningChain:
    """Simplified reasoning system with GraphRAG and Qwen-Max"""
    
    def __init__(self, vector_store: VectorStore, graph_rag: GraphRAG = None):
        self.vector_store = vector_store
        self.graph_rag = graph_rag
        self.qwen_api = QwenMaxAPI()
    
    def __call__(self, inputs: Dict) -> Dict:
        """Simplified reasoning processing with GraphRAG"""
        query = inputs.get("query", "").strip()
        
        # Use GraphRAG for enhanced retrieval if available
        graph_sources = []
        if self.graph_rag:
            graph_results = self.graph_rag.graph_based_retrieval(query, k=5)
            if graph_results:
                graph_sources = [(doc_id, score, sources) for doc_id, score, sources in graph_results]
        
        # Get professional context
        professional_docs = self.vector_store.similarity_search_by_type(query, "professional", k=3)
        
        if not professional_docs and not graph_sources:
            # No relevant information found - be honest about it
            return {"result": "I'm sorry, but I don't have relevant information in my knowledge base to answer your question about that topic. My knowledge is focused on Rory's professional background, technical skills, career experience, and related topics. Could you please ask something about his work experience, technical expertise, or professional achievements?"}
        
        # Build context from available information
        professional_context = "\n\n".join([doc[0] for doc in professional_docs])
        
        # Simple query analysis for basic routing
        query_lower = query.lower()
        
        # Always use GraphRAG + Qwen-Max for response generation (NO FALLBACK)
        if professional_context.strip() or graph_sources:
            # Collect similarity scores from professional docs
            similarity_scores = []
            for doc, metadata in professional_docs:
                # Extract similarity score if available (would need to be added to similarity_search_by_type)
                similarity_scores.append(0.8)  # Placeholder - actual scores would come from vector search
            
            # Use Qwen-Max API with GraphRAG context - NO FALLBACK
            qwen_response = self.qwen_api.generate_response(
                query, 
                professional_context, 
                graph_sources, 
                similarity_scores
            )
            
            return {"result": qwen_response}
        
        # If no relevant context found at all
        return {"result": "I'm sorry, but I don't have sufficient information in my knowledge base to answer your question about that topic. My knowledge is focused on Rory's professional background, technical skills, career experience, and related topics. Could you please ask something more specific about his work experience or expertise?"}

@st.cache_resource
def initialize_simplified_ai_system():
    """Initialize simplified AI system"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize document processor
        doc_processor = DocumentProcessor()
        
        # Process documents
        documents_dict = doc_processor.process_documents(current_dir)
        
        # Fallback content if no documents found
        if not any(documents_dict.values()):
            fallback_content = """
            Rory Chen - Data Science & Analytics Expert
            
            Professional Experience (8 years: 2017-2025):
            AVP, Data Science at China CITIC Bank International (Nov 2022 - Current)
            - Led data science initiatives in retail banking with AI-driven decision-making
            - Developed 20+ ML models achieving 1.5x business uplift vs control group
            - Created AutoML pipeline reducing coding effort by 80%
            - Implemented Next-Best-Action (NBA) model framework
            
            Assistant Data Science Manager at AXA Hong Kong and Macau (Sep 2019 - Nov 2022)
            - Advanced data capabilities for insurance analytics and predictive modeling
            - Designed AI+BI framework and ONE AXA Dashboard (100+ monthly users)
            - Led cloud migration from on-premise to Azure infrastructure
            
            Data Science Analyst at Cigna International Office (Aug 2018 - Sep 2019)
            - Supported US Medicare analytics with predictive modeling focus
            
            Research Executive at Ipsos Hong Kong (2017 - 2018)
            - Conducted market research and data analysis for various industries
            
            Education:
            Master of Arts in Quantitative Analysis for Business, City University of Hong Kong (2016-2017)
            Master of Arts in Public Policy and Management, City University of Hong Kong (2014-2015)
            Bachelor of Social Sciences (Honours), Education University of Hong Kong (2011-2014)
            
            Technical Skills:
            Programming: Python, R, SQL (Advanced proficiency)
            Machine Learning: Deep Learning, NLP, Computer Vision, MLOps
            Analytics: Predictive Modeling, Customer Analytics, Time Series Analysis
            Cloud Platforms: Azure, AWS, Databricks, Cloudera CDSW
            Visualization: Tableau, Power BI, Dashboard Development
            Tools: Spark, Hadoop, Docker, Kubernetes, AutoML
            
            Contact: chengy823@gmail.com
            Location: Hong Kong SAR
            """
            documents_dict["professional"] = [(fallback_content, {"source": "profile", "type": "text", "category": "professional"})]
        
        # Create vector store
        vector_store = VectorStore()
        vector_store.add_documents_by_type(documents_dict)
        
        # Initialize GraphRAG system
        graph_rag = GraphRAG()
        graph_rag.build_knowledge_graph(documents_dict)
        
        # Create simplified reasoning chain
        reasoning_chain = SimplifiedReasoningChain(vector_store, graph_rag)
        
        # Calculate totals
        total_docs = sum(len(docs) for docs in documents_dict.values())
        total_chunks = sum(len(chunks) for chunks in vector_store.documents.values())
        
        return reasoning_chain, total_docs, total_chunks
    
    except Exception as e:
        st.error(f"Failed to initialize AI system: {str(e)}")
        return None, 0, 0

# Custom CSS for professional styling
st.markdown("""
<style>
    .main .block-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px;
        line-height: 1.6;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .main-header h3 {
        font-size: 1.5rem !important;
        font-weight: 400 !important;
        margin-bottom: 1rem !important;
    }
    
    .main-header p {
        font-size: 1.1rem !important;
        opacity: 0.9;
    }
    
    .status-info {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin-bottom: 1rem;
        font-size: 1rem;
    }
    
    .chat-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #ddd;
        background: white;
        color: #333;
        padding: 0.75rem 1rem;
        margin: 0.3rem 0;
        transition: all 0.3s;
        font-size: 0.95rem;
        font-weight: 500;
        text-align: left;
    }
    
    .stButton > button:hover {
        background: #f0f8ff;
        border-color: #3498db;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.2);
    }
    
    .stChatMessage {
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background-color: #3498db !important;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background-color: #2ecc71 !important;
    }
    
    .stTextInput > div > div > input {
        font-size: 1rem !important;
        padding: 0.75rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 Rory's AI Assistant</h1>
    <h3>Data Science & Analytics Expert</h3>
    <p>Ask me anything about Rory's experience, skills, projects, or insights!</p>
</div>
""", unsafe_allow_html=True)

# Initialize simplified AI system and database
with st.spinner("🚀 Initializing Simplified AI system..."):
    qa_chain, total_docs, total_chunks = initialize_simplified_ai_system()
    conversation_db = get_conversation_db()

if qa_chain is None:
    st.error("❌ Rory's AI system is not available. Please check the system configuration.")
    st.stop()

# Initialize session ID for conversation tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Get conversation count for display
total_conversations = conversation_db.get_conversation_count()

# Display system status
st.markdown(f"""
<div class="status-info">
    <strong style="color: #2c3e50;">🧠 Simplified AI System Status:</strong> <span style="color: #34495e;">{total_docs} documents processed, 
    {total_chunks} knowledge chunks available | 🎯 GraphRAG Enhanced | 💾 {total_conversations} conversations stored</span>
</div>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    # CV Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h1 style="color: white; margin-bottom: 0.5rem;">👨‍💼 Rory Chen</h1>
            <h3 style="color: #e8f4fd; margin-bottom: 0.5rem;">Data Science & Analytics Expert</h3>
            <p style="color: #b8d4f0; font-size: 1.1rem;">🏢 China CITIC Bank International Limited</p>
            <p style="color: #b8d4f0;">📍 Hong Kong SAR | 🔗 LinkedIn: 688 followers, 500+ connections</p>
        </div>
    </div>
    
    <div style="background: white; padding: 2rem; border-radius: 15px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 2rem;">
        <h2 style="color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 0.5rem;">💼 Professional Experience (8 years: 2017-2025)</h2>
        <div style="margin: 1rem 0;">
            <div style="margin-bottom: 1.5rem;">
                <h4 style="color: #34495e; margin-bottom: 0.3rem;">🏦 China CITIC Bank International</h4>
                <p style="color: #7f8c8d; margin-bottom: 0.3rem;"><strong>AVP, Data Science</strong> (Nov 2022 - Current)</p>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Led data science initiatives, developed 20+ ML models with 1.5x business uplift.</p>
            </div>
            <div style="margin-bottom: 1.5rem;">
                <h4 style="color: #34495e; margin-bottom: 0.3rem;">🛡️ AXA Hong Kong and Macau</h4>
                <p style="color: #7f8c8d; margin-bottom: 0.3rem;"><strong>Assistant Data Science Manager</strong> (Sep 2019 - Nov 2022)</p>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Advanced data capabilities for insurance analytics and cloud migration.</p>
            </div>
            <div style="margin-bottom: 1.5rem;">
                <h4 style="color: #34495e; margin-bottom: 0.3rem;">🏥 Cigna International Office</h4>
                <p style="color: #7f8c8d; margin-bottom: 0.3rem;"><strong>Data Science Analyst</strong> (Aug 2018 - Sep 2019)</p>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Supported US Medicare analytics with predictive modeling focus.</p>
            </div>
            <div style="margin-bottom: 1.5rem;">
                <h4 style="color: #34495e; margin-bottom: 0.3rem;">📊 Ipsos Hong Kong</h4>
                <p style="color: #7f8c8d; margin-bottom: 0.3rem;"><strong>Research Executive</strong> (2017 - 2018)</p>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Conducted market research and statistical modeling for consumer behavior.</p>
            </div>
        </div>
    </div>
    
    <div style="background: white; padding: 2rem; border-radius: 15px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 2rem;">
        <h2 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 0.5rem;">🎓 Education</h2>
        <div style="margin: 1rem 0;">
            <div style="margin-bottom: 1rem;">
                <h4 style="color: #34495e; margin-bottom: 0.3rem;">🏛️ City University of Hong Kong</h4>
                <p style="color: #7f8c8d; margin-bottom: 0.2rem;"><strong>Master of Arts in Quantitative Analysis for Business</strong> (2016-2017)</p>
                <p style="color: #7f8c8d; margin-bottom: 0.2rem;"><strong>Master of Arts in Public Policy and Management</strong> (2014-2015)</p>
            </div>
            <div>
                <h4 style="color: #34495e; margin-bottom: 0.3rem;">🏛️ Education University of Hong Kong</h4>
                <p style="color: #7f8c8d;"><strong>Bachelor of Social Sciences (Honours)</strong> (2011-2014)</p>
            </div>
        </div>
    </div>
    
    <div style="background: white; padding: 2rem; border-radius: 15px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
        <h2 style="color: #2c3e50; border-bottom: 3px solid #f39c12; padding-bottom: 0.5rem;">🚀 Core Expertise</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
            <div>
                <h5 style="color: #2980b9; margin-bottom: 0.5rem;">📊 Data Science</h5>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Python, R, SQL, Machine Learning</p>
            </div>
            <div>
                <h5 style="color: #27ae60; margin-bottom: 0.5rem;">🤖 AI & ML</h5>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Deep Learning, NLP, Computer Vision</p>
            </div>
            <div>
                <h5 style="color: #8e44ad; margin-bottom: 0.5rem;">📈 Analytics</h5>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Predictive Modeling, Business Intelligence</p>
            </div>
            <div>
                <h5 style="color: #e67e22; margin-bottom: 0.5rem;">🛠️ Tools</h5>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Tableau, Power BI, Cloud Platforms</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Q&A Input Box Section
    st.markdown("### ❓ Ask Rory's AI Assistant")
    st.markdown("Type your question below or click on the sample questions:")
    
    # Direct Q&A input box
    user_question = st.text_input(
        "Your Question:",
        placeholder="Ask me anything about Rory's experience, skills, or expertise...",
        key="direct_question_input"
    )
    
    if st.button("🚀 Ask Question", key="ask_direct_question") and user_question.strip():
        st.session_state.selected_question = user_question.strip()
        st.rerun()
    
    st.markdown("---")
    
    # Sample Questions Section
    st.markdown("### 💡 Sample Questions - Click to Ask!")
    
    sample_questions = [
        "Tell me about Rory's data science experience and expertise",
        "What programming languages and tools does Rory specialize in?",
        "What are Rory's key achievements in analytics and data science?",
        "What industries has Rory worked in for data science projects?",
        "How can I contact Rory for potential collaboration?",
        "I want to hire a GenAI manager, is Rory a good fit?",
        "How old is Rory based on his education timeline?",
        "What makes Rory unique as a data science professional?"
    ]
    
    # Create clickable sample questions
    for i, question in enumerate(sample_questions):
        if st.button(f"❓ {question}", key=f"sample_q_{i}", help="Click to ask this question"):
            st.session_state.selected_question = question
            st.rerun()
    
    # Chat Interface
    st.markdown("---")
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat header with clear button
    col_chat1, col_chat2 = st.columns([3, 1])
    with col_chat1:
        st.markdown("### 💬 Chat with Rory's AI")
    with col_chat2:
        if st.button("🗑️ Clear History", key="clear_history", help="Clear all conversation history"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "👋 Hello! I'm Rory's simplified AI assistant with GraphRAG technology. I can provide information about Rory's professional background, technical expertise, and career achievements. If I don't have relevant information in my knowledge base, I'll let you know honestly!"
        })

    # Handle selected question from sample questions
    if "selected_question" in st.session_state:
        selected_q = st.session_state.selected_question
        st.session_state.messages.append({"role": "user", "content": selected_q})
        
        # Show processing indicator
        with st.spinner("🧠 Processing with GraphRAG..."):
            try:
                result = qa_chain({"query": selected_q})
                
                if isinstance(result, dict):
                    response = result.get("result", "Sorry, I couldn't find a relevant answer.")
                else:
                    response = str(result)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Save conversation to database
                conversation_db.save_conversation(
                    session_id=st.session_state.session_id,
                    question=selected_q,
                    response=response
                )
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error while processing your question: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        del st.session_state.selected_question
        st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about Rory's experience and expertise..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing with GraphRAG..."):
                try:
                    result = qa_chain({"query": prompt})
                    
                    if isinstance(result, dict):
                        response = result.get("result", "Sorry, I couldn't find a relevant answer.")
                    else:
                        response = str(result)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Save conversation to database
                    conversation_db.save_conversation(
                        session_id=st.session_state.session_id,
                        question=prompt,
                        response=response
                    )
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error while processing your question: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🤖 <strong>Rory's Simplified AI Assistant</strong> | Powered by GraphRAG Technology | 📄 PDF Processing with OCR</p>
    <p>📧 For direct contact: chengy823@gmail.com</p>
    <p><em>Simplified system with honest responses - if information isn't in the knowledge base, I'll tell you directly</em></p>
</div>
""", unsafe_allow_html=True)
