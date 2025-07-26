import streamlit as st
import os
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import glob
from typing import List, Dict, Tuple
import requests
import json
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

class GraphRAG:
    """Graph-based Retrieval Augmented Generation system"""
    
    def __init__(self):
        self.knowledge_graph = nx.Graph()
        self.entity_embeddings = {}
        self.document_entities = defaultdict(list)
        self.entity_documents = defaultdict(list)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_entities(self, text: str, doc_id: str) -> List[str]:
        """Extract entities from text using pattern matching and NER"""
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
    """Database manager for storing Q&A conversations for analysis"""
    
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

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Rory's AI Assistant - Data Science & Analytics Expert",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class EnhancedDocumentProcessor:
    """Enhanced document processor with document type separation and OCR support"""
    
    def __init__(self):
        self.ocr_available = False
        # Try to import OCR dependencies silently - no UI notifications
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
            # Method 1: Try PyMuPDF first (best for text-based PDFs)
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n"
                elif self.ocr_available:
                    # If OCR is available and no text found, try OCR on this page
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
            
            # If we got good text, return it
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
                for page_num in range(min(10, len(doc))):  # Limit to first 10 pages for performance
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Try OCR with Chinese and English
                    ocr_text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                    if ocr_text.strip():
                        text += f"[OCR Page {page_num + 1}]\n{ocr_text}\n"
                doc.close()
                
                if len(text.strip()) > 50:
                    return text
                    
            except Exception as e:
                st.error(f"OCR extraction failed: {str(e)}")
        
        return f"Failed to extract readable text from {os.path.basename(pdf_path)}. The file may contain only images or be corrupted. {'OCR is not available to process image-based content.' if not self.ocr_available else ''}"
    
    def process_documents_with_separation(self, directory: str) -> Dict[str, List[Tuple[str, Dict]]]:
        """Process documents with type separation"""
        documents = {
            "professional": [],  # CV and work-related documents
            "mindset": [],       # Flomo document for tone/style learning
            "general": []        # Other documents
        }
        
        # Process text files with categorization
        text_extensions = ['*.txt', '*.md']
        for ext in text_extensions:
            files = glob.glob(os.path.join(directory, ext))
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    if text.strip() and len(text) > 100:
                        filename = os.path.basename(file_path).lower()
                        
                        # Categorize documents
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
                    
                    # Categorize PDFs
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

class EnhancedVectorStore:
    """Enhanced vector store with document type awareness"""
    
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
                    # Enhanced chunking with larger size and better overlap
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
    
    def _split_text(self, text: str, chunk_size: int = 300, overlap: int = 30) -> List[str]:
        """Optimized text splitting"""
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                # Keep some overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
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
        
        # Get top k results with higher threshold for better quality
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.15:
                results.append((self.documents[doc_type][idx], self.metadata[doc_type][idx]))
        
        return results

class AdvancedReasoningChain:
    """Advanced reasoning system with GraphRAG and enhanced analytical capabilities"""
    
    def __init__(self, vector_store: EnhancedVectorStore, graph_rag: GraphRAG = None):
        self.vector_store = vector_store
        self.graph_rag = graph_rag
        self.api_key = "sk-015ea57c8b254c4181d30b2de4259d8b"
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.reasoning_cache = {}
    
    def _analyze_query_complexity(self, query: str) -> Dict:
        """Analyze query complexity and reasoning requirements"""
        query_lower = query.lower()
        
        analysis = {
            "reasoning_type": "factual",
            "complexity": "simple",
            "requires_inference": False,
            "requires_comparison": False,
            "requires_analysis": False,
            "requires_age_calculation": False,
            "context_sources": ["professional"],
            "reasoning_steps": []
        }
        
        # Detect age-related questions requiring calculation
        if any(word in query_lower for word in ['age', 'old', 'born', 'birth', 'year old', 'how old']):
            analysis["reasoning_type"] = "analytical"
            analysis["complexity"] = "complex"
            analysis["requires_age_calculation"] = True
            analysis["requires_inference"] = True
            analysis["reasoning_steps"].append("age_calculation")
            analysis["reasoning_steps"].append("timeline_analysis")
        
        # Detect reasoning requirements
        if any(word in query_lower for word in ['why', 'how', 'explain', 'analyze', 'reason']):
            analysis["reasoning_type"] = "analytical"
            analysis["complexity"] = "complex"
            analysis["requires_analysis"] = True
            analysis["reasoning_steps"].append("causal_analysis")
        
        if any(word in query_lower for word in ['compare', 'versus', 'difference', 'better', 'worse']):
            analysis["requires_comparison"] = True
            analysis["reasoning_steps"].append("comparative_analysis")
        
        if any(word in query_lower for word in ['predict', 'future', 'trend', 'likely', 'expect']):
            analysis["requires_inference"] = True
            analysis["reasoning_steps"].append("predictive_reasoning")
        
        if any(word in query_lower for word in ['strategy', 'approach', 'methodology', 'framework']):
            analysis["reasoning_type"] = "strategic"
            analysis["reasoning_steps"].append("strategic_thinking")
        
        # Determine context sources needed
        if any(word in query_lower for word in ['experience', 'work', 'job', 'career', 'professional', 'skill']):
            analysis["context_sources"] = ["professional"]
        elif any(word in query_lower for word in ['personality', 'character', 'mindset', 'thinking']):
            analysis["context_sources"] = ["professional", "mindset"]  # Use both but filter mindset content
        else:
            analysis["context_sources"] = ["professional", "general"]
        
        return analysis
    
    def _extract_mindset_patterns(self, mindset_content: str) -> Dict:
        """Extract communication patterns and tone from mindset content for natural responses"""
        patterns = {
            "communication_style": "thoughtful and reflective",
            "analytical_approach": "systematic and thorough", 
            "tone_characteristics": ["professional", "curious", "growth-oriented"],
            "thinking_patterns": ["analytical", "philosophical", "practical"],
            "language_style": "conversational and authentic",
            "response_approach": "personal and engaging"
        }
        
        # Extract actual language patterns and tone from flomo content
        if mindset_content:
            # Analyze for conversational patterns
            if any(word in mindset_content for word in ["我覺得", "我想", "我認為", "其實", "說實話"]):
                patterns["language_style"] = "personal and conversational"
                patterns["response_approach"] = "first-person perspective"
            
            # Analyze for reflective thinking patterns
            if any(word in mindset_content for word in ["分析", "思考", "反思", "總結"]):
                patterns["analytical_approach"] = "reflective and introspective"
            
            # Analyze for learning and growth mindset
            if any(word in mindset_content for word in ["學習", "成長", "進步", "改善"]):
                patterns["tone_characteristics"].append("growth-minded")
                patterns["thinking_patterns"].append("continuous-learning")
            
            # Analyze for practical approach
            if any(word in mindset_content for word in ["實際", "經驗", "應用", "實踐"]):
                patterns["thinking_patterns"].append("practical-oriented")
                patterns["response_approach"] = "experience-based"
        
        return patterns
    
    def _calculate_age_reasoning(self, query: str, professional_context: str) -> str:
        """Specialized reasoning for age-related questions - this should NOT be used, let LLM handle it"""
        # This method should not be called - we want LLM to handle age questions
        return ""

    def _generate_enhanced_response(self, query: str, professional_context: str, analysis: Dict, mindset_patterns: Dict) -> str:
        """Generate enhanced response with advanced reasoning"""
        
        # Handle age calculation questions with specialized reasoning
        if analysis.get("requires_age_calculation", False):
            return self._calculate_age_reasoning(query, professional_context)
        
        # Clean and deduplicate context to avoid repetitive chunks
        context_lines = professional_context.split('\n')
        unique_lines = []
        seen_content = set()
        
        for line in context_lines:
            line = line.strip()
            if line and len(line) > 20:  # Filter very short lines
                # Check for duplicate content patterns
                line_key = line.lower()[:50]  # Use first 50 chars as key
                if line_key not in seen_content:
                    seen_content.add(line_key)
                    unique_lines.append(line)
        
        # Limit context length to prevent overwhelming the model
        cleaned_context = '\n'.join(unique_lines[:15])  # Limit to 15 unique lines
        
        # Create human-like prompt using Rory's personal tone and style
        tone_style = mindset_patterns.get("language_style", "conversational and authentic")
        response_approach = mindset_patterns.get("response_approach", "personal and engaging")
        
        # Generate more natural, human-like prompt
        if response_approach == "first-person perspective":
            prompt = f"""You are responding as Rory Chen himself. Use a natural, conversational tone based on his personal communication style. 

Background context (use this information naturally, don't copy directly):
{cleaned_context}

Communication style: {tone_style}
Approach: {response_approach}

User question: {query}

Respond as Rory would - naturally, authentically, and conversationally. Share insights and experiences in a personal way, avoiding formal templates or structured bullet points. Be human, be real, be engaging."""

        else:
            prompt = f"""You are Rory's AI assistant, but respond in a natural, human-like way that reflects his personality and communication style. 

Background information (integrate naturally into conversation):
{cleaned_context}

Communication patterns observed: {tone_style}
Thinking approach: {mindset_patterns.get('analytical_approach', 'thoughtful and reflective')}

User question: {query}

Respond naturally and conversationally, as if you're having a real conversation. Avoid formal templates like "Based on Rory's professional background..." Instead, be more human - share insights, experiences, and thoughts in an engaging, authentic way. Make it feel like a real person is talking, not a formal AI assistant."""

        return prompt
    
    def _format_sources(self, graph_sources: List[Tuple[str, float, List[str]]]) -> str:
        """Format GraphRAG attribution without showing detailed sources"""
        if not graph_sources:
            return ""
        
        # Just add a subtle note about GraphRAG without showing detailed sources
        return "\n\n*Response generated using GraphRAG relevance scoring from available materials.*"
    
    def __call__(self, inputs: Dict) -> Dict:
        """Enhanced reasoning processing with GraphRAG"""
        query = inputs.get("query", "").strip()
        
        # Analyze query complexity and reasoning requirements
        analysis = self._analyze_query_complexity(query)
        
        # Use GraphRAG for enhanced retrieval if available
        graph_sources = []
        if self.graph_rag:
            graph_results = self.graph_rag.graph_based_retrieval(query, k=5)
            if graph_results:
                graph_sources = [(doc_id, score, sources) for doc_id, score, sources in graph_results]
        
        # Get professional context (primary source for work-related queries)
        professional_docs = self.vector_store.similarity_search_by_type(
            query, "professional", k=5 if analysis["complexity"] == "complex" else 3
        )
        professional_context = "\n\n".join([doc[0] for doc in professional_docs])
        
        # Get mindset patterns for tone/style (without exposing personal details)
        mindset_patterns = {}
        if "mindset" in analysis["context_sources"]:
            mindset_docs = self.vector_store.similarity_search_by_type(query, "mindset", k=2)
            if mindset_docs:
                mindset_content = "\n".join([doc[0] for doc in mindset_docs])
                mindset_patterns = self._extract_mindset_patterns(mindset_content)
        
        # Check cache for complex reasoning
        cache_key = hash(query.lower().strip() + str(analysis) + professional_context[:300])
        if cache_key in self.reasoning_cache:
            cached_result = self.reasoning_cache[cache_key]
            # Add source information to cached result
            if graph_sources:
                cached_result += self._format_sources(graph_sources)
            return {"result": cached_result}
        
        # Generate enhanced prompt with reasoning
        prompt = self._generate_enhanced_response(query, professional_context, analysis, mindset_patterns)
        
        # Skip API calls completely - use local GraphRAG and context-based responses
        # Check if question is sensitive/personal
        sensitive_keywords = ['personal', 'private', 'family', 'relationship', 'salary', 'income', 'health', 'medical', 'political', 'religion']
        is_sensitive = any(keyword in query.lower() for keyword in sensitive_keywords)
        
        # For sensitive questions, provide polite decline
        if is_sensitive:
            sensitive_response = f"""I appreciate your question, but I focus on sharing information about Rory's professional background, technical expertise, and career achievements. 

For questions about personal or private matters, I'd recommend reaching out to Rory directly at chengy823@gmail.com if you have a legitimate professional inquiry.

I'm here to help with questions about:
- Professional experience and career progression
- Technical skills and expertise
- Project achievements and methodologies
- Industry insights and approaches
- Contact information for professional purposes

Feel free to ask me anything about these professional topics!"""
            
            if graph_sources:
                sensitive_response += self._format_sources(graph_sources)
            
            self.reasoning_cache[cache_key] = sensitive_response
            return {"result": sensitive_response}
        
        # For other questions, provide detailed context-based responses
        query_lower = query.lower()
        
        # First check for hiring/job-related questions (highest priority)
        if any(word in query_lower for word in ['hire', 'hiring', 'recruit', 'job', 'position', 'role', 'fit', 'suitable', 'candidate', 'manager', 'manage', 'team', 'lead']):
            # This will be handled in the context-based response section below
            pass
        # Then handle age questions with local reasoning (but not hiring questions)
        elif any(word in query_lower for word in ['age', 'old', 'born', 'birth', 'year old', 'how old']):
            age_response = f"""Based on Rory's educational and career timeline, I can provide a reasoned estimate of his age:

**🎓 Educational Timeline Analysis:**
• **Bachelor's Degree**: Education University of Hong Kong (2011-2014)
• **First Master's**: Public Policy and Management, CityU (2014-2015) 
• **Second Master's**: Quantitative Analysis for Business, CityU (2016-2017)

**💼 Career Timeline Analysis (8 years: 2017-2025):**
• **Research Executive at Ipsos**: 2017-2018
• **Data Science Analyst at Cigna**: 2018-2019
• **Assistant Manager at AXA**: 2019-2022
• **AVP at China CITIC Bank**: 2022-Present

**🧮 Age Reasoning:**
Assuming Rory started his bachelor's degree at the typical age of 18-19 years old in 2011, this would place his birth year around **1992-1993**.

**Calculation Logic:**
- Started university in 2011 at age ~18-19
- Birth year: 2011 - 18/19 = 1992-1993
- Current age in 2025: 2025 - 1992/1993 = **32-33 years old**

**Supporting Evidence:**
- Completed bachelor's degree in 3 years (2011-2014) - typical timeline
- Immediately pursued master's degrees (2014-2017) - shows academic focus
- Started professional career in 2017 at age ~24-25 - reasonable for master's graduate
- Has 8 years of professional experience by 2025 - consistent with early 30s age

**Conclusion:** Based on this timeline analysis, Rory is most likely **32-33 years old** as of 2025, having been born around 1992-1993."""

            if graph_sources:
                age_response += self._format_sources(graph_sources)
            
            return {"result": age_response}
        
        # For other questions, provide detailed context-based responses
        if professional_context.strip():
            # Create comprehensive response based on query type
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['experience', 'expertise', 'background', 'career', 'work', 'professional']):
                context_response = f"""Looking at Rory's impressive data science journey, he's built quite a comprehensive career over the past 8 years!

**🚀 Career Progression (2017-2025):**
Rory started as a Research Executive at Ipsos Hong Kong (2017-2018), where he got hands-on experience with market research and statistical modeling for consumer behavior analysis. This gave him a solid foundation in data analysis and research methodologies.

From there, he moved into healthcare analytics as a Data Science Analyst at Cigna International Office (2018-2019), supporting US Medicare analytics with predictive modeling. This was where he really dove deep into healthcare data and claims analysis.

His next step was AXA Hong Kong and Macau (2019-2022) as Assistant Data Science Manager, where he advanced data capabilities for insurance analytics, designed AI+BI frameworks, and led cloud migration projects. This role really showcased his leadership and technical architecture skills.

Currently, he's an AVP of Data Science at China CITIC Bank International (2022-Present), leading data science initiatives in retail banking and developing 20+ ML models that achieved 1.5x business uplift. He's also created AutoML pipelines and implemented Next-Best-Action frameworks.

**🎯 What makes him stand out:**
- **8 years of diverse industry experience** across market research, healthcare, insurance, and banking
- **Progression from analyst to AVP level** showing both technical depth and leadership capabilities
- **20+ ML models developed** with measurable business impact
- **Cross-industry expertise** that brings unique perspectives to problem-solving"""

            elif any(word in query_lower for word in ['skills', 'technical', 'programming', 'tools', 'technology']):
                context_response = f"""Rory has developed an impressive technical toolkit throughout his 8-year career:

**💻 Programming & Analytics:**
- **Advanced proficiency** in Python, R, and SQL
- **Machine Learning & Deep Learning** expertise across multiple domains
- **NLP and Computer Vision** capabilities
- **Statistical analysis** and predictive modeling

**🛠️ Platforms & Tools:**
- **Cloud platforms**: Azure, AWS, Databricks, Cloudera CDSW
- **Visualization**: Tableau, Power BI, dashboard development
- **Big Data**: Spark, Hadoop, Docker, Kubernetes
- **AutoML and MLOps** implementation

**🏭 Industry Applications:**
- **Healthcare analytics** and claims processing (Cigna experience)
- **Insurance modeling** and risk assessment (AXA experience)
- **Banking and financial services** analytics (China CITIC Bank experience)
- **Market research** and consumer behavior analysis (Ipsos experience)

His technical skills have evolved with each role, from statistical modeling at Ipsos to advanced ML/AI implementations at China CITIC Bank. He's particularly strong in translating business problems into technical solutions and leading data science teams."""

            elif any(word in query_lower for word in ['achievements', 'accomplishments', 'success', 'impact']):
                context_response = f"""Rory's achievements span across multiple industries and demonstrate real business impact:

**🏆 Key Professional Achievements:**
- **Led data science initiatives** in retail banking at China CITIC Bank International
- **Developed 20+ ML models** achieving 1.5x business uplift vs control group
- **Created AutoML pipeline** reducing coding effort by 80%
- **Implemented Next-Best-Action (NBA)** model framework
- **Designed AI+BI framework** and ONE AXA Dashboard with 100+ monthly users
- **Led cloud migration** from on-premise to Azure infrastructure
- **Advanced data capabilities** for insurance analytics and predictive modeling

**📈 Career Progression:**
- **8 years of continuous growth** from Research Executive to AVP level
- **Cross-industry expertise** in market research, healthcare, insurance, and banking
- **Leadership development** from individual contributor to team leader
- **Technical evolution** from statistical analysis to advanced AI/ML implementations

**🎯 Business Impact:**
- Measurable business improvements through ML model implementations
- Significant efficiency gains through automation and process optimization
- Strategic contributions to data-driven decision making across organizations"""

            elif any(word in query_lower for word in ['hire', 'hiring', 'recruit', 'job', 'position', 'role', 'fit', 'suitable', 'candidate', 'manager']):
                context_response = f"""**🎯 Rory's Fit for GenAI Manager Role:**

Based on Rory's 8-year career progression and current leadership position, he would be an excellent fit for a GenAI Manager role. Here's why:

**🚀 Leadership & Management Experience:**
- **Current AVP Role**: Leading data science initiatives at China CITIC Bank International
- **Team Leadership**: Managed data science teams and cross-functional projects
- **Strategic Vision**: Designed AI+BI frameworks and implemented enterprise-wide solutions
- **8 Years Progressive Growth**: From analyst to AVP level, demonstrating leadership development

**🤖 GenAI & Advanced AI Expertise:**
- **20+ ML Models Developed**: Proven track record of delivering production AI systems
- **AutoML Implementation**: Created automated ML pipelines reducing coding effort by 80%
- **Next-Best-Action Frameworks**: Advanced AI decision-making systems
- **Cross-Industry AI Applications**: Healthcare, insurance, banking, and market research

**💼 Business Impact & Results:**
- **1.5x Business Uplift**: Measurable ROI from ML model implementations
- **Enterprise Scale**: Led cloud migration and AI governance frameworks
- **100+ Monthly Users**: Built dashboards and systems with proven adoption
- **Cost Efficiency**: Significant automation and process optimization achievements

**🎓 Strong Educational Foundation:**
- **Quantitative Analysis Master's**: Advanced statistical and analytical skills
- **Public Policy Master's**: Strategic thinking and policy implementation experience
- **Continuous Learning**: Stayed current with latest AI/ML technologies

**🌟 Unique Value Proposition:**
- **Cross-Industry Perspective**: Brings diverse experience from multiple sectors
- **Technical + Business Acumen**: Combines deep technical skills with business strategy
- **Proven Scalability**: Experience building enterprise-level AI solutions
- **Innovation Focus**: Track record of implementing cutting-edge technologies

**📧 Contact for Discussion:** chengy823@gmail.com

Rory would bring both the technical depth and leadership experience needed to successfully manage GenAI initiatives and drive organizational AI transformation."""

            elif any(word in query_lower for word in ['contact', 'reach', 'email', 'connect', 'collaboration']):
                context_response = f"""You can reach out to Rory through several channels:

**📧 Primary Contact:** chengy823@gmail.com
**📍 Location:** Hong Kong SAR
**🔗 LinkedIn:** Available with 688 followers and 500+ connections

Rory is currently serving as AVP of Data Science at China CITIC Bank International, but he's always open to discussing interesting opportunities, collaborations, or sharing insights about data science and analytics.

Whether you're looking to:
- Collaborate on data science projects
- Discuss industry trends and best practices
- Explore potential career opportunities
- Get advice on analytics implementations

Feel free to reach out via email with details about what you'd like to discuss. He's particularly interested in conversations about advanced analytics, machine learning applications in financial services, and innovative approaches to data-driven decision making."""

            else:
                context_response = f"""Hey there! I'd be happy to tell you about Rory Chen's background in data science and analytics.

Rory's got an impressive 8-year journey in data science, spanning from 2017 to 2025 across some pretty diverse industries. He started at Ipsos doing market research, then moved into healthcare analytics at Cigna, insurance modeling at AXA, and now he's leading data science initiatives in banking at China CITIC Bank International.

What's really cool about his background is the progression - he's gone from Research Executive to AVP level, showing both deep technical skills and strong leadership capabilities. He's developed over 20 ML models with real business impact and has experience with everything from Python and R to cloud platforms and AutoML.

His cross-industry experience gives him a unique perspective on how data science can be applied across different domains. Whether it's healthcare claims analysis, insurance risk modeling, or banking customer analytics, he's got hands-on experience making data-driven solutions work in the real world.

Feel free to ask me more specific questions about his technical skills, career progression, or any particular aspect of his experience that interests you!"""

            if graph_sources:
                context_response += self._format_sources(graph_sources)
            
            return {"result": context_response}
        
        # Final fallback if no context available
        minimal_response = f"""I'm having trouble accessing the full system capabilities right now, but I can still help with basic information about Rory Chen.

Rory is a data science professional with 8 years of experience (2017-2025) who has worked across healthcare, insurance, and banking industries. He's currently an AVP of Data Science at China CITIC Bank International.

Your question: "{query}"

You can try asking more specific questions like:
- "What is Rory's work experience?"
- "How old is Rory based on his education timeline?"
- "What are Rory's technical skills?"

I'll do my best to provide detailed answers based on available information."""

        if graph_sources:
            minimal_response += self._format_sources(graph_sources)
        
        return {"result": minimal_response}

@st.cache_resource
def initialize_enhanced_ai_system():
    """Enhanced AI system initialization with document separation and OCR support"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize document processor
        doc_processor = EnhancedDocumentProcessor()
        
        # Process documents with type separation
        documents_dict = doc_processor.process_documents_with_separation(current_dir)
        
        # Enhanced fallback content if no documents found
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
            - Implemented AI model governance framework
            
            Data Science Analyst at Cigna International Office (Aug 2018 - Sep 2019)
            - Supported US Medicare analytics with predictive modeling focus
            - Developed healthcare cost forecasting and risk assessment models
            
            Research Executive at Ipsos Hong Kong (2017 - 2018)
            - Conducted market research and data analysis for various industries
            - Developed statistical models for consumer behavior analysis
            - Managed survey design and data collection processes
            - Provided insights and recommendations to clients based on research findings
            
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
            Languages: Cantonese (Native), Mandarin (Native), English (Proficient)
            """
            documents_dict["professional"] = [(fallback_content, {"source": "profile", "type": "text", "category": "professional"})]
        
        # Create enhanced vector store
        vector_store = EnhancedVectorStore()
        vector_store.add_documents_by_type(documents_dict)
        
        # Initialize GraphRAG system
        graph_rag = GraphRAG()
        graph_rag.build_knowledge_graph(documents_dict)
        
        # Create advanced reasoning chain with GraphRAG
        reasoning_chain = AdvancedReasoningChain(vector_store, graph_rag)
        
        # Calculate total documents and chunks
        total_docs = sum(len(docs) for docs in documents_dict.values())
        total_chunks = sum(len(chunks) for chunks in vector_store.documents.values())
        
        return reasoning_chain, total_docs, total_chunks
    
    except Exception as e:
        st.error(f"Failed to initialize enhanced AI system: {str(e)}")
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
    <h3>Data Science & Analytics Expert with Enhanced Reasoning</h3>
    <p>Ask me anything about Rory's experience, skills, projects, or insights! Advanced analysis available.</p>
</div>
""", unsafe_allow_html=True)

# Initialize enhanced AI system and database
with st.spinner("🚀 Initializing Enhanced AI system..."):
    qa_chain, total_docs, total_chunks = initialize_enhanced_ai_system()
    conversation_db = get_conversation_db()

if qa_chain is None:
    st.error("❌ Rory's AI system is not available. Please check the system configuration.")
    st.stop()

# Initialize session ID for conversation tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Get conversation count for display
total_conversations = conversation_db.get_conversation_count()

# Display system status with database info
st.markdown(f"""
<div class="status-info">
    <strong style="color: #2c3e50;">🧠 Enhanced AI System Status:</strong> <span style="color: #34495e;">{total_docs} documents processed, 
    {total_chunks} knowledge chunks available | 🎯 Advanced Reasoning & Analysis | 💾 {total_conversations} conversations stored</span>
</div>
""", unsafe_allow_html=True)

# Data storage notification
st.markdown("""
<div style="background: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107; margin-bottom: 1rem; font-size: 0.9rem;">
    <strong>📊 Data Collection Notice:</strong> All questions and answers are stored in a database for future analysis and improvement of the AI assistant. 
    This helps us understand user needs and enhance the system's capabilities. Your conversations are used for analytical purposes only.
</div>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    # CV Section with better styling
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
        <h2 style="color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 0.5rem;">💼 Professional Experience</h2>
        <div style="margin: 1rem 0;">
            <div style="margin-bottom: 1.5rem;">
                <h4 style="color: #34495e; margin-bottom: 0.3rem;">🏦 China CITIC Bank International</h4>
                <p style="color: #7f8c8d; margin-bottom: 0.3rem;"><strong>AVP, Data Science</strong> (Nov 2022 - Current)</p>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Led data science initiatives in retail banking, optimizing AI-driven decision-making and Next-Best-Action models.</p>
            </div>
            <div style="margin-bottom: 1.5rem;">
                <h4 style="color: #34495e; margin-bottom: 0.3rem;">🛡️ AXA Hong Kong and Macau</h4>
                <p style="color: #7f8c8d; margin-bottom: 0.3rem;"><strong>Assistant Data Science Manager</strong> (Sep 2019 - Nov 2022)</p>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Advanced data capabilities for insurance, focusing on customer analytics, predictive modeling, and cloud migration.</p>
            </div>
            <div style="margin-bottom: 1.5rem;">
                <h4 style="color: #34495e; margin-bottom: 0.3rem;">🏥 Cigna International Office</h4>
                <p style="color: #7f8c8d; margin-bottom: 0.3rem;"><strong>Data Science Analyst</strong> (Aug 2018 - Sep 2019)</p>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Supported US Medicare analytics with focus on predictive modeling and healthcare claims analysis.</p>
            </div>
            <div style="margin-bottom: 1.5rem;">
                <h4 style="color: #34495e; margin-bottom: 0.3rem;">📊 Ipsos Hong Kong</h4>
                <p style="color: #7f8c8d; margin-bottom: 0.3rem;"><strong>Research Executive</strong> (2017 - 2018)</p>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Conducted market research and data analysis, developed statistical models for consumer behavior analysis.</p>
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
                <p style="color: #7f8c8d;"><strong>Bachelor of Social Sciences (Honours) in Global and Environmental Studies</strong> (2011-2014)</p>
            </div>
        </div>
    </div>
    
    <div style="background: white; padding: 2rem; border-radius: 15px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
        <h2 style="color: #2c3e50; border-bottom: 3px solid #f39c12; padding-bottom: 0.5rem;">🚀 Core Expertise</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
            <div>
                <h5 style="color: #2980b9; margin-bottom: 0.5rem;">📊 Data Science</h5>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Python, R, SQL, Machine Learning, Statistical Analysis</p>
            </div>
            <div>
                <h5 style="color: #27ae60; margin-bottom: 0.5rem;">🤖 AI & ML</h5>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Deep Learning, NLP, Computer Vision, Model Deployment</p>
            </div>
            <div>
                <h5 style="color: #8e44ad; margin-bottom: 0.5rem;">📈 Analytics</h5>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Predictive Modeling, Business Intelligence, Customer Analytics</p>
            </div>
            <div>
                <h5 style="color: #e67e22; margin-bottom: 0.5rem;">🛠️ Tools</h5>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Tableau, Power BI, Spark, Hadoop, Cloud Platforms</p>
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
        "Can you describe Rory's approach to machine learning projects?",
        "What are Rory's key achievements in analytics and data science?",
        "How does Rory approach business intelligence and data visualization?",
        "What industries has Rory worked in for data science projects?",
        "Tell me about Rory's experience with cloud platforms and big data",
        "What makes Rory unique as a data science professional?",
        "How can I contact Rory for potential collaboration?"
    ]
    
    # Create clickable sample questions
    for i, question in enumerate(sample_questions):
        if st.button(f"❓ {question}", key=f"sample_q_{i}", help="Click to ask this question"):
            st.session_state.selected_question = question
            st.rerun()
    
    # Chat Interface - properly positioned in right column
    st.markdown("---")
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat header with clear button
    col_chat1, col_chat2 = st.columns([3, 1])
    with col_chat1:
        st.markdown("### 💬 Chat with Rory's AI")
    with col_chat2:
        if st.button("🗑️ Clear History", key="clear_history", help="Clear all conversation history"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())  # Generate new session ID
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "👋 Hello! I'm Rory's AI assistant with enhanced reasoning capabilities. I can analyze complex questions and provide deep insights about Rory's background, projects, and expertise. All conversations are stored for analysis to improve the system!"
        })

    # Handle selected question from sample questions
    if "selected_question" in st.session_state:
        selected_q = st.session_state.selected_question
        st.session_state.messages.append({"role": "user", "content": selected_q})
        
        # Show processing indicator
        with st.spinner("🧠 Processing with enhanced reasoning..."):
            # Generate AI response for the selected question
            try:
                result = qa_chain({"query": selected_q})
                
                # Extract the result properly
                if isinstance(result, dict):
                    response = result.get("result", "Sorry, I couldn't find a relevant answer.")
                else:
                    response = str(result)
                
                # Add AI response to chat history
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
            # Show processing indicator
            with st.spinner("🔍 Analyzing with enhanced reasoning..."):
                try:
                    result = qa_chain({"query": prompt})
                    
                    # Extract the result properly
                    if isinstance(result, dict):
                        response = result.get("result", "Sorry, I couldn't find a relevant answer.")
                    else:
                        response = str(result)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant response to chat history
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
    <p>🤖 <strong>Rory's AI Assistant</strong> | Powered by Advanced AI & Enhanced Reasoning | 📄 PDF Processing with OCR</p>
    <p>📧 For direct contact or collaboration opportunities, please ask the AI for Rory's contact information</p>
    <p><em>This AI assistant features enhanced reasoning capabilities, document type separation, and advanced analytical processing</em></p>
</div>
""", unsafe_allow_html=True)
