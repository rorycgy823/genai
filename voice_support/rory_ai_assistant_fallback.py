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

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Rory's AI Assistant - Data Science & Analytics Expert",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class EnhancedDocumentProcessor:
    """Enhanced document processor with PDF support (OCR optional)"""
    
    def __init__(self):
        self.ocr_available = False
        # Try to import OCR dependencies
        try:
            import pytesseract
            # Try to get tesseract version to verify it's working
            pytesseract.get_tesseract_version()
            self.ocr_available = True
            st.success("✅ OCR capabilities available - PDF images can be processed")
        except Exception as e:
            st.warning(f"⚠️ OCR not available: {str(e)}. PDF text extraction will use standard methods only.")
    
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
    
    def process_documents(self, directory: str) -> List[Tuple[str, Dict]]:
        """Process all documents in directory with PDF support"""
        documents = []
        
        # Process text files
        text_extensions = ['*.txt', '*.md']
        for ext in text_extensions:
            files = glob.glob(os.path.join(directory, ext))
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    if text.strip() and len(text) > 100:
                        documents.append((text, {
                            "source": os.path.basename(file_path), 
                            "type": "text",
                            "path": file_path
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
                    documents.append((text, {
                        "source": os.path.basename(pdf_path), 
                        "type": "pdf",
                        "path": pdf_path
                    }))
                    st.success(f"Successfully processed {os.path.basename(pdf_path)}")
                else:
                    st.warning(f"No readable text found in {os.path.basename(pdf_path)}")
            except Exception as e:
                st.error(f"Failed to process PDF {pdf_path}: {str(e)}")
        
        return documents

class OptimizedVectorStore:
    """Optimized in-memory vector store with faster initialization"""
    
    def __init__(self):
        # Use a smaller, faster model for better performance
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_documents(self, texts: List[str], metadatas: List[Dict]):
        """Add documents to the vector store with optimized chunking"""
        for text, metadata in zip(texts, metadatas):
            if text.strip():
                # Optimized chunking - smaller chunks for faster processing
                chunks = self._split_text(text, chunk_size=300, overlap=30)
                for i, chunk in enumerate(chunks):
                    if chunk.strip() and len(chunk) > 50:  # Filter very short chunks
                        self.documents.append(chunk)
                        chunk_metadata = metadata.copy()
                        chunk_metadata['chunk_index'] = i
                        self.metadata.append(chunk_metadata)
        
        # Generate embeddings for all documents at once (more efficient)
        if self.documents:
            self.embeddings = self.model.encode(self.documents, batch_size=32, show_progress_bar=False)
    
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
    
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, Dict]]:
        """Optimized similarity search with fewer results for speed"""
        if not self.documents:
            return []
        
        query_embedding = self.model.encode([query], show_progress_bar=False)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k results with higher threshold for better quality
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.15:  # Higher threshold for better results
                results.append((self.documents[idx], self.metadata[idx]))
        
        return results

class IntelligentGraphRAGChain:
    """Advanced GraphRAG implementation with human-level analysis and reasoning"""
    
    def __init__(self, vector_store: OptimizedVectorStore):
        self.vector_store = vector_store
        self.api_key = "sk-015ea57c8b254c4181d30b2de4259d8b"
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        # Cache for complex analysis to avoid repeated processing
        self.analysis_cache = {}
    
    def _analyze_query_intent(self, query: str) -> Dict:
        """Analyze query intent for sophisticated response generation"""
        query_lower = query.lower()
        
        intent_analysis = {
            "type": "general",
            "depth": "surface",
            "context_needed": [],
            "analysis_required": False
        }
        
        # Determine query complexity and intent
        if any(word in query_lower for word in ['why', 'how', 'analyze', 'compare', 'evaluate', 'assess']):
            intent_analysis["type"] = "analytical"
            intent_analysis["depth"] = "deep"
            intent_analysis["analysis_required"] = True
        
        if any(word in query_lower for word in ['approach', 'methodology', 'strategy', 'framework']):
            intent_analysis["type"] = "methodological"
            intent_analysis["context_needed"] = ["experience", "projects", "technical_skills"]
        
        if any(word in query_lower for word in ['career', 'journey', 'progression', 'growth']):
            intent_analysis["type"] = "narrative"
            intent_analysis["context_needed"] = ["experience", "education", "achievements"]
        
        if any(word in query_lower for word in ['unique', 'different', 'standout', 'special']):
            intent_analysis["type"] = "comparative"
            intent_analysis["analysis_required"] = True
        
        return intent_analysis
    
    def _extract_factual_information(self, query: str, context: str) -> str:
        """Extract and reason from factual information in the context"""
        query_lower = query.lower()
        
        # Age-related reasoning
        if any(word in query_lower for word in ['age', 'old', 'born', 'birth']):
            return self._reason_about_age(context)
        
        # Education-related queries
        if any(word in query_lower for word in ['education', 'degree', 'university', 'study', 'graduate']):
            return self._extract_education_info(context)
        
        # Experience duration queries
        if any(word in query_lower for word in ['experience', 'years', 'how long', 'duration']):
            return self._calculate_experience_duration(context)
        
        # Contact information queries
        if any(word in query_lower for word in ['contact', 'email', 'phone', 'reach']):
            return self._extract_contact_info(context)
        
        return None
    
    def _reason_about_age(self, context: str) -> str:
        """Reason about Rory's age based on available information"""
        reasoning_parts = []
        estimated_age = None
        
        # Look for graduation years
        if "2011-2014" in context:
            reasoning_parts.append("• Bachelor's degree: 2011-2014 (graduated at ~22-23 years old)")
            estimated_age = 2025 - 2014 + 23  # Current year - graduation year + typical graduation age
        
        if "2014-2015" in context:
            reasoning_parts.append("• First Master's degree: 2014-2015")
        
        if "2016-2017" in context:
            reasoning_parts.append("• Second Master's degree: 2016-2017")
        
        # Look for work experience start
        if "Aug 2018" in context:
            reasoning_parts.append("• Started professional career: August 2018")
            if not estimated_age:
                estimated_age = 2025 - 2018 + 25  # Assuming started work at ~25
        
        if reasoning_parts and estimated_age:
            return f"""**Age Estimation Based on Available Information**

Based on Rory's educational and career timeline, I can estimate his age:

**Reasoning:**
{chr(10).join(reasoning_parts)}

**Age Estimation:**
If Rory graduated with his Bachelor's degree in 2014 at the typical age of 22-23, he would be approximately **{estimated_age} years old** as of 2025.

*This is an estimation based on typical graduation ages and career progression patterns from the available information.*"""
        
        return "I don't have direct age information, but I can see Rory's educational timeline (Bachelor's 2011-2014, Master's 2014-2015 and 2016-2017) and career start (2018). Would you like me to estimate his age based on typical graduation patterns?"
    
    def _extract_education_info(self, context: str) -> str:
        """Extract education information from context"""
        education_info = []
        
        if "Master of Arts in Quantitative Analysis" in context:
            education_info.append("🎓 **Master of Arts in Quantitative Analysis for Business** - City University of Hong Kong (2016-2017)")
        
        if "Master of Arts in Public Policy" in context:
            education_info.append("🎓 **Master of Arts in Public Policy and Management** - City University of Hong Kong (2014-2015)")
        
        if "Bachelor of Social Sciences" in context:
            education_info.append("🎓 **Bachelor of Social Sciences (Honours) in Global and Environmental Studies** - Education University of Hong Kong (2011-2014)")
        
        if education_info:
            return f"""**Rory's Educational Background**

{chr(10).join(education_info)}

**Analysis:** Rory has a strong quantitative and policy background with dual Master's degrees, providing both analytical skills and business/policy understanding."""
        
        return "I can see educational information in the context, but need more specific details to provide a comprehensive answer."
    
    def _calculate_experience_duration(self, context: str) -> str:
        """Calculate work experience duration"""
        total_years = 0
        experience_breakdown = []
        
        # Current role
        if "Nov 2022 - Current" in context or "Nov 2022 - Present" in context:
            current_duration = 2025 - 2022 + (1-11)/12  # From Nov 2022 to 2025
            total_years += current_duration
            experience_breakdown.append(f"🏦 **China CITIC Bank International** (Nov 2022 - Present): ~{current_duration:.1f} years")
        
        # AXA experience
        if "Sep 2019 - Nov 2022" in context:
            axa_duration = 2022 - 2019 + (11-9)/12  # Sep 2019 to Nov 2022
            total_years += axa_duration
            experience_breakdown.append(f"🛡️ **AXA Hong Kong** (Sep 2019 - Nov 2022): ~{axa_duration:.1f} years")
        
        # Cigna experience
        if "Aug 2018 - Sep 2019" in context:
            cigna_duration = 2019 - 2018 + (9-8)/12  # Aug 2018 to Sep 2019
            total_years += cigna_duration
            experience_breakdown.append(f"🏥 **Cigna International** (Aug 2018 - Sep 2019): ~{cigna_duration:.1f} years")
        
        if experience_breakdown:
            return f"""**Rory's Professional Experience Duration**

{chr(10).join(experience_breakdown)}

**Total Professional Experience:** Approximately **{total_years:.1f} years** in data science and analytics (as of 2025)

**Career Progression:** Analyst → Manager → AVP over {total_years:.1f} years, showing consistent growth and advancement."""
        
        return "I can see work experience information but need more specific details to calculate exact durations."
    
    def _extract_contact_info(self, context: str) -> str:
        """Extract contact information"""
        contact_parts = []
        
        if "chengy823@gmail.com" in context:
            contact_parts.append("📧 **Email:** chengy823@gmail.com")
        
        if "Hong Kong" in context:
            contact_parts.append("📍 **Location:** Hong Kong SAR")
        
        if "Cantonese" in context and "Mandarin" in context and "English" in context:
            contact_parts.append("🗣️ **Languages:** Cantonese (Native), Mandarin (Native), English (Proficient)")
        
        if contact_parts:
            return f"""**Contact Information for Rory Chen**

{chr(10).join(contact_parts)}

*Available for data science consulting, collaboration, and professional opportunities.*"""
        
        return "I can help you find Rory's contact information. Let me search through the available materials."
    
    def _generate_intelligent_response(self, query: str, context: str, intent: Dict) -> str:
        """Generate sophisticated AI response with human-level analysis"""
        
        # Create advanced prompt based on intent
        prompt = f"""你的角色是Rory的智能回答助手，目的是学习并使用Rory的语气语调语言风格帮助向其他人更好的介绍Rory的履历，以及分享他的人生感悟，但请不要分享他的私人隐私除了联系方式，如果用户提问关于Rory的工作以及履历，你必须要基于以下上下文信息回答问题。如果用户提问内容与Rory的工作经验履历不相关，上下文中没有相关信息，请说明无法从提供的文档中找到答案。

CONTEXT INFORMATION:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Provide comprehensive information beyond basic facts
2. Add analytical insights and implications
3. Connect different aspects of Rory's experience
4. Use conversational, engaging tone
5. Anticipate follow-up questions
6. Provide actionable insights where relevant

Provide a thoughtful, comprehensive response:"""

        return prompt
    
    def __call__(self, inputs: Dict) -> Dict:
        """Advanced GraphRAG processing with intelligent analysis"""
        query = inputs.get("query", "").strip()
        
        # Analyze query intent for sophisticated response
        intent = self._analyze_query_intent(query)
        
        # Get relevant documents with expanded search for complex queries
        k = 5 if intent["analysis_required"] else 3
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        
        # Build comprehensive context
        context = "\n\n".join([doc[0] for doc in relevant_docs])
        
        # First try to extract factual information
        factual_response = self._extract_factual_information(query, context)
        if factual_response:
            return {"result": factual_response}
        
        # Check cache for complex analysis
        cache_key = hash(query.lower().strip() + str(intent) + context[:300])
        if cache_key in self.analysis_cache:
            return {"result": self.analysis_cache[cache_key]}
        
        # Generate intelligent prompt
        prompt = self._generate_intelligent_response(query, context, intent)
        
        # Try API call with parameters optimized for quality
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": "qwen-plus",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'output' in result and 'choices' in result['output']:
                    answer = result['output']['choices'][0]['message']['content']
                    # Cache the intelligent response
                    self.analysis_cache[cache_key] = answer
                    return {"result": answer}
                    
        except Exception as e:
            # Log error for debugging but continue with fallback
            print(f"API Error: {str(e)}")
        
        # Intelligent fallback with sophisticated analysis
        if context.strip():
            fallback_response = f"""**Information Based on Available Materials**

From Rory's documented background:

{context[:500]}...

**Key Points:**
• Professional experience spans healthcare, insurance, and banking industries
• Technical expertise in Python, R, SQL, and modern ML/AI technologies  
• Leadership progression from analyst to AVP level
• Proven business impact with measurable results

*For specific questions about age, education timeline, or other details, I can reason from the available information to provide estimates.*"""
        else:
            fallback_response = """**Welcome to Rory's AI Assistant**

I can help answer questions about Rory Chen's background by reasoning from available materials. I can:

🔍 **Extract Facts**: Direct information from documents
🧮 **Make Calculations**: Age estimates, experience duration, etc.
📊 **Provide Analysis**: Strategic insights based on documented experience
💡 **Reason Logically**: Connect information to answer specific questions

**Try asking specific questions like:**
• "How old is Rory?" (I'll estimate from education/career timeline)
• "How many years of experience does Rory have?"
• "What's Rory's educational background?"

*I base all responses on available materials and clearly show my reasoning.*"""
        
        self.analysis_cache[cache_key] = fallback_response
        return {"result": fallback_response}

@st.cache_resource
def initialize_enhanced_ai_system():
    """Enhanced AI system initialization with PDF support"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize document processor
        doc_processor = EnhancedDocumentProcessor()
        
        # Process all documents including PDFs
        documents = doc_processor.process_documents(current_dir)
        
        # Enhanced fallback content if no documents found
        if not documents:
            fallback_content = """
            Rory Chen - Data Science & Analytics Expert
            
            Professional Experience:
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
            documents = [(fallback_content, {"source": "profile", "type": "text"})]
        
        # Create optimized vector store
        vector_store = OptimizedVectorStore()
        texts = [doc[0] for doc in documents]
        metadatas = [doc[1] for doc in documents]
        vector_store.add_documents(texts, metadatas)
        
        # Create intelligent GraphRAG chain
        qa_chain = IntelligentGraphRAGChain(vector_store)
        
        return qa_chain, len(documents), len(vector_store.documents)
    
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
    <h3>Data Science & Analytics Expert with PDF Support</h3>
    <p>Ask me anything about Rory's experience, skills, projects, or insights! PDF processing available.</p>
</div>
""", unsafe_allow_html=True)

# Initialize enhanced AI system
with st.spinner("🚀 Initializing Enhanced AI system..."):
    qa_chain, total_docs, total_chunks = initialize_enhanced_ai_system()

if qa_chain is None:
    st.error("❌ Rory's AI system is not available. Please check the system configuration.")
    st.stop()

# Display system status
st.markdown(f"""
<div class="status-info">
    <strong style="color: #2c3e50;">🧠 Enhanced AI System Status:</strong> <span style="color: #34495e;">{total_docs} documents processed, 
    {total_chunks} knowledge chunks available | 📄 PDF Processing Enabled | 🎯 Advanced Analysis & Human-Level Reasoning</span>
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
    st.markdown("### 💬 Chat with Rory's AI")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "👋 Hello! I'm Rory's AI assistant with PDF processing capabilities. I can extract text from PDF documents and answer questions about Rory's background, projects, and expertise. Feel free to ask me anything!"
        })

    # Handle selected question from sample questions
    if "selected_question" in st.session_state:
        selected_q = st.session_state.selected_question
        st.session_state.messages.append({"role": "user", "content": selected_q})
        
        # Show processing indicator
        with st.spinner("🧠 Processing your question..."):
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
            with st.spinner("🔍 Analyzing your question..."):
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
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error while processing your question: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🤖 <strong>Rory's AI Assistant</strong> | Powered by Advanced AI & Machine Learning | 📄 PDF Processing Enabled</p>
    <p>📧 For direct contact or collaboration opportunities, please ask the AI for Rory's contact information</p>
    <p><em>This AI assistant is trained on Rory's professional experience and knowledge base with PDF document support</em></p>
</div>
""", unsafe_allow_html=True)
