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
    page_title="Rory's AI Assistant - Enhanced Reasoning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class EnhancedDocumentProcessor:
    """Enhanced document processor with document type separation"""
    
    def __init__(self):
        self.ocr_available = False
        # Try to import OCR dependencies
        try:
            import pytesseract
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
                    # Optimized chunking
                    chunks = self._split_text(text, chunk_size=300, overlap=30)
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
    """Advanced reasoning system with enhanced analytical capabilities"""
    
    def __init__(self, vector_store: EnhancedVectorStore):
        self.vector_store = vector_store
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
            "context_sources": ["professional"],
            "reasoning_steps": []
        }
        
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
        """Extract communication patterns and tone from mindset content without revealing personal details"""
        patterns = {
            "communication_style": "thoughtful and reflective",
            "analytical_approach": "systematic and thorough",
            "tone_characteristics": ["professional", "curious", "growth-oriented"],
            "thinking_patterns": ["analytical", "philosophical", "practical"]
        }
        
        # Analyze patterns without exposing personal content
        if "分析" in mindset_content or "思考" in mindset_content:
            patterns["analytical_approach"] = "deep analytical thinking"
        
        if "學習" in mindset_content or "成長" in mindset_content:
            patterns["tone_characteristics"].append("learning-focused")
        
        if "經驗" in mindset_content or "總結" in mindset_content:
            patterns["thinking_patterns"].append("experience-based reasoning")
        
        return patterns
    
    def _perform_advanced_reasoning(self, query: str, context: str, analysis: Dict, mindset_patterns: Dict) -> str:
        """Perform advanced reasoning based on query analysis"""
        reasoning_steps = []
        
        # Step 1: Information extraction
        reasoning_steps.append("**信息提取與分析**")
        reasoning_steps.append(f"基於Rory的專業履歷，我識別出以下關鍵信息：")
        
        # Step 2: Pattern recognition
        if analysis["requires_analysis"]:
            reasoning_steps.append("\n**模式識別與深度分析**")
            reasoning_steps.append("通過分析Rory的職業發展軌跡，我發現以下模式：")
            
            # Analyze career progression
            if "data science" in context.lower():
                reasoning_steps.append("• **技術演進路徑**: 從數據分析師到助理經理再到AVP，展現了技術深度與管理廣度的雙重發展")
                reasoning_steps.append("• **行業洞察能力**: 跨越保險、醫療、銀行三個行業，積累了豐富的業務理解")
                reasoning_steps.append("• **創新驅動思維**: 多次獲得創新獎項，說明具備前瞻性思維和實踐能力")
        
        # Step 3: Comparative analysis
        if analysis["requires_comparison"]:
            reasoning_steps.append("\n**比較分析與差異化優勢**")
            reasoning_steps.append("相比同行業專業人士，Rory的獨特優勢在於：")
            reasoning_steps.append("• **跨行業經驗**: 醫療保險→保險→銀行的完整金融服務鏈條經驗")
            reasoning_steps.append("• **技術與業務平衡**: 既有深度技術能力，又具備業務洞察和管理經驗")
            reasoning_steps.append("• **持續學習能力**: 雙碩士學位背景體現了持續學習和知識更新的能力")
        
        # Step 4: Strategic reasoning
        if "strategic" in analysis["reasoning_type"]:
            reasoning_steps.append("\n**戰略思維與方法論**")
            reasoning_steps.append("基於Rory的經驗，我推斷其解決問題的方法論包括：")
            reasoning_steps.append("• **數據驅動決策**: 利用機器學習和統計分析支撐業務決策")
            reasoning_steps.append("• **敏捷迭代優化**: 通過A/B測試和模型監控持續優化效果")
            reasoning_steps.append("• **跨部門協作**: 具備與業務、IT、合規等多部門協作的經驗")
        
        # Step 5: Predictive reasoning
        if analysis["requires_inference"]:
            reasoning_steps.append("\n**預測性推理與趨勢判斷**")
            reasoning_steps.append("基於Rory的技術發展軌跡，可以推斷：")
            reasoning_steps.append("• **AI/ML專精方向**: 從傳統統計分析向深度學習和生成式AI發展")
            reasoning_steps.append("• **業務價值創造**: 注重技術與業務結合，創造可衡量的商業價值")
            reasoning_steps.append("• **領導力發展**: 從技術專家向技術管理者的成功轉型")
        
        # Step 6: Synthesis and conclusion
        reasoning_steps.append("\n**綜合推理與結論**")
        reasoning_steps.append("綜合以上分析，我的推理結論是：")
        
        return "\n".join(reasoning_steps)
    
    def _generate_enhanced_response(self, query: str, professional_context: str, analysis: Dict, mindset_patterns: Dict) -> str:
        """Generate enhanced response with advanced reasoning"""
        
        # Perform advanced reasoning
        reasoning_analysis = self._perform_advanced_reasoning(query, professional_context, analysis, mindset_patterns)
        
        # Create sophisticated prompt
        prompt = f"""你是Rory的智能助手，具備高級推理能力。你需要基於專業信息進行深度分析和推理，而不僅僅是信息提取。

**查詢分析結果**:
- 推理類型: {analysis['reasoning_type']}
- 複雜度: {analysis['complexity']}
- 需要推理步驟: {', '.join(analysis['reasoning_steps'])}

**專業背景信息**:
{professional_context}

**用戶問題**: {query}

**推理分析過程**:
{reasoning_analysis}

**回答要求**:
1. 進行深度推理分析，不僅提取信息，更要分析背後的邏輯和模式
2. 展示推理過程，說明如何從已知信息得出結論
3. 提供洞察性觀點，而非簡單的事實陳述
4. 採用專業但易懂的語言風格
5. 如果涉及預測或建議，要基於邏輯推理給出依據
6. 保持客觀和專業，避免過度推測

請基於以上分析提供一個深度、有洞察力的回答："""

        return prompt
    
    def __call__(self, inputs: Dict) -> Dict:
        """Enhanced reasoning processing"""
        query = inputs.get("query", "").strip()
        
        # Analyze query complexity and reasoning requirements
        analysis = self._analyze_query_complexity(query)
        
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
            return {"result": self.reasoning_cache[cache_key]}
        
        # Generate enhanced prompt with reasoning
        prompt = self._generate_enhanced_response(query, professional_context, analysis, mindset_patterns)
        
        # Try API call
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
                    "max_tokens": 1500,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'output' in result and 'choices' in result['output']:
                    answer = result['output']['choices'][0]['message']['content']
                    self.reasoning_cache[cache_key] = answer
                    return {"result": answer}
                    
        except Exception as e:
            print(f"API Error: {str(e)}")
        
        # Enhanced fallback with reasoning
        if professional_context.strip():
            # Perform local reasoning analysis
            reasoning_analysis = self._perform_advanced_reasoning(query, professional_context, analysis, mindset_patterns)
            
            fallback_response = f"""**基於可用資料的深度分析**

{reasoning_analysis}

**專業背景摘要**:
{professional_context[:800]}...

**分析結論**:
基於以上推理分析，Rory在數據科學領域展現出了從技術專家到業務領導者的完整發展軌跡。其獨特價值在於跨行業經驗、技術與業務的平衡能力，以及持續創新的思維模式。

*此分析基於專業履歷信息，通過邏輯推理和模式識別得出。如需更具體的信息，歡迎提出更詳細的問題。*"""
        else:
            fallback_response = """**歡迎使用Rory的增強推理助手**

我具備高級推理分析能力，可以：

🧠 **深度推理**: 不僅提取信息，更分析背後的邏輯和模式
📊 **比較分析**: 識別優勢、差異和發展趨勢  
🔮 **預測推理**: 基於現有信息進行合理推測
💡 **戰略思維**: 提供方法論和解決方案建議

**試試這些深度問題**:
• "分析Rory的職業發展模式和成功因素"
• "比較Rory在不同行業的經驗，有什麼獨特優勢？"
• "基於Rory的背景，預測他在AI領域的發展方向"
• "Rory解決複雜數據科學問題的方法論是什麼？"

*我會進行深度分析和推理，而不僅僅是信息檢索。*"""
        
        self.reasoning_cache[cache_key] = fallback_response
        return {"result": fallback_response}

@st.cache_resource
def initialize_enhanced_reasoning_system():
    """Initialize enhanced reasoning system with document separation"""
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
            documents_dict["professional"] = [(fallback_content, {"source": "profile", "type": "text", "category": "professional"})]
        
        # Create enhanced vector store
        vector_store = EnhancedVectorStore()
        vector_store.add_documents_by_type(documents_dict)
        
        # Create advanced reasoning chain
        reasoning_chain = AdvancedReasoningChain(vector_store)
        
        # Calculate total documents and chunks
        total_docs = sum(len(docs) for docs in documents_dict.values())
        total_chunks = sum(len(chunks) for chunks in vector_store.documents.values())
        
        return reasoning_chain, total_docs, total_chunks, documents_dict
    
    except Exception as e:
        st.error(f"Failed to initialize enhanced reasoning system: {str(e)}")
        return None, 0, 0, {}

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
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .reasoning-badge {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
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
        border-color: #667eea;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
    }
    
    .stChatMessage {
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background-color: #667eea !important;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background-color: #764ba2 !important;
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
    <h1>🧠 Rory's Enhanced Reasoning Assistant</h1>
    <h3>Advanced AI with Deep Analytical Capabilities</h3>
    <p>Ask complex questions and get sophisticated reasoning-based answers!</p>
</div>
""", unsafe_allow_html=True)

# Initialize enhanced reasoning system
with st.spinner("🚀 Initializing Enhanced Reasoning System..."):
    reasoning_chain, total_docs, total_chunks, documents_dict = initialize_enhanced_reasoning_system()

if reasoning_chain is None:
    st.error("❌ Enhanced reasoning system is not available. Please check the system configuration.")
    st.stop()

# Display system status with document breakdown
professional_count = len(documents_dict.get("professional", []))
mindset_count = len(documents_dict.get("mindset", []))
general_count = len(documents_dict.get("general", []))

st.markdown(f"""
<div class="status-info">
    <strong>🧠 Enhanced Reasoning System Status:</strong><br>
    📄 <strong>Documents Processed:</strong> {total_docs} total ({professional_count} professional, {mindset_count} mindset, {general_count} general)<br>
    🧩 <strong>Knowledge Chunks:</strong> {total_chunks} available for reasoning<br>
    <span class="reasoning-badge">🔍 Advanced Reasoning Enabled</span>
    <span class="reasoning-badge">📊 Document Type Separation</span>
    <span class="reasoning-badge">🎯 Professional Focus</span>
</div>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    # Enhanced CV Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h1 style="color: white; margin-bottom: 0.5rem;">👨‍💼 Rory Chen</h1>
            <h3 style="color: #e8f4fd; margin-bottom: 0.5rem;">Data Science & Analytics Expert</h3>
            <p style="color: #b8d4f0; font-size: 1.1rem;">🏢 China CITIC Bank International Limited</p>
            <p style="color: #b8d4f0;">📍 Hong Kong SAR | 🧠 Enhanced Reasoning AI</p>
        </div>
    </div>
    
    <div style="background: white; padding: 2rem; border-radius: 15px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 2rem;">
        <h2 style="color: #2c3e50; border-bottom: 3px solid #667eea; padding-bottom: 0.5rem;">🧠 Reasoning Capabilities</h2>
        <div style="margin: 1rem 0;">
            <div style="margin-bottom: 1rem;">
                <h4 style="color: #667eea; margin-bottom: 0.3rem;">🔍 Deep Analysis</h4>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Analyzes patterns, identifies trends, and provides insights beyond surface-level information</p>
            </div>
            <div style="margin-bottom: 1rem;">
                <h4 style="color: #764ba2; margin-bottom: 0.3rem;">📊 Comparative Reasoning</h4>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Compares experiences across industries, identifies unique advantages and differentiators</p>
            </div>
            <div style="margin-bottom: 1rem;">
                <h4 style="color: #ff6b6b; margin-bottom: 0.3rem;">🔮 Predictive Insights</h4>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Makes informed predictions about career trends and technology directions</p>
            </div>
            <div>
                <h4 style="color: #feca57; margin-bottom: 0.3rem;">💡 Strategic Thinking</h4>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Provides methodological approaches and strategic frameworks for problem-solving</p>
            </div>
        </div>
    </div>
    
    <div style="background: white; padding: 2rem; border-radius: 15px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
        <h2 style="color: #2c3e50; border-bottom: 3px solid #f39c12; padding-bottom: 0.5rem;">📋 Document Sources</h2>
        <div style="margin: 1rem 0;">
            <div style="margin-bottom: 1rem;">
                <h4 style="color: #27ae60; margin-bottom: 0.3rem;">📄 Professional Documents ({professional_count})</h4>
                <p style="color: #7f8c8d; font-size: 0.9rem;">CV, work experience, and career-related information for factual responses</p>
            </div>
            <div style="margin-bottom: 1rem;">
                <h4 style="color: #8e44ad; margin-bottom: 0.3rem;">🧠 Mindset Analysis ({mindset_count})</h4>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Communication patterns and thinking style (personal details protected)</p>
            </div>
            <div>
                <h4 style="color: #e67e22; margin-bottom: 0.3rem;">📚 General Knowledge ({general_count})</h4>
                <p style="color: #7f8c8d; font-size: 0.9rem;">Additional context and supporting information</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Enhanced Q&A Section
    st.markdown("### 🧠 Ask the Enhanced Reasoning Assistant")
    st.markdown("Ask complex questions that require deep analysis and reasoning:")
    
    # Direct Q&A input box
    user_question = st.text_input(
        "Your Complex Question:",
        placeholder="Ask analytical questions like 'Why is Rory successful?' or 'How does his approach differ?'",
        key="reasoning_question_input"
    )
    
    if st.button("🧠 Analyze & Reason", key="ask_reasoning_question") and user_question.strip():
        st.session_state.selected_question = user_question.strip()
        st.rerun()
    
    st.markdown("---")
    
    # Enhanced Sample Questions
    st.markdown("### 💡 Complex Reasoning Questions - Click to Explore!")
    
    reasoning_questions = [
        "分析Rory的職業發展模式和成功因素是什麼？",
        "比較Rory在不同行業的經驗，他有什麼獨特優勢？",
        "基於Rory的背景，預測他在AI領域的發展方向",
        "Rory解決複雜數據科學問題的方法論是什麼？",
        "為什麼Rory能夠在技術和管理之間取得平衡？",
        "分析Rory的跨行業經驗如何影響他的思維方式",
        "Rory的創新能力體現在哪些方面？請深度分析",
        "如何評估Rory在數據科學領域的競爭優勢？",
        "預測Rory未來5年的職業發展軌跡和可能性"
    ]
    
    # Create clickable reasoning questions
    for i, question in enumerate(reasoning_questions):
        if st.button(f"🧠 {question}", key=f"reasoning_q_{i}", help="Click for deep analysis"):
            st.session_state.selected_question = question
            st.rerun()
    
    # Enhanced Chat Interface
    st.markdown("---")
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("### 💬 Reasoning Chat Interface")
    
    # Initialize chat history
    if "reasoning_messages" not in st.session_state:
        st.session_state.reasoning_messages = []
        # Add enhanced welcome message
        st.session_state.reasoning_messages.append({
            "role": "assistant", 
            "content": """🧠 **歡迎使用Rory的增強推理助手！**

我具備高級分析能力，可以：
• **深度推理**: 分析背後的邏輯和模式
• **比較分析**: 識別優勢和差異化因素  
• **預測推理**: 基於現有信息進行合理推測
• **戰略思維**: 提供方法論和解決方案

我會基於Rory的專業履歷進行分析，同時學習他的思維模式來提供更有洞察力的回答。請問一些需要深度分析的複雜問題！"""
        })

    # Handle selected question from sample questions
    if "selected_question" in st.session_state:
        selected_q = st.session_state.selected_question
        st.session_state.reasoning_messages.append({"role": "user", "content": selected_q})
        
        # Show enhanced processing indicator
        with st.spinner("🧠 進行深度推理分析中..."):
            try:
                result = reasoning_chain({"query": selected_q})
                
                # Extract the result properly
                if isinstance(result, dict):
                    response = result.get("result", "抱歉，我無法找到相關的分析結果。")
                else:
                    response = str(result)
                
                # Add AI response to chat history
                st.session_state.reasoning_messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"抱歉，在處理您的問題時遇到了錯誤：{str(e)}"
                st.session_state.reasoning_messages.append({"role": "assistant", "content": error_msg})
        
        del st.session_state.selected_question
        st.rerun()

    # Display chat messages
    for message in st.session_state.reasoning_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Enhanced chat input
    if prompt := st.chat_input("問一個需要深度分析的複雜問題..."):
        # Add user message to chat history
        st.session_state.reasoning_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate enhanced reasoning response
        with st.chat_message("assistant"):
            # Show enhanced processing indicator
            with st.spinner("🧠 執行高級推理分析..."):
                try:
                    result = reasoning_chain({"query": prompt})
                    
                    # Extract the result properly
                    if isinstance(result, dict):
                        response = result.get("result", "抱歉，我無法找到相關的分析結果。")
                    else:
                        response = str(result)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.reasoning_messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"抱歉，在處理您的問題時遇到了錯誤：{str(e)}"
                    st.error(error_msg)
                    st.session_state.reasoning_messages.append({"role": "assistant", "content": error_msg})

    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🧠 <strong>Rory's Enhanced Reasoning Assistant</strong> | Powered by Advanced AI & Deep Learning</p>
    <p>🔍 <strong>Features:</strong> Document Type Separation | Advanced Reasoning | Pattern Analysis | Predictive Insights</p>
    <p>📧 For professional inquiries, please ask the AI for Rory's contact information</p>
    <p><em>This AI assistant uses sophisticated reasoning algorithms to provide deep analytical insights based on professional documentation</em></p>
</div>
""", unsafe_allow_html=True)
