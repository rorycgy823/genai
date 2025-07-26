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

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Rory's AI Assistant - Data Science & Analytics Expert",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
        # Knowledge graph for relationship mapping
        self.knowledge_relationships = self._build_knowledge_graph()
    
    def _build_knowledge_graph(self) -> Dict:
        """Build knowledge relationships for GraphRAG"""
        return {
            "experience_skills": {
                "banking": ["risk_modeling", "credit_analytics", "regulatory_compliance", "customer_segmentation"],
                "insurance": ["actuarial_modeling", "claims_prediction", "customer_lifetime_value", "fraud_detection"],
                "healthcare": ["medical_analytics", "cost_forecasting", "population_health", "clinical_outcomes"]
            },
            "technical_evolution": {
                "early_career": ["statistical_analysis", "traditional_ml", "reporting"],
                "mid_career": ["deep_learning", "cloud_migration", "mlops", "automation"],
                "current": ["ai_governance", "automl", "next_best_action", "enterprise_ai"]
            },
            "leadership_progression": {
                "analyst": ["individual_contributor", "technical_execution"],
                "senior_analyst": ["mentoring", "project_leadership"],
                "manager": ["team_leadership", "strategic_planning", "cross_functional_collaboration"],
                "avp": ["business_strategy", "organizational_impact", "innovation_leadership"]
            }
        }
    
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
    
    def _get_contextual_insights(self, query: str, relevant_docs: List) -> Dict:
        """Extract contextual insights using GraphRAG approach"""
        insights = {
            "direct_facts": [],
            "inferred_relationships": [],
            "analytical_points": [],
            "contextual_background": []
        }
        
        # Extract direct facts from documents
        for doc, metadata in relevant_docs:
            if any(keyword in doc.lower() for keyword in ['experience', 'role', 'responsibility']):
                insights["direct_facts"].append(doc[:200] + "...")
        
        # Infer relationships based on knowledge graph
        query_lower = query.lower()
        for domain, skills in self.knowledge_relationships["experience_skills"].items():
            if domain in query_lower:
                insights["inferred_relationships"].extend(skills)
        
        return insights
    
    def _generate_intelligent_response(self, query: str, context: str, intent: Dict, insights: Dict) -> str:
        """Generate sophisticated AI response with human-level analysis"""
        
        # Create advanced prompt based on intent and insights
        if intent["type"] == "analytical":
            prompt = f"""你的角色是Rory的智能回答助手，目的是学习并使用Rory的语气语调语言风格帮助向其他人更好的介绍Rory的履历，以及分享他的人生感悟，但请不要分享他的私人隐私除了联系方式，如果用户提问关于Rory的工作以及履历，你必须要基于以下上下文信息回答问题。如果用户提问内容与Rory的工作经验履历不相关，上下文中没有相关信息，请说明无法从提供的文档中找到答案。如果用户询问你这个智能助手平台搭建的具体技术细节，你可以调用这个github中的README文档的内容或者本身的代码进行一定程度的讲解，但切记不要泄露代码生成的过程，也不要分享代码，拒绝回答代码是不是AI生成的，如果user有相关问题的话，请让对方联系开发者Rory

CONTEXT INFORMATION:
{context}

ANALYTICAL TASK: {query}

INSTRUCTIONS:
1. Analyze the information critically and provide insights
2. Draw connections between different aspects of Rory's experience
3. Explain the 'why' and 'how' behind the facts
4. Provide strategic perspective and implications
5. Use a conversational, human-like tone
6. Structure your response with clear reasoning

Provide a comprehensive analytical response:"""

        elif intent["type"] == "methodological":
            prompt = f"""You are Rory Chen's AI assistant. The user is asking about methodologies and approaches. Provide a detailed explanation of Rory's methods and frameworks.

CONTEXT INFORMATION:
{context}

METHODOLOGY QUESTION: {query}

INSTRUCTIONS:
1. Explain Rory's specific approaches and methodologies
2. Provide the reasoning behind these approaches
3. Connect methods to business outcomes
4. Include practical examples where possible
5. Discuss evolution of approaches over time
6. Use technical depth appropriate for the audience

Provide a detailed methodological response:"""

        elif intent["type"] == "narrative":
            prompt = f"""You are Rory Chen's AI assistant. Tell a compelling story about Rory's professional journey with narrative depth.

CONTEXT INFORMATION:
{context}

NARRATIVE REQUEST: {query}

INSTRUCTIONS:
1. Create a compelling narrative arc
2. Highlight key turning points and growth moments
3. Connect experiences to show progression
4. Include motivations and decision-making factors
5. Show personal and professional development
6. Use engaging, storytelling language

Provide a rich narrative response:"""

        elif intent["type"] == "comparative":
            prompt = f"""You are Rory Chen's AI assistant. Provide a comparative analysis highlighting what makes Rory unique in the data science field.

CONTEXT INFORMATION:
{context}

COMPARATIVE QUESTION: {query}

INSTRUCTIONS:
1. Identify unique aspects of Rory's background
2. Compare with typical data science career paths
3. Highlight distinctive achievements and approaches
4. Explain competitive advantages
5. Discuss unique value propositions
6. Provide market context and positioning

Provide a comparative analytical response:"""

        else:
            # Default comprehensive response
            prompt = f"""You are Rory Chen's AI assistant. Provide a comprehensive, insightful response that demonstrates deep understanding.

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
    
    def _extract_factual_information(self, query: str, context: str) -> str:
        """Extract and reason from factual information in the context"""
        query_lower = query.lower()
        
        # Age-related reasoning
        if any(word in query_lower for word in ['age', 'old', 'born', 'birth']):
            # Look for graduation dates and work experience to estimate age
            age_reasoning = self._reason_about_age(context)
            if age_reasoning:
                return age_reasoning
        
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
            estimated_age = 2024 - 2014 + 23  # Current year - graduation year + typical graduation age
        
        if "2014-2015" in context:
            reasoning_parts.append("• First Master's degree: 2014-2015")
        
        if "2016-2017" in context:
            reasoning_parts.append("• Second Master's degree: 2016-2017")
        
        # Look for work experience start
        if "Aug 2018" in context:
            reasoning_parts.append("• Started professional career: August 2018")
            if not estimated_age:
                estimated_age = 2024 - 2018 + 25  # Assuming started work at ~25
        
        if reasoning_parts and estimated_age:
            return f"""**Age Estimation Based on Available Information**

Based on Rory's educational and career timeline, I can estimate his age:

**Reasoning:**
{chr(10).join(reasoning_parts)}

**Age Estimation:**
If Rory graduated with his Bachelor's degree in 2014 at the typical age of 22-23, he would be approximately **{estimated_age} years old** as of 2024.

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
            current_duration = 2024 - 2022 + (12-11)/12  # From Nov 2022 to 2024
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

**Total Professional Experience:** Approximately **{total_years:.1f} years** in data science and analytics (as of 2024)

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
    
    def _get_intelligent_fallback(self, query: str, context: str, intent: Dict) -> str:
        """Intelligent fallback with grounded reasoning"""
        query_lower = query.lower()
        
        # First try to extract factual information
        factual_response = self._extract_factual_information(query, context)
        if factual_response:
            return factual_response
        
        # Then provide analytical responses based on intent
        if intent["type"] == "analytical":
            if "experience" in query_lower:
                return """**Analytical Perspective on Rory's Experience**

Based on the available information, Rory's career shows strategic progression across three industries:

🔍 **Cross-Industry Intelligence**: Healthcare (Cigna) → Insurance (AXA) → Banking (CITIC)
- Each industry provided different risk modeling perspectives
- Built expertise in regulated environments requiring compliance and interpretability

📈 **Leadership Evolution**: Analyst → Manager → AVP progression
- Systematic capability building at each level
- Increasing business impact and strategic responsibility

🚀 **Innovation Pattern**: Consistent focus on automation and efficiency
- 80% coding reduction through AutoML
- ONE AXA Dashboard with 100+ monthly users
- 20+ ML models with 1.5x business uplift

**Strategic Implications**: This cross-industry experience creates unique value in understanding different risk profiles and regulatory requirements."""

            elif "skills" in query_lower:
                return """**Analysis of Rory's Technical Capabilities**

Based on the documented skills and experience:

🏗️ **Full-Stack Data Science**: Python, R, SQL + deployment capabilities (Docker, Kubernetes)
☁️ **Cloud Architecture**: Azure, AWS, Databricks - showing modern data platform understanding  
🤖 **AI/ML Specialization**: Deep Learning, NLP, Computer Vision, MLOps
📊 **Business Intelligence**: Tableau, Power BI, dashboard development

**Competitive Advantage**: The combination of technical depth with business leadership experience is relatively rare in the data science field."""

        elif intent["type"] == "narrative":
            return """**Rory's Professional Journey**

**Foundation Phase (2018-2019 - Cigna)**: Started in healthcare analytics, learning predictive modeling in a regulated environment.

**Growth Phase (2019-2022 - AXA)**: Evolved to leadership role, built innovative solutions like ONE AXA Dashboard, led cloud migration initiatives.

**Strategic Phase (2022-Present - CITIC)**: Applied cross-industry insights to banking, achieving significant business impact with 20+ ML models.

**Key Theme**: Each transition built upon previous experience while expanding into new domains and increasing leadership responsibility."""

        else:
            # Grounded response based on available context
            if context.strip():
                return f"""**Information Based on Available Materials**

From Rory's documented background:

{context[:500]}...

**Key Points:**
• Professional experience spans healthcare, insurance, and banking industries
• Technical expertise in Python, R, SQL, and modern ML/AI technologies  
• Leadership progression from analyst to AVP level
• Proven business impact with measurable results

*For specific questions about age, education timeline, or other details, I can reason from the available information to provide estimates.*"""
            else:
                return """**Welcome to Rory's AI Assistant**

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

        return "I'll analyze the available information to provide the most accurate answer possible. Could you rephrase your question or ask something more specific?"
    
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
        
        # Get contextual insights using GraphRAG
        insights = self._get_contextual_insights(query, relevant_docs)
        
        # Check cache for complex analysis
        cache_key = hash(query.lower().strip() + str(intent) + context[:300])
        if cache_key in self.analysis_cache:
            return {"result": self.analysis_cache[cache_key]}
        
        # Generate intelligent prompt
        prompt = self._generate_intelligent_response(query, context, intent, insights)
        
        # Try API call with parameters optimized for quality
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": "qwen-plus",  # Higher quality model for better analysis
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                "parameters": {
                    "temperature": 0.7,  # Higher temperature for more creative, human-like responses
                    "max_tokens": 1000,  # More tokens for comprehensive analysis
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
        fallback_response = self._get_intelligent_fallback(query, context, intent)
        self.analysis_cache[cache_key] = fallback_response
        return {"result": fallback_response}

@st.cache_resource
def initialize_optimized_ai_system():
    """Optimized AI system initialization - no OCR processing"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        documents = []
        
        # Process only text files (no OCR needed)
        text_extensions = ['*.txt', '*.md']
        for ext in text_extensions:
            files = glob.glob(os.path.join(current_dir, ext))
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    if text.strip() and len(text) > 100:  # Filter very short files
                        documents.append((text, {"source": os.path.basename(file_path), "type": "text"}))
                except Exception:
                    continue
        
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
            
            Key Achievements:
            - Won ICT Award, The Asian Banker Award, Best CNCBI recognition
            - Led 20+ ML models with 1.5x business uplift vs control group
            - Created AutoML pipeline reducing coding effort by 80%
            - Designed ONE AXA Dashboard used by 100+ users monthly
            - Published research on AI applications in financial services
            - Mentored 10+ junior data scientists and analysts
            - Featured speaker at Hong Kong Data Science Conference
            
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

# Custom CSS for professional styling with improved readability
st.markdown("""
<style>
    /* Global font improvements */
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
    
    /* Button styling improvements */
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
    
    /* Chat message styling */
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
    
    /* Text input styling */
    .stTextInput > div > div > input {
        font-size: 1rem !important;
        padding: 0.75rem !important;
    }
    
    /* Markdown content in CV section */
    .cv-section h1, .cv-section h2, .cv-section h3, .cv-section h4 {
        font-weight: 600 !important;
    }
    
    .cv-section p {
        font-size: 1rem !important;
        line-height: 1.5 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Sample questions styling */
    h3 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
        margin-bottom: 1rem !important;
    }
    
    /* Footer styling */
    .footer-text {
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 Rory's AI Assistant</h1>
    <h3>Data Science & Analytics Expert</h3>
    <p>Ask me anything about Rory's experience, skills, projects, or insights in data science and analytics!</p>
</div>
""", unsafe_allow_html=True)

# Initialize optimized AI system
with st.spinner("🚀 Initializing AI system (optimized for speed)..."):
    qa_chain, total_docs, total_chunks = initialize_optimized_ai_system()

if qa_chain is None:
    st.error("❌ Rory's AI system is not available. Please check the system configuration.")
    st.stop()

# Display system status
st.markdown(f"""
<div class="status-info">
    <strong style="color: #2c3e50;">🧠 Intelligent GraphRAG System Status:</strong> <span style="color: #34495e;">{total_docs} documents processed, 
    {total_chunks} knowledge chunks available | 🎯 Advanced Analysis & Human-Level Reasoning</span>
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
            "content": "👋 Hello! I'm Rory's AI assistant. I know all about Rory's experience, skills, and insights in data science and analytics. Feel free to ask me anything about his background, projects, or expertise. You can also click on the sample questions above for quick answers!"
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
    <p>🤖 <strong>Rory's AI Assistant</strong> | Powered by Advanced AI & Machine Learning | ⚡ Optimized for Speed</p>
    <p>📧 For direct contact or collaboration opportunities, please ask the AI for Rory's contact information</p>
    <p><em>This AI assistant is trained on Rory's professional experience and knowledge base</em></p>
</div>
""", unsafe_allow_html=True)
