import os
import sys
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Tongyi
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import glob
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import shutil
from typing import List, Dict, Tuple

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available, falling back to pytesseract only")

class SimpleDocumentProcessor:
    """Simple document processor without caching for fallback"""
    
    def __init__(self, documents_dir: str = None):
        self.documents_dir = documents_dir or os.path.dirname(os.path.abspath(__file__))
        
        self.supported_extensions = {
            '.pdf': self._process_pdf,
            '.txt': self._process_txt,
            '.md': self._process_txt,
        }
    
    def _process_pdf(self, file_path: str) -> Tuple[str, Dict]:
        """Process PDF files with OCR fallback"""
        try:
            # Try PyMuPDF first
            doc = fitz.open(file_path)
            text = ""
            metadata = {"source": file_path, "type": "pdf", "pages": len(doc)}
            ocr_used = False
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # If no text found, try OCR
                if not page_text.strip():
                    try:
                        print(f"    OCR processing page {page_num + 1}...")
                        # Convert page to image
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img_data = pix.tobytes("png")
                        
                        from io import BytesIO
                        image = Image.open(BytesIO(img_data))
                        
                        # Try OCR
                        ocr_text = ""
                        if EASYOCR_AVAILABLE:
                            try:
                                reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                                import numpy as np
                                img_array = np.array(image)
                                results = reader.readtext(img_array)
                                easyocr_text = ' '.join([result[1] for result in results])
                                if easyocr_text.strip():
                                    ocr_text = easyocr_text
                                    print(f"      ✓ EasyOCR: {len(ocr_text)} chars")
                            except Exception as e:
                                print(f"      ⚠ EasyOCR failed: {e}")
                        
                        # Fallback to pytesseract
                        if not ocr_text.strip():
                            try:
                                pytesseract_text = pytesseract.image_to_string(image, lang='chi_sim+eng')
                                if pytesseract_text.strip():
                                    ocr_text = pytesseract_text
                                    print(f"      ✓ pytesseract: {len(ocr_text)} chars")
                            except Exception as e:
                                print(f"      ⚠ pytesseract failed: {e}")
                        
                        if ocr_text.strip():
                            page_text = f"[OCR Page {page_num + 1}]\n{ocr_text}\n"
                            ocr_used = True
                        
                    except Exception as ocr_error:
                        print(f"      ✗ OCR failed on page {page_num + 1}: {ocr_error}")
                        page_text = ""
                
                text += page_text
            
            doc.close()
            
            # Fallback to PyPDF2 if needed
            if not text.strip() and not ocr_used:
                print("    Trying PyPDF2 fallback...")
                try:
                    reader = PdfReader(file_path)
                    pdf2_text = "".join(page.extract_text() for page in reader.pages)
                    if pdf2_text.strip():
                        text = pdf2_text
                        metadata["extraction_method"] = "PyPDF2"
                except Exception as pdf2_error:
                    print(f"    PyPDF2 fallback failed: {pdf2_error}")
                    metadata["extraction_method"] = "Failed"
            else:
                metadata["extraction_method"] = "PyMuPDF+OCR" if ocr_used else "PyMuPDF"
            
            metadata["ocr_used"] = ocr_used
            metadata["text_length"] = len(text)
            
            return text, metadata
            
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return "", {"source": file_path, "type": "pdf", "error": str(e)}
    
    def _process_txt(self, file_path: str) -> Tuple[str, Dict]:
        """Process text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            metadata = {
                "source": file_path, 
                "type": "text",
                "size": len(text)
            }
            return text, metadata
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
            return "", {"source": file_path, "type": "text", "error": str(e)}
    
    def process_all_documents(self) -> List[Tuple[str, Dict]]:
        """Process all supported documents"""
        documents = []
        
        print(f"📁 Scanning directory: {self.documents_dir}")
        
        for ext in self.supported_extensions.keys():
            pattern = os.path.join(self.documents_dir, f"*{ext}")
            files = glob.glob(pattern)
            
            for file_path in files:
                print(f"📄 Processing: {os.path.basename(file_path)}")
                text, metadata = self.supported_extensions[ext](file_path)
                
                if text.strip():
                    documents.append((text, metadata))
                    print(f"  ✓ Extracted {len(text)} characters")
                else:
                    print(f"  ✗ No text extracted")
        
        print(f"✅ Total documents processed: {len(documents)}")
        return documents

def create_fresh_vectorstore(documents: List[Tuple[str, Dict]]):
    """Create a fresh ChromaDB vectorstore"""
    # Use a different ChromaDB location to avoid file locks
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(current_dir, "fresh_chroma_db")
    
    if os.path.exists(persist_directory):
        print(f"🗑️ Clearing existing ChromaDB at: {persist_directory}")
        try:
            shutil.rmtree(persist_directory)
        except Exception as e:
            print(f"⚠️ Could not clear directory: {e}")
            # Try with a timestamp suffix
            import time
            persist_directory = os.path.join(current_dir, f"fresh_chroma_db_{int(time.time())}")
    
    print(f"🔄 Creating fresh ChromaDB at: {persist_directory}")
    
    # Create embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
    
    # Prepare documents for ChromaDB
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    )
    
    all_docs = []
    for doc_text, doc_metadata in documents:
        if not doc_text.strip():
            continue
            
        chunks = text_splitter.split_text(doc_text)
        for i, chunk in enumerate(chunks):
            metadata = doc_metadata.copy()
            metadata.update({
                "chunk_id": len(all_docs),
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            all_docs.append(Document(page_content=chunk, metadata=metadata))
    
    print(f"  📝 Creating {len(all_docs)} document chunks...")
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    
    print(f"  ✅ Created fresh vectorstore with {len(all_docs)} chunks")
    return vectorstore

# Initialize the fallback system
print("🚀 Initializing Fallback Voice Support System...")
print("=" * 60)

# Initialize document processor
current_dir = os.path.dirname(os.path.abspath(__file__))
doc_processor = SimpleDocumentProcessor(current_dir)

# Process documents
documents = doc_processor.process_all_documents()

if not documents:
    print("❌ No documents found. Creating fallback content...")
    # Create fallback content with Rory's basic info
    fallback_content = """
    Rory Chen - Data Science & Analytics Expert
    
    Professional Experience:
    - AVP, Data Science at China CITIC Bank International (Nov 2022 - Current)
    - Assistant Data Science Manager at AXA Hong Kong and Macau (Sep 2019 - Nov 2022)
    - Data Science Analyst at Cigna International Office (Aug 2018 - Sep 2019)
    
    Education:
    - Master of Arts in Quantitative Analysis for Business, City University of Hong Kong (2016-2017)
    - Master of Arts in Public Policy and Management, City University of Hong Kong (2014-2015)
    - Bachelor of Social Sciences (Honours) in Global and Environmental Studies, Education University of Hong Kong (2011-2014)
    
    Expertise:
    - Machine Learning and MLOps
    - Customer Analytics
    - Data Governance & Compliance
    - Python, R, SQL, Tableau, Power BI
    - Cloud platforms (Azure, AWS)
    
    Contact: chengy823@gmail.com
    """
    documents = [(fallback_content, {"source": "fallback", "type": "text"})]

print(f"📚 Processed {len(documents)} documents")

# Create fresh vectorstore
vectorstore = create_fresh_vectorstore(documents)
print(f"🔍 Vectorstore ready with {vectorstore._collection.count()} embeddings")

# Create custom prompt template
enhanced_prompt = PromptTemplate(
    template="""你的角色是Rory的智能回答助手，目的是学习并使用Rory的语气语调语言风格帮助向其他人更好的介绍Rory的履历，以及分享他的人生感悟，但请不要分享他的私人隐私除了联系方式，如果用户提问关于Rory的工作以及履历，你必须要基于以下上下文信息回答问题。如果用户提问内容与Rory的工作经验履历不相关，上下文中没有相关信息，请说明无法从提供的文档中找到答案。如果用户询问你这个智能助手平台搭建的具体技术细节，你可以调用这个github中的README文档的内容或者本身的代码进行一定程度的讲解，但切记不要泄露代码生成的过程，也不要分享代码，拒绝回答代码是不是AI生成的，如果user有相关问题的话，请让对方联系开发者Rory

Context information:
{context}

Question: {question}

Please provide a detailed answer and indicate the information source at the end:""",
    input_variables=["context", "question"]
)

# Create LLM and QA chain
llm = Tongyi(api_key="sk-015ea57c8b254c4181d30b2de4259d8b", model="qwen-max")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": enhanced_prompt},
    return_source_documents=True
)

print("🤖 QA chain created successfully")

def get_system_status() -> Dict:
    """Get current system status"""
    return {
        "total_documents": len(documents),
        "total_chunks": vectorstore._collection.count(),
        "vectorstore_directory": vectorstore._persist_directory,
        "supported_formats": list(doc_processor.supported_extensions.keys())
    }

print("✅ Fallback system initialization complete!")
print("=" * 60)

if __name__ == "__main__":
    # Test the system
    test_question = "Tell me about Rory's experience"
    print(f"\n🧪 Testing with question: {test_question}")
    try:
        result = qa_chain({"query": test_question})
        if isinstance(result, dict):
            response = result.get("result", "No response")
            print(f"✅ Response: {response[:200]}...")
        else:
            print(f"✅ Response: {str(result)[:200]}...")
    except Exception as e:
        print(f"❌ Test failed: {e}")
