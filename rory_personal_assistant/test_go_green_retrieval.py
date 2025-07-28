"""
Test script to verify if the "go green" project from Project experience highlight - Rory.docx
can be successfully extracted and retrieved by the AI system.
"""

import os
import sys
sys.path.append('.')

# Import the classes from the main application
from streamlit_compatible_assistant import TextChunker, DocumentProcessor, ChromaDBManager

def test_go_green_retrieval():
    """Test if go green project can be extracted and retrieved"""
    
    print("🌱 Testing Go Green Project Retrieval")
    print("=" * 50)
    
    # Initialize components
    chunker = TextChunker(chunk_size=1000, overlap=200, min_chunk_size=100)
    processor = DocumentProcessor(chunker=chunker)
    chroma_db = ChromaDBManager(persist_directory="./test_chroma_db")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docx_file = "Project experience highlight - Rory.docx"
    docx_path = os.path.join(current_dir, docx_file)
    
    print(f"📁 Looking for: {docx_path}")
    print(f"📋 File exists: {'✅' if os.path.exists(docx_path) else '❌'}")
    
    if not os.path.exists(docx_path):
        print("❌ DOCX file not found!")
        return
    
    # Test DOCX processing
    print(f"\n📄 Processing {docx_file}:")
    try:
        # Extract text from DOCX
        text = processor.extract_text_from_docx(docx_path)
        print(f"   📊 Extracted text length: {len(text)} characters")
        
        if len(text) < 50:
            print(f"   ❌ Text extraction failed or too short: {text}")
            return
        
        # Check if "go green" content is in the extracted text
        go_green_keywords = ["go green", "green", "ESG", "environmental", "sustainability", "homesurance"]
        found_keywords = []
        
        text_lower = text.lower()
        for keyword in go_green_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        print(f"   🔍 Keywords found: {found_keywords}")
        
        # Show relevant excerpts
        if "go green" in text_lower:
            # Find the context around "go green"
            go_green_pos = text_lower.find("go green")
            start = max(0, go_green_pos - 200)
            end = min(len(text), go_green_pos + 300)
            excerpt = text[start:end]
            print(f"   📝 Go Green excerpt: ...{excerpt}...")
        
        # Apply chunking
        chunks = chunker.chunk_text(text, docx_file)
        print(f"   🧩 Generated {len(chunks)} chunks")
        
        # Check which chunks contain go green content
        go_green_chunks = []
        for i, (chunk_text, metadata) in enumerate(chunks):
            if any(keyword.lower() in chunk_text.lower() for keyword in go_green_keywords):
                go_green_chunks.append((i, chunk_text, metadata))
        
        print(f"   🎯 Chunks with go green content: {len(go_green_chunks)}")
        
        for i, (chunk_idx, chunk_text, metadata) in enumerate(go_green_chunks):
            print(f"      Chunk {chunk_idx + 1}: {len(chunk_text)} chars")
            print(f"      Preview: {chunk_text[:200]}...")
            print()
        
        # Test ChromaDB storage and retrieval
        print(f"\n💾 Testing ChromaDB Storage and Retrieval:")
        
        # Prepare documents for storage
        documents = [chunk[0] for chunk in chunks]
        metadatas = [chunk[1] for chunk in chunks]
        ids = [f"{docx_file}_chunk_{i}" for i in range(len(chunks))]
        
        # Add to ChromaDB
        success = chroma_db.add_documents(documents, metadatas, ids)
        print(f"   📥 Storage success: {'✅' if success else '❌'}")
        
        if success:
            # Test retrieval with go green queries
            test_queries = [
                "go green project",
                "ESG home solution", 
                "environmental project",
                "green home insurance",
                "sustainability initiative"
            ]
            
            print(f"   🔍 Testing retrieval with go green queries:")
            
            for query in test_queries:
                print(f"\n      Query: '{query}'")
                results = chroma_db.query_documents(query, n_results=3)
                
                if results['documents'] and results['documents'][0]:
                    retrieved_docs = results['documents'][0]
                    print(f"      Retrieved {len(retrieved_docs)} chunks")
                    
                    # Check if any retrieved chunks contain go green content
                    relevant_chunks = 0
                    for doc in retrieved_docs:
                        if any(keyword.lower() in doc.lower() for keyword in go_green_keywords):
                            relevant_chunks += 1
                    
                    print(f"      Relevant chunks: {relevant_chunks}/{len(retrieved_docs)}")
                    
                    if relevant_chunks > 0:
                        print(f"      ✅ Successfully retrieved go green content!")
                        # Show first relevant chunk
                        for doc in retrieved_docs:
                            if any(keyword.lower() in doc.lower() for keyword in go_green_keywords):
                                print(f"      Sample: {doc[:150]}...")
                                break
                    else:
                        print(f"      ❌ No relevant go green content retrieved")
                else:
                    print(f"      ❌ No documents retrieved")
        
    except Exception as e:
        print(f"   ❌ Error processing DOCX: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Go Green Retrieval Test Complete!")
    
    # Cleanup test database
    try:
        import shutil
        if os.path.exists("./test_chroma_db"):
            shutil.rmtree("./test_chroma_db")
            print("🧹 Cleaned up test database")
    except:
        pass

if __name__ == "__main__":
    test_go_green_retrieval()
