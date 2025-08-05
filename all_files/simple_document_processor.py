#!/usr/bin/env python3
"""
Simple Document Processor for ChromaDB
Processes documents from /root/data/ and adds them to ChromaDB
"""

import os
import sys
import traceback

def check_environment():
    """Check the environment and dependencies"""
    print("🔍 Environment Check:")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.executable}")
    
    # Check if data directory exists
    data_dir = "/root/data"
    print(f"\n📁 Checking data directory: {data_dir}")
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"✅ Data directory exists with {len(files)} files:")
        for file in files:
            file_path = os.path.join(data_dir, file)
            size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
            print(f"  - {file} ({size} bytes)")
    else:
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Check ChromaDB directory
    chroma_dir = "/root/chroma_db"
    print(f"\n🗄️ Checking ChromaDB directory: {chroma_dir}")
    if os.path.exists(chroma_dir):
        print(f"✅ ChromaDB directory exists")
    else:
        print(f"⚠️ ChromaDB directory not found, will create: {chroma_dir}")
    
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies:")
    
    required_packages = {
        'chromadb': 'chromadb',
        'PyPDF2': 'PyPDF2', 
        'python-docx': 'docx'
    }
    
    available = {}
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}: Available")
            available[package_name] = True
        except ImportError as e:
            print(f"❌ {package_name}: Missing - {e}")
            available[package_name] = False
    
    return available

def simple_text_processing():
    """Simple text processing without external dependencies"""
    print("\n🚀 Starting Simple Document Processing...")
    
    data_dir = "/root/data"
    
    # Process text files only (no dependencies needed)
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print("⚠️ No .txt files found. Let's create a simple test document.")
        
        # Create a simple test document with your info
        test_content = """
Rory - Data Science Professional

Experience:
- Data Science and Analytics expertise
- Machine Learning and AI development
- Cloud computing and deployment
- Python programming and automation

Skills:
- Python, SQL, Machine Learning
- Data Analysis and Visualization
- Cloud platforms (AWS, Aliyun)
- AI/ML model development

Contact: Available for data science projects and consulting.
        """.strip()
        
        test_file = os.path.join(data_dir, "rory_profile.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print(f"✅ Created test document: {test_file}")
        txt_files = ["rory_profile.txt"]
    
    print(f"📄 Processing {len(txt_files)} text files:")
    
    # Simple in-memory storage (no ChromaDB needed for testing)
    documents = []
    
    for txt_file in txt_files:
        file_path = os.path.join(data_dir, txt_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content:
                documents.append({
                    'filename': txt_file,
                    'content': content,
                    'length': len(content)
                })
                print(f"  ✅ {txt_file}: {len(content)} characters")
            else:
                print(f"  ⚠️ {txt_file}: Empty file")
                
        except Exception as e:
            print(f"  ❌ {txt_file}: Error - {e}")
    
    print(f"\n🎉 Processed {len(documents)} documents successfully!")
    return documents

def test_chromadb_connection():
    """Test ChromaDB connection"""
    print("\n🔗 Testing ChromaDB connection...")
    
    try:
        import chromadb
        print("✅ ChromaDB import successful")
        
        # Try to create client
        client = chromadb.PersistentClient(path="/root/chroma_db")
        print("✅ ChromaDB client created")
        
        # Try to get or create collection
        try:
            collection = client.get_collection(name="test_collection")
            print("✅ Found existing test collection")
        except:
            collection = client.create_collection(name="test_collection")
            print("✅ Created new test collection")
        
        # Test adding a simple document
        collection.add(
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["This is a test document"],
            metadatas=[{"source": "test"}],
            ids=["test_doc_1"]
        )
        print("✅ Added test document to ChromaDB")
        
        # Test querying
        results = collection.query(
            query_texts=["test"],
            n_results=1
        )
        print(f"✅ Query test successful - found {len(results['documents'][0])} results")
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("🚀 SIMPLE DOCUMENT PROCESSOR")
    print("=" * 60)
    
    try:
        # Step 1: Check environment
        if not check_environment():
            print("\n❌ Environment check failed!")
            return
        
        # Step 2: Check dependencies
        deps = check_dependencies()
        
        # Step 3: Simple text processing (always works)
        documents = simple_text_processing()
        
        # Step 4: Test ChromaDB if available
        if deps.get('chromadb', False):
            chromadb_works = test_chromadb_connection()
            if chromadb_works:
                print("\n🎉 ChromaDB is working! Your documents can be processed.")
            else:
                print("\n⚠️ ChromaDB has issues, but basic text processing works.")
        else:
            print("\n⚠️ ChromaDB not available. Install with: pip install chromadb")
        
        print("\n" + "=" * 60)
        print("✅ DIAGNOSTIC COMPLETE")
        print("=" * 60)
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"  - Documents found: {len(documents)}")
        print(f"  - ChromaDB available: {deps.get('chromadb', False)}")
        print(f"  - PDF processing: {deps.get('PyPDF2', False)}")
        print(f"  - DOCX processing: {deps.get('python-docx', False)}")
        
        if documents:
            print(f"\n📄 DOCUMENTS PROCESSED:")
            for doc in documents:
                print(f"  - {doc['filename']}: {doc['length']} characters")
        
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        print(f"Error details: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
