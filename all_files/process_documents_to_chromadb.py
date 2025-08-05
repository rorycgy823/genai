#!/usr/bin/env python3
"""
Document Processing Script for ChromaDB
Processes documents from /root/data/ and adds them to ChromaDB
"""

import os
import sys
import chromadb
from pathlib import Path
import hashlib
import json

# Document processing imports
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    try:
        import python_docx
        from python_docx import Document
        DOCX_AVAILABLE = True
    except ImportError:
        DOCX_AVAILABLE = False

def simple_embedding(text):
    """Create a simple embedding from text using hash-based approach"""
    # Create a hash of the text
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Convert hash to numeric embedding
    embedding = []
    for i in range(0, len(text_hash), 2):
        hex_pair = text_hash[i:i+2]
        embedding.append(float(int(hex_pair, 16)) / 255.0)
    
    # Pad to 384 dimensions
    while len(embedding) < 384:
        embedding.append(0.0)
    
    return embedding[:384]

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    if not PDF_AVAILABLE:
        return f"PDF processing not available. File: {file_path}"
    
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF {file_path}: {str(e)}"

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    if not DOCX_AVAILABLE:
        return f"DOCX processing not available. File: {file_path}"
    
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading DOCX {file_path}: {str(e)}"

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read().strip()
        except Exception as e2:
            return f"Error reading TXT {file_path}: {str(e2)}"

def process_document(file_path):
    """Process a single document and extract text"""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    print(f"Processing: {file_path}")
    
    if extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif extension == '.docx':
        return extract_text_from_docx(file_path)
    elif extension == '.txt':
        return extract_text_from_txt(file_path)
    else:
        return f"Unsupported file type: {extension}"

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks for better processing"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence or paragraph
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start + chunk_size - 200, start), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def main():
    """Main function to process documents and add to ChromaDB"""
    
    # Configuration
    DATA_DIR = "/root/data"
    CHROMA_DB_PATH = "/root/chroma_db"
    COLLECTION_NAME = "documents"
    
    print("🚀 Starting Document Processing...")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"🗄️ ChromaDB path: {CHROMA_DB_PATH}")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        return
    
    # List files in data directory
    files = list(Path(DATA_DIR).glob("*"))
    print(f"📄 Found {len(files)} files in {DATA_DIR}")
    
    for file in files:
        print(f"  - {file.name}")
    
    # Initialize ChromaDB
    try:
        print("\n🔗 Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Get or create collection
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            print(f"✅ Found existing collection: {COLLECTION_NAME}")
        except:
            collection = client.create_collection(name=COLLECTION_NAME)
            print(f"✅ Created new collection: {COLLECTION_NAME}")
        
        # Check current collection size
        current_count = collection.count()
        print(f"📊 Current documents in collection: {current_count}")
        
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        return
    
    # Process each file
    processed_count = 0
    
    for file_path in files:
        if file_path.is_file():
            try:
                # Extract text
                text = process_document(file_path)
                
                if text and not text.startswith("Error") and not text.startswith("Unsupported"):
                    # Split into chunks
                    chunks = chunk_text(text)
                    print(f"  📝 Extracted {len(chunks)} chunks from {file_path.name}")
                    
                    # Add each chunk to ChromaDB
                    for i, chunk in enumerate(chunks):
                        doc_id = f"{file_path.stem}_chunk_{i}"
                        
                        # Create simple embedding
                        embedding = simple_embedding(chunk)
                        
                        # Add to collection
                        collection.add(
                            embeddings=[embedding],
                            documents=[chunk],
                            metadatas=[{
                                "filename": file_path.name,
                                "chunk_id": i,
                                "file_path": str(file_path),
                                "file_type": file_path.suffix.lower()
                            }],
                            ids=[doc_id]
                        )
                    
                    processed_count += 1
                    print(f"  ✅ Added {len(chunks)} chunks from {file_path.name}")
                    
                else:
                    print(f"  ⚠️ Could not extract text from {file_path.name}: {text[:100]}...")
                    
            except Exception as e:
                print(f"  ❌ Error processing {file_path.name}: {e}")
    
    # Final status
    final_count = collection.count()
    print(f"\n🎉 Processing Complete!")
    print(f"📊 Documents processed: {processed_count}")
    print(f"📊 Total chunks in ChromaDB: {final_count}")
    print(f"📊 New chunks added: {final_count - current_count}")
    
    # Test query
    try:
        print("\n🔍 Testing ChromaDB query...")
        results = collection.query(
            query_texts=["Rory"],
            n_results=3
        )
        print(f"✅ Query test successful - found {len(results['documents'][0])} results")
        for i, doc in enumerate(results['documents'][0]):
            print(f"  Result {i+1}: {doc[:100]}...")
    except Exception as e:
        print(f"⚠️ Query test failed: {e}")

if __name__ == "__main__":
    main()
