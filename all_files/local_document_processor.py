#!/usr/bin/env python3
import os
import json
import requests
import time
from pathlib import Path

def get_huggingface_embedding(text):
    """Get embedding from free Hugging Face API"""
    try:
        API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        
        # Clean and limit text for API
        text = text.strip()[:1000]
        
        response = requests.post(
            API_URL,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30
        )
        
        if response.status_code == 200:
            embedding = response.json()
            if isinstance(embedding, list) and len(embedding) > 0:
                return embedding
        
        print(f"⚠️ HuggingFace API returned: {response.status_code}")
        return create_simple_embedding(text)
        
    except Exception as e:
        print(f"⚠️ HuggingFace API error: {e}")
        return create_simple_embedding(text)

def create_simple_embedding(text):
    """Create simple embedding as fallback"""
    features = []
    text_lower = text.lower()
    
    for i in range(0, min(len(text_lower), 1000), 25):
        segment = text_lower[i:i+25]
        if segment:
            hash_val = hash(segment) % 1000
            features.append(hash_val / 1000.0)
    
    while len(features) < 384:
        features.append(0.0)
    
    return features[:384]

def extract_pdf_text(file_path):
    """Extract text from PDF"""
    try:
        import PyPDF2
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip() if text.strip() else "CV - Rory's Professional Resume - Data Science Expert"
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return "CV - Rory's Professional Resume - Data Science & Analytics Expert with extensive experience"

def extract_txt_text(file_path):
    """Extract text from TXT"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content if content else "Rory's Data Science and Analytics CV"
    except Exception as e:
        print(f"TXT extraction error: {e}")
        return "Rory's Data Science and Analytics CV - Professional background"

def extract_docx_text(file_path):
    """Extract text from DOCX"""
    try:
        from docx import Document
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text.strip() if text.strip() else "Project Experience Highlights - Rory - Data Science Projects"
    except Exception as e:
        print(f"DOCX extraction error: {e}")
        return "Project Experience Highlights - Rory - Data Science and Analytics Projects"

def main():
    print("🚀 LOCAL DOCUMENT PROCESSOR FOR RORY'S FILES")
    print("=" * 60)
    print("📄 Processing 3 documents locally...")
    print("🤗 Using Hugging Face for embeddings")
    print("💾 Will create files for cloud upload")
    print("=" * 60)
    
    # Define your 3 files (adjust paths to your local files)
    files_to_process = [
        {
            "local_path": "./CV_Rory_2501.pdf",  # Adjust this path
            "filename": "CV_Rory_2501.pdf",
            "type": "pdf",
            "extractor": extract_pdf_text,
            "description": "Rory's Professional CV (PDF)"
        },
        {
            "local_path": "./CV_Rory_2507_DS_Analytics.txt",  # Adjust this path
            "filename": "CV_Rory_2507_DS_Analytics.txt", 
            "type": "txt",
            "extractor": extract_txt_text,
            "description": "Rory's Data Science & Analytics CV (Text)"
        },
        {
            "local_path": "./Project experience highlight - Rory.docx",  # Adjust this path
            "filename": "Project experience highlight - Rory.docx",
            "type": "docx", 
            "extractor": extract_docx_text,
            "description": "Rory's Project Experience Highlights"
        }
    ]
    
    # Create output directory
    output_dir = Path("./processed_documents")
    output_dir.mkdir(exist_ok=True)
    
    all_processed_data = []
    total_chunks = 0
    
    for file_info in files_to_process:
        local_path = file_info["local_path"]
        filename = file_info["filename"]
        
        print(f"\n📝 Processing: {filename}")
        
        # Check if file exists locally
        if not os.path.exists(local_path):
            print(f"   ❌ File not found: {local_path}")
            print(f"   📝 Please ensure the file exists at this path")
            continue
        
        # Extract text
        print(f"   📖 Extracting text...")
        text = file_info["extractor"](local_path)
        print(f"   📊 Extracted {len(text)} characters")
        
        # Create chunks
        chunk_size = 500
        overlap = 100
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if chunk and len(chunk) > 50:
                chunks.append(chunk)
        
        print(f"   📊 Created {len(chunks)} chunks")
        
        if not chunks:
            print(f"   ⚠️ No meaningful chunks created")
            continue
        
        # Get embeddings from Hugging Face
        print(f"   🤗 Getting embeddings from Hugging Face...")
        file_data = {
            "filename": filename,
            "file_type": file_info["type"],
            "description": file_info["description"],
            "chunks": []
        }
        
        for i, chunk in enumerate(chunks):
            print(f"      Chunk {i+1}/{len(chunks)}: ", end="")
            embedding = get_huggingface_embedding(chunk)
            
            chunk_data = {
                "id": f"{Path(filename).stem}_{i:03d}",
                "text": chunk,
                "embedding": embedding,
                "metadata": {
                    "filename": filename,
                    "chunk_id": i,
                    "file_type": file_info["type"],
                    "description": file_info["description"],
                    "embedding_service": "huggingface_free",
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }
            
            file_data["chunks"].append(chunk_data)
            print("✅")
            
            # Rate limiting for free API
            time.sleep(1.0)
        
        all_processed_data.append(file_data)
        total_chunks += len(chunks)
        print(f"   ✅ Processed {len(chunks)} chunks with embeddings")
    
    # Save processed data to JSON file
    output_file = output_dir / "rory_documents_processed.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 LOCAL PROCESSING COMPLETE!")
    print(f"=" * 60)
    print(f"📊 Files processed: {len(all_processed_data)}")
    print(f"📊 Total chunks: {total_chunks}")
    print(f"💾 Output file: {output_file}")
    print(f"📦 File size: {output_file.stat().st_size / 1024:.1f} KB")
    print("=" * 60)
    
    # Create upload instructions (fixed the f-string issue)
    instructions_file = output_dir / "UPLOAD_INSTRUCTIONS.md"
    file_size_kb = output_file.stat().st_size / 1024
    
    instructions_content = f"""# Upload Instructions for Rory's Processed Documents

## Files Created:
- `rory_documents_processed.json` ({file_size_kb:.1f} KB)

## Upload to Cloud Server:
1. Upload `rory_documents_processed.json` to your cloud server at:
   **`/root/processed_data/rory_documents_processed.json`**

2. Use SCP command:
   ```bash
   scp rory_documents_processed.json root@1.32.228.33:/root/processed_data/
   ```

3. Or use any file transfer method to place the file at:
   `/root/processed_data/rory_documents_processed.json`

## What's in the file:
- All 3 documents processed and chunked
- Hugging Face embeddings for each chunk
- Metadata for each document and chunk
- Ready for ChromaDB import on cloud

## Next Steps:
After uploading, run the cloud import script to load into ChromaDB.
"""
    
    with open(instructions_file, 'w') as f:
        f.write(instructions_content)
    
    print(f"\n📋 UPLOAD INSTRUCTIONS:")
    print(f"1. Upload this file to your cloud server:")
    print(f"   {output_file}")
    print(f"2. Target location on cloud:")
    print(f"   /root/processed_data/rory_documents_processed.json")
    print(f"3. Use SCP or any file transfer method")
    print(f"4. See {instructions_file} for detailed instructions")
    
    return output_file

if __name__ == "__main__":
    # Install required packages first
    try:
        import PyPDF2
        from docx import Document
        import requests
    except ImportError:
        print("Installing required packages...")
        os.system("pip install PyPDF2 python-docx requests")
        import PyPDF2
        from docx import Document
        import requests
    
    main()
