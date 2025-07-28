#!/usr/bin/env python3
"""
Test OCR Functionality for PDF Processing
========================================

This script tests the OCR capabilities added to the Streamlit assistant
for handling PDFs that fail regular text extraction.

Author: Rory Chen
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from streamlit_compatible_assistant import DocumentProcessor, TextChunker
    print("✅ Successfully imported DocumentProcessor")
except ImportError as e:
    print(f"❌ Failed to import DocumentProcessor: {e}")
    sys.exit(1)

def test_ocr_libraries():
    """Test if OCR libraries are available"""
    print("\n🔍 Testing OCR Library Availability:")
    
    # Test EasyOCR
    try:
        import easyocr
        print("✅ EasyOCR is available")
        easyocr_available = True
    except ImportError:
        print("❌ EasyOCR is not available")
        easyocr_available = False
    
    # Test Tesseract
    try:
        import pytesseract
        from PIL import Image
        print("✅ Tesseract and PIL are available")
        tesseract_available = True
    except ImportError:
        print("❌ Tesseract or PIL is not available")
        tesseract_available = False
    
    # Test PyMuPDF for PDF to image conversion
    try:
        import fitz
        print("✅ PyMuPDF is available")
        pymupdf_available = True
    except ImportError:
        print("❌ PyMuPDF is not available")
        pymupdf_available = False
    
    return easyocr_available, tesseract_available, pymupdf_available

def create_test_pdf_with_image():
    """Create a simple test PDF with image-based text (simulated)"""
    try:
        import fitz
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        # Create a simple image with text
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Add some text to the image
        test_text = """
        Rory Chen - Professional Summary
        
        AVP of Data Science at China CITIC Bank International
        8 years of experience in data science and analytics
        
        Key Skills:
        • Python, PySpark, SQL
        • Machine Learning & AI
        • Cloud Platforms (Azure, Google Cloud)
        • Data Visualization
        
        Contact: chengy823@gmail.com
        """
        
        draw.multiline_text((50, 50), test_text, fill='black', font=font)
        
        # Save image to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Create PDF with the image
        doc = fitz.open()
        page = doc.new_page()
        
        # Insert the image
        img_bytes.seek(0)
        page.insert_image(fitz.Rect(0, 0, 595, 842), stream=img_bytes.getvalue())
        
        # Save to temporary file
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        doc.save(temp_pdf.name)
        doc.close()
        
        print(f"✅ Created test PDF with image-based text: {temp_pdf.name}")
        return temp_pdf.name
        
    except Exception as e:
        print(f"❌ Failed to create test PDF: {e}")
        return None

def test_pdf_processing(pdf_path):
    """Test PDF processing with OCR fallback"""
    if not pdf_path or not os.path.exists(pdf_path):
        print("❌ No valid PDF file to test")
        return
    
    print(f"\n📄 Testing PDF processing on: {os.path.basename(pdf_path)}")
    
    # Initialize document processor
    chunker = TextChunker(chunk_size=500, overlap=100)
    processor = DocumentProcessor(chunker=chunker)
    
    # Test text extraction
    try:
        extracted_text = processor.extract_text_from_pdf(pdf_path)
        
        print(f"📝 Extracted text length: {len(extracted_text)} characters")
        
        if len(extracted_text) > 100:
            print("✅ Text extraction successful!")
            print("\n📋 First 300 characters of extracted text:")
            print("-" * 50)
            print(extracted_text[:300] + "..." if len(extracted_text) > 300 else extracted_text)
            print("-" * 50)
        else:
            print("⚠️ Text extraction returned minimal content")
            print(f"Content: {extracted_text}")
        
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")

def test_ocr_methods_directly():
    """Test OCR methods directly if libraries are available"""
    print("\n🔬 Testing OCR Methods Directly:")
    
    # Create a simple test image with text
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create test image
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((20, 50), "Rory Chen - Data Science Expert", fill='black', font=font)
        draw.text((20, 100), "Email: chengy823@gmail.com", fill='black', font=font)
        
        # Save to temporary file
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        img.save(temp_img.name)
        
        print(f"✅ Created test image: {temp_img.name}")
        
        # Test EasyOCR
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            results = reader.readtext(temp_img.name)
            
            print("✅ EasyOCR Results:")
            for result in results:
                if result[2] > 0.5:  # Confidence > 0.5
                    print(f"   Text: '{result[1]}' (Confidence: {result[2]:.2f})")
        except Exception as e:
            print(f"❌ EasyOCR test failed: {e}")
        
        # Test Tesseract
        try:
            import pytesseract
            text = pytesseract.image_to_string(img)
            print(f"✅ Tesseract Results: '{text.strip()}'")
        except Exception as e:
            print(f"❌ Tesseract test failed: {e}")
        
        # Clean up
        try:
            os.unlink(temp_img.name)
        except:
            pass
            
    except Exception as e:
        print(f"❌ Direct OCR testing failed: {e}")

def main():
    """Main test function"""
    print("🧪 OCR Functionality Test Suite")
    print("=" * 50)
    
    # Test library availability
    easyocr_available, tesseract_available, pymupdf_available = test_ocr_libraries()
    
    if not (easyocr_available or tesseract_available):
        print("\n⚠️ No OCR libraries available. Install with:")
        print("pip install easyocr pytesseract Pillow")
        return
    
    if not pymupdf_available:
        print("\n⚠️ PyMuPDF not available. Install with:")
        print("pip install PyMuPDF")
        return
    
    # Test OCR methods directly
    test_ocr_methods_directly()
    
    # Create and test with a sample PDF
    test_pdf_path = create_test_pdf_with_image()
    if test_pdf_path:
        test_pdf_processing(test_pdf_path)
        
        # Clean up
        try:
            os.unlink(test_pdf_path)
            print(f"\n🧹 Cleaned up test file: {os.path.basename(test_pdf_path)}")
        except:
            pass
    
    # Test with any existing PDF files in the directory
    current_dir = Path(__file__).parent
    pdf_files = list(current_dir.glob("*.pdf"))
    
    if pdf_files:
        print(f"\n📁 Found {len(pdf_files)} PDF files in current directory:")
        for pdf_file in pdf_files[:3]:  # Test up to 3 files
            test_pdf_processing(str(pdf_file))
    else:
        print("\n📁 No existing PDF files found in current directory")
    
    print("\n✅ OCR functionality test completed!")

if __name__ == "__main__":
    main()
