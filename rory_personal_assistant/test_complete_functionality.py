#!/usr/bin/env python3
"""
Complete functionality test for the enhanced AI assistant
Tests all components including sample questions, API responses, and ChromaDB
"""

import requests
import json
import os
import time
from datetime import datetime

def test_cloud_services():
    """Test cloud ChromaDB and embedding services"""
    print("🔍 Testing Cloud Services...")
    print("="*80)
    
    host = "1.32.228.33"
    chroma_port = 8000
    embedding_port = 8001
    
    try:
        # Test ChromaDB service
        chroma_response = requests.get(f"http://{host}:{chroma_port}/health", timeout=10)
        print(f"✅ ChromaDB Service: {chroma_response.status_code} - {'Healthy' if chroma_response.status_code == 200 else 'Error'}")
        
        # Test Embedding service
        embedding_response = requests.get(f"http://{host}:{embedding_port}/health", timeout=10)
        print(f"✅ Embedding Service: {embedding_response.status_code} - {'Healthy' if embedding_response.status_code == 200 else 'Error'}")
        
        # Test document count
        collections_response = requests.get(f"http://{host}:{chroma_port}/collections", timeout=10)
        if collections_response.status_code == 200:
            result = collections_response.json()
            knowledge_base_info = result.get("knowledge_base", {})
            doc_count = knowledge_base_info.get("count", 0)
            print(f"📊 Documents in ChromaDB: {doc_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cloud services error: {str(e)}")
        return False

def test_qwen_api_with_sample_questions():
    """Test Qwen API with all sample questions"""
    print("\n🤖 Testing Qwen API with Sample Questions...")
    print("="*80)
    
    api_key = os.getenv("QWEN_API_KEY", "sk-015ea57c8b254c4181d30b2de4259d8b")
    api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Sample questions from the Streamlit app
    sample_questions = [
        "Tell me about Rory's current role at China CITIC Bank International and his key responsibilities",
        "What are Rory's key technical skills in data science and machine learning, and what major achievements has he accomplished?",
        "Can you describe Rory's career progression from his early roles to becoming AVP of Data Science? What industries has he worked in?"
    ]
    
    system_prompt = """You are Rory Chen's professional AI assistant with access to his comprehensive career information.

CORE INFORMATION ABOUT RORY:
- Current Role: AVP of Data Science at China CITIC Bank International (Nov 2022 - Current)
- 8 years of professional experience (2017-2025)
- Career progression: Research Executive → Data Science Analyst → Assistant Data Science Manager → AVP
- Industries: Market Research, Healthcare (Medicare), Insurance, Banking
- Education: Master's in Quantitative Analysis (2016-2017), Master's in Public Policy (2014-2015), Bachelor's (2011-2014)
- Location: Hong Kong SAR
- Contact: chengy823@gmail.com

TECHNICAL EXPERTISE:
- Programming: Python, Pyspark, SQL (Advanced proficiency)
- Machine Learning: Deep Learning, NLP, Computer Vision, MLOps, AutoML, LangChain
- Data Processing and ETL
- Pega Customer Decision Hub
- Analytics: Predictive Modeling, Customer Analytics, Time Series Analysis
- Cloud Platforms: Azure, Google Cloud, Databricks, Cloudera CDSW
- Visualization: Tableau, Power BI, Dashboard Development

KEY ACHIEVEMENTS:
- Developed 20+ ML models achieving 1.5x business uplift vs control groups
- Created AutoML pipeline reducing coding effort by 80%
- Designed AI+BI framework when develop models in China Citic Bank Int'l
- Created ONE AXA Dashboard which consolidated all LoBs customer and product data into a single dasboard 
- Led cloud migration from on-premise to Azure infrastructure

Provide professional, accurate responses based on this information."""

    success_count = 0
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\n📝 Testing Sample Question {i}:")
        print(f"'{question}'")
        print("-" * 60)
        
        user_prompt = f"Question: {question}\n\nPlease provide a comprehensive, professional response."
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        data = {
            "model": "qwen-turbo",
            "input": {"prompt": combined_prompt},
            "parameters": {
                "max_tokens": 2000,
                "temperature": 0.7,
                "top_p": 0.8
            }
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['output']['text'].strip()
                
                print("✅ API Response:")
                print(ai_response[:300] + "..." if len(ai_response) > 300 else ai_response)
                print(f"\n📊 Response length: {len(ai_response)} characters")
                success_count += 1
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        # Small delay between requests
        time.sleep(1)
    
    print(f"\n🎯 Sample Questions Test Results: {success_count}/{len(sample_questions)} successful")
    return success_count == len(sample_questions)

def test_chromadb_query():
    """Test ChromaDB query functionality"""
    print("\n🗄️ Testing ChromaDB Query Functionality...")
    print("="*80)
    
    host = "1.32.228.33"
    chroma_port = 8000
    
    try:
        # Test query with a sample question
        test_query = "data science experience"
        
        data = {
            "query_texts": [test_query],
            "n_results": 5
        }
        
        response = requests.post(
            f"http://{host}:{chroma_port}/collections/knowledge_base/query",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            documents = result.get("documents", [[]])[0] if result.get("documents") else []
            metadatas = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
            
            print(f"✅ Query successful: Retrieved {len(documents)} documents")
            
            if documents:
                print("\n📄 Sample retrieved document:")
                print(documents[0][:200] + "..." if len(documents[0]) > 200 else documents[0])
                
                if metadatas:
                    print(f"\n📋 Metadata: {metadatas[0]}")
            
            return True
        else:
            print(f"❌ Query failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ChromaDB query error: {str(e)}")
        return False

def test_streamlit_app():
    """Test if Streamlit app is running"""
    print("\n🌐 Testing Streamlit App...")
    print("="*80)
    
    try:
        response = requests.get("http://localhost:8888", timeout=10)
        
        if response.status_code == 200:
            print("✅ Streamlit app is running on http://localhost:8888")
            
            # Check if the response contains expected content
            content = response.text.lower()
            if "rory" in content and "assistant" in content:
                print("✅ App content looks correct")
                return True
            else:
                print("⚠️ App is running but content may be incomplete")
                return False
        else:
            print(f"❌ Streamlit app not accessible: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Streamlit app test error: {str(e)}")
        return False

def run_complete_test():
    """Run all tests and provide comprehensive report"""
    print("🚀 COMPLETE FUNCTIONALITY TEST")
    print("="*80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    test_results = {}
    
    # Test 1: Cloud Services
    test_results['cloud_services'] = test_cloud_services()
    
    # Test 2: Qwen API with Sample Questions
    test_results['qwen_api'] = test_qwen_api_with_sample_questions()
    
    # Test 3: ChromaDB Query
    test_results['chromadb_query'] = test_chromadb_query()
    
    # Test 4: Streamlit App
    test_results['streamlit_app'] = test_streamlit_app()
    
    # Final Report
    print("\n" + "="*80)
    print("📊 FINAL TEST REPORT")
    print("="*80)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Your AI assistant is fully functional.")
        print("\n🔗 Access your assistant at: http://localhost:8888")
        print("📧 Contact: chengy823@gmail.com")
    else:
        print("⚠️ Some tests failed. Please check the individual test results above.")
    
    print("="*80)
    return passed_tests == total_tests

if __name__ == "__main__":
    run_complete_test()
