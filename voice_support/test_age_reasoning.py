#!/usr/bin/env python3
"""
Test script for age reasoning functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from final_enhanced_assistant import AdvancedReasoningChain, EnhancedVectorStore

def test_age_reasoning():
    """Test the age reasoning functionality"""
    
    # Create a mock vector store with professional context
    vector_store = EnhancedVectorStore()
    
    # Add mock professional data
    mock_professional_data = """
    Rory Chen - Data Science & Analytics Expert
    
    Education:
    Master of Arts in Quantitative Analysis for Business, City University of Hong Kong (2016-2017)
    Master of Arts in Public Policy and Management, City University of Hong Kong (2014-2015)
    Bachelor of Social Sciences (Honours), Education University of Hong Kong (2011-2014)
    
    Professional Experience:
    AVP, Data Science at China CITIC Bank International (Nov 2022 - Current)
    Assistant Data Science Manager at AXA Hong Kong and Macau (Sep 2019 - Nov 2022)
    Data Science Analyst at Cigna International Office (Aug 2018 - Sep 2019)
    Research Executive at Ipsos Hong Kong (2017 - 2018)
    """
    
    # Mock the documents
    documents_dict = {
        "professional": [(mock_professional_data, {"source": "test", "type": "text", "category": "professional"})],
        "mindset": [],
        "general": []
    }
    
    vector_store.add_documents_by_type(documents_dict)
    
    # Create reasoning chain
    reasoning_chain = AdvancedReasoningChain(vector_store)
    
    # Test age-related questions
    age_questions = [
        "How old is Rory?",
        "What is Rory's age?",
        "How old is Rory Chen?",
        "Can you tell me Rory's age?"
    ]
    
    print("🧮 Testing Age Reasoning Functionality")
    print("=" * 50)
    
    for question in age_questions:
        print(f"\n❓ Question: {question}")
        print("-" * 30)
        
        # Analyze query
        analysis = reasoning_chain._analyze_query_complexity(question)
        print(f"🔍 Analysis: {analysis}")
        
        # Test the reasoning
        result = reasoning_chain({"query": question})
        response = result.get("result", "No response")
        
        print(f"🤖 Response Preview:")
        print(response[:200] + "..." if len(response) > 200 else response)
        print("\n" + "="*50)

if __name__ == "__main__":
    test_age_reasoning()
