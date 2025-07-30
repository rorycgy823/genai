#!/usr/bin/env python3
"""
Comprehensive Cloud Service Health Check
======================================

Advanced monitoring system for Chroma service and embedding API
- Tests service availability and functionality
- Validates document processing from /root/data/
- Monitors database integrity and embedding quality
- Provides detailed health reports for web service integration

Author: Rory Chen
Usage: python cloud_service_health_check.py
"""

import os
import sys
import json
import time
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cloud_health_check.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CloudServiceHealthChecker:
    """Comprehensive health checker for cloud services"""
    
    def __init__(self, vm_ip: str = "1.32.228.33", 
                 chroma_port: int = 8000, 
                 embedding_port: int = 8001):
        self.vm_ip = vm_ip
        self.chroma_port = chroma_port
        self.embedding_port = embedding_port
        
        self.chroma_url = f"http://{vm_ip}:{chroma_port}"
        self.embedding_url = f"http://{vm_ip}:{embedding_port}"
        
        self.health_report = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "document_processing": {},
            "overall_status": "unknown",
            "recommendations": []
        }
        
        # Expected documents in /root/data/
        self.expected_documents = [
            "CV_Rory_2507_DS_Analytics.txt",
            "Project experience highlight - Rory.docx", 
            "CV_Rory_2501.pdf"
        ]
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run complete health check suite"""
        logger.info("Starting comprehensive cloud service health check...")
        
        try:
            # 1. Basic connectivity tests
            self._test_service_connectivity()
            
            # 2. Service health endpoints
            self._test_service_health()
            
            # 3. Embedding API functionality
            self._test_embedding_functionality()
            
            # 4. ChromaDB functionality
            self._test_chromadb_functionality()
            
            # 5. Document processing validation
            self._test_document_processing()
            
            # 6. Database integrity check
            self._test_database_integrity()
            
            # 7. Performance benchmarks
            self._test_performance()
            
            # 8. Generate overall assessment
            self._generate_overall_assessment()
            
            logger.info("Health check completed successfully")
            return self.health_report
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            self.health_report["overall_status"] = "failed"
            self.health_report["error"] = str(e)
            return self.health_report
    
    def _test_service_connectivity(self):
        """Test basic network connectivity to services"""
        logger.info("Testing service connectivity...")
        
        connectivity = {
            "chroma_service": self._test_connection(self.chroma_url),
            "embedding_service": self._test_connection(self.embedding_url)
        }
        
        self.health_report["services"]["connectivity"] = connectivity
        logger.info(f"Connectivity results: {connectivity}")
    
    def _test_connection(self, url: str, timeout: int = 5) -> Dict[str, Any]:
        """Test connection to a service"""
        try:
            start_time = time.time()
            response = requests.get(f"{url}/", timeout=timeout)
            response_time = time.time() - start_time
            
            return {
                "status": "connected",
                "response_code": response.status_code,
                "response_time_ms": round(response_time * 1000, 2),
                "accessible": response.status_code == 200
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "failed",
                "error": str(e),
                "accessible": False
            }
    
    def _test_service_health(self):
        """Test service health endpoints"""
        logger.info("Testing service health endpoints...")
        
        health_results = {}
        
        # Test ChromaDB health
        health_results["chroma_health"] = self._get_service_health(
            f"{self.chroma_url}/health", "ChromaDB"
        )
        
        # Test Embedding API health
        health_results["embedding_health"] = self._get_service_health(
            f"{self.embedding_url}/health", "Embedding API"
        )
        
        self.health_report["services"]["health"] = health_results
        logger.info("Service health check completed")
    
    def _get_service_health(self, health_url: str, service_name: str) -> Dict[str, Any]:
        """Get health status from a service"""
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                return {
                    "status": "healthy",
                    "details": health_data,
                    "service": service_name
                }
            else:
                return {
                    "status": "unhealthy",
                    "response_code": response.status_code,
                    "service": service_name
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "service": service_name
            }
    
    def _test_embedding_functionality(self):
        """Test embedding API functionality"""
        logger.info("Testing embedding API functionality...")
        
        test_texts = [
            "This is a test document for embedding generation",
            "Rory Chen is a data scientist with expertise in AI",
            "ChromaDB is a vector database for AI applications"
        ]
        
        embedding_tests = {}
        
        # Test basic embedding generation
        embedding_tests["basic_embedding"] = self._test_embedding_generation(test_texts)
        
        # Test batch embedding
        embedding_tests["batch_embedding"] = self._test_batch_embedding(test_texts * 10)
        
        # Test model availability
        embedding_tests["available_models"] = self._test_available_models()
        
        # Test similarity computation
        embedding_tests["similarity_test"] = self._test_similarity_computation()
        
        self.health_report["services"]["embedding_functionality"] = embedding_tests
        logger.info("Embedding API functionality test completed")
    
    def _test_embedding_generation(self, texts: List[str]) -> Dict[str, Any]:
        """Test basic embedding generation"""
        try:
            payload = {
                "texts": texts,
                "model": "default",
                "normalize": True
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.embedding_url}/embeddings",
                json=payload,
                timeout=30
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                embeddings = result.get("embeddings", [])
                
                return {
                    "status": "success",
                    "embeddings_generated": len(embeddings),
                    "embedding_dimension": result.get("dimension", 0),
                    "model_used": result.get("model_used", "unknown"),
                    "processing_time_ms": round(processing_time * 1000, 2),
                    "avg_time_per_text_ms": round((processing_time * 1000) / len(texts), 2)
                }
            else:
                return {
                    "status": "failed",
                    "response_code": response.status_code,
                    "error": response.text
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _test_batch_embedding(self, texts: List[str]) -> Dict[str, Any]:
        """Test batch embedding processing"""
        try:
            payload = {
                "texts": texts,
                "model": "default"
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.embedding_url}/embeddings/batch",
                json=payload,
                timeout=60
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "total_texts": len(texts),
                    "embeddings_generated": len(result.get("embeddings", [])),
                    "batch_size": result.get("batch_size", 0),
                    "total_processing_time_ms": round(processing_time * 1000, 2)
                }
            else:
                return {
                    "status": "failed",
                    "response_code": response.status_code
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _test_available_models(self) -> Dict[str, Any]:
        """Test available embedding models"""
        try:
            response = requests.get(f"{self.embedding_url}/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                return {
                    "status": "success",
                    "available_models": models_data.get("available_models", {}),
                    "default_model": models_data.get("default_model", "unknown"),
                    "total_models": models_data.get("total_models", 0)
                }
            else:
                return {
                    "status": "failed",
                    "response_code": response.status_code
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _test_similarity_computation(self) -> Dict[str, Any]:
        """Test similarity computation functionality"""
        try:
            texts1 = ["Data science and machine learning"]
            texts2 = ["AI and artificial intelligence"]
            
            response = requests.post(
                f"{self.embedding_url}/embeddings/similarity",
                json={"texts1": texts1, "texts2": texts2},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "similarities": result.get("similarities", []),
                    "model_used": result.get("model_used", "unknown")
                }
            else:
                return {
                    "status": "failed",
                    "response_code": response.status_code
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _test_chromadb_functionality(self):
        """Test ChromaDB functionality"""
        logger.info("Testing ChromaDB functionality...")
        
        chroma_tests = {}
        
        # Test collections info
        chroma_tests["collections_info"] = self._test_collections_info()
        
        # Test document addition
        chroma_tests["document_addition"] = self._test_document_addition()
        
        # Test document querying
        chroma_tests["document_querying"] = self._test_document_querying()
        
        # Test conversation logging
        chroma_tests["conversation_logging"] = self._test_conversation_logging()
        
        self.health_report["services"]["chromadb_functionality"] = chroma_tests
        logger.info("ChromaDB functionality test completed")
    
    def _test_collections_info(self) -> Dict[str, Any]:
        """Test collections information retrieval"""
        try:
            response = requests.get(f"{self.chroma_url}/collections", timeout=10)
            if response.status_code == 200:
                collections = response.json()
                return {
                    "status": "success",
                    "collections": collections,
                    "collection_count": len(collections)
                }
            else:
                return {
                    "status": "failed",
                    "response_code": response.status_code
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _test_document_addition(self) -> Dict[str, Any]:
        """Test document addition to ChromaDB"""
        try:
            test_doc_id = f"health_check_{int(time.time())}"
            payload = {
                "documents": ["This is a health check test document"],
                "metadatas": [{"source": "health_check", "timestamp": datetime.now().isoformat()}],
                "ids": [test_doc_id]
            }
            
            response = requests.post(
                f"{self.chroma_url}/collections/knowledge_base/add",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "added_count": result.get("added_count", 0),
                    "total_count": result.get("total_count", 0),
                    "test_doc_id": test_doc_id
                }
            else:
                return {
                    "status": "failed",
                    "response_code": response.status_code,
                    "error": response.text
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _test_document_querying(self) -> Dict[str, Any]:
        """Test document querying from ChromaDB"""
        try:
            payload = {
                "query_texts": ["health check test"],
                "n_results": 5
            }
            
            response = requests.post(
                f"{self.chroma_url}/collections/knowledge_base/query",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                documents = result.get("documents", [[]])
                return {
                    "status": "success",
                    "query_results": len(documents[0]) if documents else 0,
                    "has_results": len(documents[0]) > 0 if documents else False
                }
            else:
                return {
                    "status": "failed",
                    "response_code": response.status_code
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _test_conversation_logging(self) -> Dict[str, Any]:
        """Test conversation logging functionality"""
        try:
            payload = {
                "user_query": "Health check test query",
                "ai_response": "Health check test response",
                "chunks_used": 1
            }
            
            response = requests.post(
                f"{self.chroma_url}/conversations/log",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "logged": True,
                    "collection": result.get("collection", "unknown")
                }
            else:
                return {
                    "status": "failed",
                    "response_code": response.status_code
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _test_document_processing(self):
        """Test document processing for the 3 expected documents"""
        logger.info("Testing document processing for /root/data/ documents...")
        
        processing_results = {
            "expected_documents": self.expected_documents,
            "document_status": {},
            "processing_summary": {}
        }
        
        # Check if documents are processed in ChromaDB
        for doc_name in self.expected_documents:
            processing_results["document_status"][doc_name] = self._check_document_in_db(doc_name)
        
        # Generate processing summary
        processed_count = sum(1 for status in processing_results["document_status"].values() 
                            if status.get("found_in_db", False))
        
        processing_results["processing_summary"] = {
            "total_expected": len(self.expected_documents),
            "found_in_db": processed_count,
            "processing_rate": round((processed_count / len(self.expected_documents)) * 100, 2),
            "all_processed": processed_count == len(self.expected_documents)
        }
        
        self.health_report["document_processing"] = processing_results
        logger.info(f"Document processing check completed: {processed_count}/{len(self.expected_documents)} documents found")
    
    def _check_document_in_db(self, doc_name: str) -> Dict[str, Any]:
        """Check if a specific document is processed and stored in ChromaDB"""
        try:
            # Query for the document by name
            payload = {
                "query_texts": [doc_name],
                "n_results": 10,
                "where": {"source": doc_name}
            }
            
            response = requests.post(
                f"{self.chroma_url}/collections/knowledge_base/query",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                documents = result.get("documents", [[]])
                metadatas = result.get("metadatas", [[]])
                
                found_docs = len(documents[0]) if documents else 0
                
                # Check metadata for source matching
                source_matches = 0
                if metadatas and metadatas[0]:
                    source_matches = sum(1 for meta in metadatas[0] 
                                       if meta and meta.get("source", "").endswith(doc_name))
                
                return {
                    "found_in_db": found_docs > 0 or source_matches > 0,
                    "document_chunks": found_docs,
                    "source_matches": source_matches,
                    "status": "processed" if (found_docs > 0 or source_matches > 0) else "not_found"
                }
            else:
                return {
                    "found_in_db": False,
                    "status": "query_failed",
                    "error": f"Query failed with status {response.status_code}"
                }
        except Exception as e:
            return {
                "found_in_db": False,
                "status": "error",
                "error": str(e)
            }
    
    def _test_database_integrity(self):
        """Test database integrity and consistency"""
        logger.info("Testing database integrity...")
        
        integrity_tests = {}
        
        # Test collection consistency
        integrity_tests["collection_consistency"] = self._test_collection_consistency()
        
        # Test data quality
        integrity_tests["data_quality"] = self._test_data_quality()
        
        self.health_report["services"]["database_integrity"] = integrity_tests
        logger.info("Database integrity test completed")
    
    def _test_collection_consistency(self) -> Dict[str, Any]:
        """Test collection consistency"""
        try:
            collections_response = requests.get(f"{self.chroma_url}/collections", timeout=10)
            if collections_response.status_code != 200:
                return {"status": "failed", "error": "Cannot retrieve collections"}
            
            collections = collections_response.json()
            consistency_results = {}
            
            for collection_name, collection_info in collections.items():
                count = collection_info.get("count", 0)
                consistency_results[collection_name] = {
                    "document_count": count,
                    "has_documents": count > 0,
                    "status": "healthy" if count >= 0 else "error"
                }
            
            return {
                "status": "success",
                "collections": consistency_results,
                "total_collections": len(collections)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _test_data_quality(self) -> Dict[str, Any]:
        """Test data quality in the database"""
        try:
            # Sample some documents from knowledge_base
            payload = {
                "query_texts": ["data science", "machine learning", "AI"],
                "n_results": 5
            }
            
            response = requests.post(
                f"{self.chroma_url}/collections/knowledge_base/query",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                documents = result.get("documents", [[]])
                metadatas = result.get("metadatas", [[]])
                
                if documents and documents[0]:
                    sample_docs = documents[0]
                    sample_metas = metadatas[0] if metadatas else []
                    
                    # Check document quality
                    avg_length = sum(len(doc) for doc in sample_docs) / len(sample_docs)
                    has_metadata = len(sample_metas) > 0
                    
                    return {
                        "status": "success",
                        "sample_size": len(sample_docs),
                        "avg_document_length": round(avg_length, 2),
                        "has_metadata": has_metadata,
                        "quality_score": "good" if avg_length > 50 and has_metadata else "fair"
                    }
                else:
                    return {
                        "status": "warning",
                        "message": "No documents found for quality assessment"
                    }
            else:
                return {
                    "status": "failed",
                    "response_code": response.status_code
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _test_performance(self):
        """Test performance benchmarks"""
        logger.info("Running performance benchmarks...")
        
        performance_tests = {}
        
        # Test embedding performance
        performance_tests["embedding_performance"] = self._benchmark_embedding_performance()
        
        # Test query performance
        performance_tests["query_performance"] = self._benchmark_query_performance()
        
        self.health_report["services"]["performance"] = performance_tests
        logger.info("Performance benchmarks completed")
    
    def _benchmark_embedding_performance(self) -> Dict[str, Any]:
        """Benchmark embedding generation performance"""
        try:
            test_texts = ["Performance test document"] * 50
            
            start_time = time.time()
            payload = {
                "texts": test_texts,
                "model": "default"
            }
            
            response = requests.post(
                f"{self.embedding_url}/embeddings",
                json=payload,
                timeout=60
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                embeddings_count = len(result.get("embeddings", []))
                
                return {
                    "status": "success",
                    "total_texts": len(test_texts),
                    "embeddings_generated": embeddings_count,
                    "total_time_ms": round(total_time * 1000, 2),
                    "avg_time_per_text_ms": round((total_time * 1000) / len(test_texts), 2),
                    "throughput_texts_per_second": round(len(test_texts) / total_time, 2)
                }
            else:
                return {
                    "status": "failed",
                    "response_code": response.status_code
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _benchmark_query_performance(self) -> Dict[str, Any]:
        """Benchmark query performance"""
        try:
            queries = [
                "data science experience",
                "machine learning projects",
                "AI development skills",
                "Python programming",
                "analytics expertise"
            ]
            
            query_times = []
            successful_queries = 0
            
            for query in queries:
                start_time = time.time()
                payload = {
                    "query_texts": [query],
                    "n_results": 5
                }
                
                response = requests.post(
                    f"{self.chroma_url}/collections/knowledge_base/query",
                    json=payload,
                    timeout=15
                )
                
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                if response.status_code == 200:
                    successful_queries += 1
            
            if query_times:
                avg_query_time = sum(query_times) / len(query_times)
                return {
                    "status": "success",
                    "total_queries": len(queries),
                    "successful_queries": successful_queries,
                    "avg_query_time_ms": round(avg_query_time * 1000, 2),
                    "success_rate": round((successful_queries / len(queries)) * 100, 2)
                }
            else:
                return {
                    "status": "failed",
                    "error": "No queries completed"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_overall_assessment(self):
        """Generate overall health assessment and recommendations"""
        logger.info("Generating overall assessment...")
        
        # Collect status indicators
        connectivity_ok = all(
            conn.get("accessible", False) 
            for conn in self.health_report["services"].get("connectivity", {}).values()
        )
        
        health_ok = all(
            health.get("status") == "healthy" 
            for health in self.health_report["services"].get("health", {}).values()
        )
        
        embedding_ok = all(
            test.get("status") == "success" 
            for test in self.health_report["services"].get("embedding_functionality", {}).values()
        )
        
        chromadb_ok = all(
            test.get("status") == "success" 
            for test in self.health_report["services"].get("chromadb_functionality", {}).values()
        )
        
        docs_processed = self.health_report["document_processing"].get("processing_summary", {}).get("all_processed", False)
        
        # Determine overall status
        if connectivity_ok and health_ok and embedding_ok and chromadb_ok and docs_processed:
            self.health_report["overall_status"] = "healthy"
        elif connectivity_ok and health_ok:
            self.health_report["overall_status"] = "partially_healthy"
        else:
            self.health_report["overall_status"] = "unhealthy"
        
        # Generate recommendations
        recommendations = []
        
        if not connectivity_ok:
            recommendations.append("Check network connectivity to VM services")
        
        if not health_ok:
            recommendations.append("Restart unhealthy services on the VM")
        
        if not embedding_ok:
            recommendations.append("Check embedding API configuration and model availability")
        
        if not chromadb_ok:
            recommendations.append("Verify ChromaDB service and database integrity")
        
        if not docs_processed:
            recommendations.append("Process the 3 documents in /root/data/ and add them to the database")
        
        if not recommendations:
            recommendations.append("All systems are functioning normally")
        
        self.health_report["recommendations"] = recommendations
        
        logger.info(f"Overall assessment: {self.health_report['overall_status']}")
    
    def print_health_report(self):
        """Print a formatted health report"""
        print("\n" + "="*60)
        print("           CLOUD SERVICE HEALTH REPORT")
        print("="*60)
        print(f"Timestamp: {self.health_report['timestamp']}")
        print(f"Overall Status: {self.health_report['overall_status'].upper()}")
        print("="*60)
        
        # Service connectivity
        print("\n🔗 SERVICE CONNECTIVITY:")
        connectivity = self.health_report["services"].get("connectivity", {})
        for service, status in connectivity.items():
            icon = "✅" if status.get("accessible") else "❌"
            print(f"  {icon} {service}: {status.get('status', 'unknown')}")
        
        # Service health
        print("\n🏥 SERVICE HEALTH:")
        health = self.health_report["services"].get("health", {})
        for service, status in health.items():
            icon = "✅" if status.get("status") == "healthy" else "❌"
            print(f"  {icon} {status.get('service', service)}: {status.get('status', 'unknown')}")
        
        # Document processing
        print("\n📄 DOCUMENT PROCESSING:")
        doc_summary = self.health_report["document_processing"].get("processing_summary", {})
        processed = doc_summary.get("found_in_db", 0)
        total = doc_summary.get("total_expected", 0)
        rate = doc_summary.get("processing_rate", 0)
        
        icon = "✅" if doc_summary.get("all_processed") else "⚠️"
        print(f"  {icon} Documents processed: {processed}/{total} ({rate}%)")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(self.health_report.get("recommendations", []), 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*60)
        
        # Web service readiness
        if self.health_report["overall_status"] == "healthy":
            print("🚀 READY FOR WEB SERVICE INTEGRATION")
            print("   Your webpage can safely call the Chroma service!")
        else:
            print("⚠️  NOT READY FOR WEB SERVICE")
            print("   Please address the issues above before using the web service.")
        
        print("="*60)

def main():
    """Main function to run health check"""
    print("Starting Cloud Service Health Check...")
    
    # Initialize health checker
    checker = CloudServiceHealthChecker()
    
    # Run comprehensive check
    health_report = checker.run_comprehensive_check()
    
    # Print formatted report
    checker.print_health_report()
    
    # Save detailed report to file
    report_filename = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(health_report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_filename}")
    
    # Return exit code based on health status
    if health_report["overall_status"] == "healthy":
        return 0
    elif health_report["overall_status"] == "partially_healthy":
        return 1
    else:
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
