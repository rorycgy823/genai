#!/usr/bin/env python3
"""
Remote Server Configuration Review Script
========================================

This script reviews the remote server configuration for:
1. ChromaDB service setup
2. Embedding API configuration
3. Data files presence
4. Environment setup
5. Service health checks

Author: Rory Chen
Usage: python remote_config_review.py
"""

import os
import sys
import json
import requests
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RemoteConfigReviewer:
    """Review remote server configuration for AI Assistant deployment"""
    
    def __init__(self):
        self.server_ip = "1.32.228.33"
        self.chroma_port = 8000
        self.embedding_port = 8001
        self.streamlit_port = 8501
        
        self.review_results = {
            "timestamp": datetime.now().isoformat(),
            "server_info": {},
            "directory_structure": {},
            "service_status": {},
            "configuration_issues": [],
            "recommendations": []
        }
    
    def run_full_review(self):
        """Run complete configuration review"""
        logger.info("Starting remote server configuration review...")
        
        try:
            # 1. Check directory structure
            self._check_directory_structure()
            
            # 2. Check required files
            self._check_required_files()
            
            # 3. Check Python environment
            self._check_python_environment()
            
            # 4. Check service configurations
            self._check_service_configurations()
            
            # 5. Check data files
            self._check_data_files()
            
            # 6. Test service connectivity
            self._test_service_connectivity()
            
            # 7. Generate recommendations
            self._generate_recommendations()
            
            logger.info("Configuration review completed")
            return self.review_results
            
        except Exception as e:
            logger.error(f"Configuration review failed: {str(e)}")
            self.review_results["error"] = str(e)
            return self.review_results
    
    def _check_directory_structure(self):
        """Check expected directory structure"""
        logger.info("Checking directory structure...")
        
        expected_dirs = [
            "/root/code",
            "/root/data",
            "/root/data/chroma_db",
            "/var/log"
        ]
        
        structure_check = {}
        
        for dir_path in expected_dirs:
            exists = os.path.exists(dir_path)
            structure_check[dir_path] = {
                "exists": exists,
                "is_directory": os.path.isdir(dir_path) if exists else False,
                "permissions": oct(os.stat(dir_path).st_mode)[-3:] if exists else None
            }
            
            if not exists:
                self.review_results["configuration_issues"].append(
                    f"Missing directory: {dir_path}"
                )
        
        self.review_results["directory_structure"] = structure_check
    
    def _check_required_files(self):
        """Check for required files"""
        logger.info("Checking required files...")
        
        required_files = {
            "/root/code/chroma_service.py": "ChromaDB service",
            "/root/code/embedding_api.py": "Embedding API service",
            "/root/code/enhanced_streamlit_assistant.py": "Streamlit application",
            "/root/code/vm_requirements.txt": "Python requirements",
            "/root/code/setup_vm.sh": "VM setup script",
            "/root/code/start_services.sh": "Service startup script",
            "/root/code/stop_services.sh": "Service stop script"
        }
        
        file_check = {}
        
        for file_path, description in required_files.items():
            exists = os.path.exists(file_path)
            file_check[file_path] = {
                "exists": exists,
                "description": description,
                "size": os.path.getsize(file_path) if exists else 0,
                "executable": os.access(file_path, os.X_OK) if exists else False
            }
            
            if not exists:
                self.review_results["configuration_issues"].append(
                    f"Missing file: {file_path} ({description})"
                )
        
        self.review_results["required_files"] = file_check
    
    def _check_python_environment(self):
        """Check Python environment and packages"""
        logger.info("Checking Python environment...")
        
        try:
            # Check Python version
            python_version = subprocess.check_output(
                ["python3", "--version"], 
                text=True
            ).strip()
            
            # Check virtual environment
            venv_path = "/root/code/venv"
            venv_exists = os.path.exists(venv_path)
            
            # Check pip packages if venv exists
            installed_packages = {}
            if venv_exists:
                try:
                    pip_list = subprocess.check_output(
                        [f"{venv_path}/bin/pip", "list", "--format=json"],
                        text=True
                    )
                    packages = json.loads(pip_list)
                    installed_packages = {pkg["name"]: pkg["version"] for pkg in packages}
                except Exception as e:
                    logger.warning(f"Could not get pip list: {e}")
            
            # Check critical packages
            critical_packages = [
                "fastapi", "uvicorn", "chromadb", "sentence-transformers",
                "streamlit", "requests", "pandas"
            ]
            
            missing_packages = []
            for package in critical_packages:
                if package not in installed_packages:
                    missing_packages.append(package)
            
            self.review_results["python_environment"] = {
                "python_version": python_version,
                "venv_exists": venv_exists,
                "venv_path": venv_path,
                "installed_packages": installed_packages,
                "missing_packages": missing_packages
            }
            
            if missing_packages:
                self.review_results["configuration_issues"].append(
                    f"Missing Python packages: {', '.join(missing_packages)}"
                )
                
        except Exception as e:
            logger.error(f"Python environment check failed: {e}")
            self.review_results["configuration_issues"].append(
                f"Python environment check failed: {str(e)}"
            )
    
    def _check_service_configurations(self):
        """Check service configuration files"""
        logger.info("Checking service configurations...")
        
        service_configs = {}
        
        # Check ChromaDB service
        chroma_file = "/root/code/chroma_service.py"
        if os.path.exists(chroma_file):
            with open(chroma_file, 'r') as f:
                content = f.read()
                service_configs["chroma_service"] = {
                    "file_exists": True,
                    "has_fastapi": "FastAPI" in content,
                    "has_chromadb": "chromadb" in content,
                    "has_port_config": "8000" in content,
                    "content_length": len(content)
                }
        
        # Check Embedding API
        embedding_file = "/root/code/embedding_api.py"
        if os.path.exists(embedding_file):
            with open(embedding_file, 'r') as f:
                content = f.read()
                service_configs["embedding_api"] = {
                    "file_exists": True,
                    "has_fastapi": "FastAPI" in content,
                    "has_sentence_transformers": "sentence_transformers" in content,
                    "has_port_config": "8001" in content,
                    "content_length": len(content)
                }
        
        # Check Streamlit app
        streamlit_file = "/root/code/enhanced_streamlit_assistant.py"
        if os.path.exists(streamlit_file):
            with open(streamlit_file, 'r') as f:
                content = f.read()
                service_configs["streamlit_app"] = {
                    "file_exists": True,
                    "has_streamlit": "streamlit" in content,
                    "has_requests": "requests" in content,
                    "has_chroma_config": "8000" in content,
                    "has_embedding_config": "8001" in content,
                    "content_length": len(content)
                }
        
        self.review_results["service_configurations"] = service_configs
    
    def _check_data_files(self):
        """Check data files that need to be processed"""
        logger.info("Checking data files...")
        
        expected_data_files = [
            "/root/data/CV_Rory_2507_DS_Analytics.txt",
            "/root/data/Project experience highlight - Rory.docx",
            "/root/data/CV_Rory_2501.pdf"
        ]
        
        data_files_check = {}
        
        for file_path in expected_data_files:
            exists = os.path.exists(file_path)
            data_files_check[file_path] = {
                "exists": exists,
                "size": os.path.getsize(file_path) if exists else 0,
                "readable": os.access(file_path, os.R_OK) if exists else False
            }
            
            if not exists:
                self.review_results["configuration_issues"].append(
                    f"Missing data file: {file_path}"
                )
        
        self.review_results["data_files"] = data_files_check
    
    def _test_service_connectivity(self):
        """Test service connectivity"""
        logger.info("Testing service connectivity...")
        
        services = {
            "chroma_service": f"http://{self.server_ip}:{self.chroma_port}/health",
            "embedding_api": f"http://{self.server_ip}:{self.embedding_port}/health",
            "streamlit_app": f"http://{self.server_ip}:{self.streamlit_port}"
        }
        
        connectivity_results = {}
        
        for service_name, url in services.items():
            try:
                response = requests.get(url, timeout=10)
                connectivity_results[service_name] = {
                    "url": url,
                    "accessible": True,
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
            except requests.exceptions.RequestException as e:
                connectivity_results[service_name] = {
                    "url": url,
                    "accessible": False,
                    "error": str(e)
                }
        
        self.review_results["service_connectivity"] = connectivity_results
    
    def _generate_recommendations(self):
        """Generate recommendations based on review results"""
        logger.info("Generating recommendations...")
        
        recommendations = []
        
        # Check for missing directories
        if self.review_results.get("configuration_issues"):
            recommendations.append("Fix configuration issues listed above")
        
        # Check Python environment
        python_env = self.review_results.get("python_environment", {})
        if python_env.get("missing_packages"):
            recommendations.append(
                "Install missing Python packages using: pip install -r vm_requirements.txt"
            )
        
        # Check service connectivity
        connectivity = self.review_results.get("service_connectivity", {})
        for service, status in connectivity.items():
            if not status.get("accessible"):
                recommendations.append(f"Start {service} service")
        
        # Check data files
        data_files = self.review_results.get("data_files", {})
        missing_data = [path for path, info in data_files.items() if not info.get("exists")]
        if missing_data:
            recommendations.append("Upload missing data files to /root/data/")
        
        self.review_results["recommendations"] = recommendations
    
    def print_review_report(self):
        """Print formatted review report"""
        print("\n" + "="*70)
        print("         REMOTE SERVER CONFIGURATION REVIEW")
        print("="*70)
        print(f"Timestamp: {self.review_results['timestamp']}")
        print(f"Server: {self.server_ip}")
        print("="*70)
        
        # Configuration Issues
        issues = self.review_results.get("configuration_issues", [])
        if issues:
            print("\n❌ CONFIGURATION ISSUES:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\n✅ NO CONFIGURATION ISSUES FOUND")
        
        # Directory Structure
        print("\n📁 DIRECTORY STRUCTURE:")
        dirs = self.review_results.get("directory_structure", {})
        for dir_path, info in dirs.items():
            status = "✅" if info["exists"] else "❌"
            print(f"  {status} {dir_path}")
        
        # Required Files
        print("\n📄 REQUIRED FILES:")
        files = self.review_results.get("required_files", {})
        for file_path, info in files.items():
            status = "✅" if info["exists"] else "❌"
            size = f"({info['size']} bytes)" if info["exists"] else ""
            print(f"  {status} {file_path} {size}")
        
        # Python Environment
        print("\n🐍 PYTHON ENVIRONMENT:")
        python_env = self.review_results.get("python_environment", {})
        print(f"  Python Version: {python_env.get('python_version', 'Unknown')}")
        print(f"  Virtual Environment: {'✅' if python_env.get('venv_exists') else '❌'}")
        
        missing_packages = python_env.get("missing_packages", [])
        if missing_packages:
            print(f"  Missing Packages: {', '.join(missing_packages)}")
        
        # Service Connectivity
        print("\n🌐 SERVICE CONNECTIVITY:")
        connectivity = self.review_results.get("service_connectivity", {})
        for service, status in connectivity.items():
            accessible = "✅" if status.get("accessible") else "❌"
            url = status.get("url", "")
            print(f"  {accessible} {service}: {url}")
        
        # Data Files
        print("\n📊 DATA FILES:")
        data_files = self.review_results.get("data_files", {})
        for file_path, info in data_files.items():
            status = "✅" if info["exists"] else "❌"
            size = f"({info['size']} bytes)" if info["exists"] else ""
            print(f"  {status} {os.path.basename(file_path)} {size}")
        
        # Recommendations
        recommendations = self.review_results.get("recommendations", [])
        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*70)
        
        # Overall Status
        if not issues and not recommendations:
            print("🚀 CONFIGURATION READY FOR DEPLOYMENT!")
        else:
            print("⚠️  CONFIGURATION NEEDS ATTENTION")
            print("   Please address the issues and recommendations above.")
        
        print("="*70)

def main():
    """Main function"""
    print("Starting Remote Server Configuration Review...")
    
    reviewer = RemoteConfigReviewer()
    results = reviewer.run_full_review()
    
    # Print report
    reviewer.print_review_report()
    
    # Save results
    results_file = f"config_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Return exit code
    if results.get("configuration_issues") or results.get("recommendations"):
        return 1
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
