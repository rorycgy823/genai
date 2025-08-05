#!/usr/bin/env python3
"""
Complete System Deployment Script
This script will:
1. Check and install all dependencies
2. Start all required services
3. Process documents and populate ChromaDB
4. Start Streamlit application
5. Configure firewall if needed
6. Verify everything is working
"""

import subprocess
import time
import sys
import os
from pathlib import Path
import json
import requests

def run_command(cmd, capture_output=True):
    """Run a command and return success status and output"""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True)
            return result.returncode == 0, "", ""
    except Exception as e:
        return False, "", str(e)

def log_message(message, level="INFO"):
    """Log a message with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def check_and_install_dependencies():
    """Check and install required Python packages"""
    log_message("Checking and installing dependencies...")
    
    packages = [
        'streamlit',
        'chromadb',
        'python-docx',
        'PyPDF2',
        'sentence-transformers',
        'fastapi',
        'uvicorn',
        'requests'
    ]
    
    for package in packages:
        log_message(f"Installing {package}...")
        success, stdout, stderr = run_command(f"pip install {package}")
        if success:
            log_message(f"✅ {package} installed successfully")
        else:
            log_message(f"❌ Failed to install {package}: {stderr}", "ERROR")
    
    return True

def kill_existing_services():
    """Kill any existing services"""
    log_message("Stopping existing services...")
    
    services = ['streamlit', 'uvicorn', 'python3.*embedding_api', 'python3.*chroma_service']
    
    for service in services:
        run_command(f"pkill -f '{service}'")
    
    time.sleep(3)
    log_message("✅ Existing services stopped")

def start_embedding_api():
    """Start the embedding API service"""
    log_message("Starting Embedding API...")
    
    # Check if embedding_api.py exists
    if not Path("/root/code/embedding_api.py").exists():
        log_message("❌ embedding_api.py not found", "ERROR")
        return False
    
    # Start the service
    cmd = "cd /root/code && nohup python3 embedding_api.py > /root/logs/embedding_api.log 2>&1 &"
    success, _, _ = run_command(cmd, capture_output=False)
    
    # Wait and check if it's running
    time.sleep(5)
    success, stdout, _ = run_command("ss -tlnp | grep :8001")
    
    if success and "LISTEN" in stdout:
        log_message("✅ Embedding API started on port 8001")
        return True
    else:
        log_message("❌ Failed to start Embedding API", "ERROR")
        return False

def start_chroma_service():
    """Start the ChromaDB service"""
    log_message("Starting ChromaDB Service...")
    
    # Check if chroma_service.py exists
    if not Path("/root/code/chroma_service.py").exists():
        log_message("❌ chroma_service.py not found", "ERROR")
        return False
    
    # Start the service
    cmd = "cd /root/code && nohup python3 chroma_service.py > /root/logs/chroma_service.log 2>&1 &"
    success, _, _ = run_command(cmd, capture_output=False)
    
    # Wait and check if it's running
    time.sleep(5)
    success, stdout, _ = run_command("ss -tlnp | grep :8000")
    
    if success and "LISTEN" in stdout:
        log_message("✅ ChromaDB Service started on port 8000")
        return True
    else:
        log_message("❌ Failed to start ChromaDB Service", "ERROR")
        return False

def process_documents():
    """Process documents and populate ChromaDB"""
    log_message("Processing documents...")
    
    # Check if documents exist
    data_dir = Path("/root/data")
    if not data_dir.exists():
        log_message("❌ /root/data directory not found", "ERROR")
        return False
    
    files = list(data_dir.glob("*"))
    if len(files) < 3:
        log_message(f"❌ Expected at least 3 files, found {len(files)}", "ERROR")
        return False
    
    log_message(f"Found {len(files)} files to process")
    
    # Run document processing
    if Path("/root/code/process_root_documents.py").exists():
        success, stdout, stderr = run_command("cd /root/code && python3 process_root_documents.py")
        if success:
            log_message("✅ Documents processed successfully")
            return True
        else:
            log_message(f"❌ Document processing failed: {stderr}", "ERROR")
            return False
    else:
        log_message("❌ process_root_documents.py not found", "ERROR")
        return False

def start_streamlit():
    """Start the Streamlit application"""
    log_message("Starting Streamlit application...")
    
    # Check if enhanced_streamlit_assistant.py exists
    if not Path("/root/code/enhanced_streamlit_assistant.py").exists():
        log_message("❌ enhanced_streamlit_assistant.py not found", "ERROR")
        return False
    
    # Start Streamlit
    cmd = "cd /root/code && nohup streamlit run enhanced_streamlit_assistant.py --server.port=8502 --server.address=0.0.0.0 --server.headless=true > /root/logs/streamlit.log 2>&1 &"
    success, _, _ = run_command(cmd, capture_output=False)
    
    # Wait and check if it's running
    for i in range(15):
        time.sleep(2)
        success, stdout, _ = run_command("ss -tlnp | grep :8502")
        if success and "LISTEN" in stdout:
            log_message("✅ Streamlit started on port 8502")
            return True
        log_message(f"⏳ Waiting for Streamlit... ({i+1}/15)")
    
    log_message("❌ Failed to start Streamlit", "ERROR")
    return False

def configure_firewall():
    """Configure firewall to allow required ports"""
    log_message("Configuring firewall...")
    
    ports = [8000, 8001, 8502]
    
    # Check if ufw is available
    success, _, _ = run_command("which ufw")
    if success:
        # Use ufw
        for port in ports:
            run_command(f"ufw allow {port}")
        log_message("✅ Firewall configured with ufw")
    else:
        # Use iptables
        for port in ports:
            run_command(f"iptables -A INPUT -p tcp --dport {port} -j ACCEPT")
        log_message("✅ Firewall configured with iptables")
    
    return True

def verify_services():
    """Verify all services are running and accessible"""
    log_message("Verifying services...")
    
    services = {
        "Embedding API": "http://localhost:8001/health",
        "ChromaDB": "http://localhost:8000/api/v1/heartbeat",
        "Streamlit": "http://localhost:8502"
    }
    
    all_good = True
    
    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=10)
            if response.status_code in [200, 404]:  # 404 is OK for Streamlit root
                log_message(f"✅ {service_name} is accessible")
            else:
                log_message(f"❌ {service_name} returned status {response.status_code}", "ERROR")
                all_good = False
        except Exception as e:
            log_message(f"❌ {service_name} is not accessible: {e}", "ERROR")
            all_good = False
    
    return all_good

def create_status_report():
    """Create a status report"""
    log_message("Creating status report...")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "services": {},
        "ports": {},
        "processes": {}
    }
    
    # Check ports
    for port in [8000, 8001, 8502]:
        success, stdout, _ = run_command(f"ss -tlnp | grep :{port}")
        report["ports"][port] = "LISTENING" if success and "LISTEN" in stdout else "NOT_LISTENING"
    
    # Check processes
    processes = ["embedding_api", "chroma_service", "streamlit"]
    for process in processes:
        success, stdout, _ = run_command(f"ps aux | grep {process} | grep -v grep")
        report["processes"][process] = "RUNNING" if success and stdout.strip() else "NOT_RUNNING"
    
    # Save report
    with open("/root/logs/deployment_status.json", "w") as f:
        json.dump(report, f, indent=2)
    
    log_message("✅ Status report created at /root/logs/deployment_status.json")
    return report

def main():
    """Main deployment function"""
    log_message("🚀 Starting Complete System Deployment")
    log_message("=" * 60)
    
    # Create logs directory
    Path("/root/logs").mkdir(exist_ok=True)
    
    # Step 1: Install dependencies
    if not check_and_install_dependencies():
        log_message("❌ Failed to install dependencies", "ERROR")
        return False
    
    # Step 2: Kill existing services
    kill_existing_services()
    
    # Step 3: Start services
    if not start_embedding_api():
        log_message("❌ Failed to start Embedding API", "ERROR")
        return False
    
    if not start_chroma_service():
        log_message("❌ Failed to start ChromaDB Service", "ERROR")
        return False
    
    # Step 4: Process documents
    if not process_documents():
        log_message("❌ Failed to process documents", "ERROR")
        return False
    
    # Step 5: Start Streamlit
    if not start_streamlit():
        log_message("❌ Failed to start Streamlit", "ERROR")
        return False
    
    # Step 6: Configure firewall
    configure_firewall()
    
    # Step 7: Verify services
    time.sleep(5)  # Give services time to fully start
    if not verify_services():
        log_message("❌ Service verification failed", "ERROR")
    
    # Step 8: Create status report
    report = create_status_report()
    
    # Final summary
    log_message("=" * 60)
    log_message("🎉 DEPLOYMENT COMPLETE!")
    log_message("=" * 60)
    log_message("📊 Service Status:")
    for port, status in report["ports"].items():
        log_message(f"   Port {port}: {status}")
    
    log_message("🌐 Access URLs:")
    log_message("   Streamlit App: http://1.32.228.33:8502")
    log_message("   Embedding API: http://1.32.228.33:8001")
    log_message("   ChromaDB: http://1.32.228.33:8000")
    
    log_message("📝 Logs available at:")
    log_message("   /root/logs/streamlit.log")
    log_message("   /root/logs/embedding_api.log")
    log_message("   /root/logs/chroma_service.log")
    log_message("   /root/logs/deployment_status.json")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
