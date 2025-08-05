#!/usr/bin/env python3
"""
Streamlit Deployment Fix Script
Diagnoses and fixes common deployment issues
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run shell command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_port_availability(port):
    """Check if port is available"""
    success, stdout, stderr = run_command(f"ss -tlnp | grep :{port}")
    return "LISTEN" in stdout if success else False

def kill_streamlit_processes():
    """Kill existing Streamlit processes"""
    print("🔄 Killing existing Streamlit processes...")
    run_command("pkill -f streamlit", capture_output=False)
    time.sleep(2)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Check Python packages
    packages = ['streamlit', 'chromadb', 'uvicorn', 'fastapi']
    missing = []
    
    for package in packages:
        success, _, _ = run_command(f"python3 -c 'import {package}'")
        if not success:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {missing}")
        print("📦 Installing missing packages...")
        for package in missing:
            success, _, _ = run_command(f"pip install {package}")
            if success:
                print(f"✅ Installed {package}")
            else:
                print(f"❌ Failed to install {package}")
    else:
        print("✅ All dependencies are installed")

def check_files():
    """Check if required files exist"""
    print("📁 Checking required files...")
    
    required_files = [
        '/root/code/enhanced_streamlit_assistant.py',
        '/root/code/chroma_service.py',
        '/root/code/embedding_api.py',
        '/root/code/process_root_documents.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

def start_streamlit():
    """Start Streamlit application"""
    print("🚀 Starting Streamlit application...")
    
    # Change to code directory
    os.chdir('/root/code')
    
    # Start Streamlit with proper configuration
    cmd = """
    source venv/bin/activate && \
    streamlit run enhanced_streamlit_assistant.py \
    --server.port=8502 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
    """
    
    # Start in background
    subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for service to start
    print("⏳ Waiting for service to start...")
    for i in range(30):  # Wait up to 30 seconds
        time.sleep(1)
        if check_port_availability(8502):
            print("✅ Streamlit is running on port 8502")
            return True
        print(f"⏳ Waiting... ({i+1}/30)")
    
    print("❌ Failed to start Streamlit service")
    return False

def test_connection():
    """Test if the service is accessible"""
    print("🌐 Testing connection...")
    
    try:
        response = requests.get("http://localhost:8502", timeout=10)
        if response.status_code == 200:
            print("✅ Service is accessible locally")
            return True
        else:
            print(f"❌ Service returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def check_firewall():
    """Check firewall settings"""
    print("🔥 Checking firewall settings...")
    
    # Check if ufw is active
    success, stdout, _ = run_command("ufw status")
    if success and "Status: active" in stdout:
        print("🔥 UFW firewall is active")
        # Check if port 8502 is allowed
        if "8502" not in stdout:
            print("🔧 Opening port 8502...")
            run_command("ufw allow 8502")
            print("✅ Port 8502 opened")
        else:
            print("✅ Port 8502 is already allowed")
    else:
        print("✅ UFW firewall is not active")

def main():
    """Main troubleshooting function"""
    print("🔧 Streamlit Deployment Troubleshooting")
    print("=" * 50)
    
    # Step 1: Kill existing processes
    kill_streamlit_processes()
    
    # Step 2: Check dependencies
    check_dependencies()
    
    # Step 3: Check files
    if not check_files():
        print("❌ Cannot proceed without required files")
        return False
    
    # Step 4: Check firewall
    check_firewall()
    
    # Step 5: Start Streamlit
    if not start_streamlit():
        print("❌ Failed to start Streamlit")
        return False
    
    # Step 6: Test connection
    if test_connection():
        print("\n🎉 SUCCESS! Streamlit is running and accessible")
        print("🌐 Access URL: http://1.32.228.33:8502")
        return True
    else:
        print("\n❌ Service started but not accessible")
        print("🔍 Check server firewall and network settings")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
