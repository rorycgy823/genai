#!/bin/bash
# Complete VM Setup Script for Rory's AI Assistant
# Sets up Python environment and installs dependencies
# Author: Rory Chen
# Usage: ./setup_vm.sh

echo "=========================================="
echo "  Rory's AI Assistant VM Setup"
echo "=========================================="
echo "Server: 1.32.228.33"
echo "Setting up Python environment and dependencies..."
echo "=========================================="

# Set working directory
cd /root/code

# Check if we're in the right directory
if [ ! -f "vm_requirements.txt" ]; then
    echo "❌ vm_requirements.txt not found!"
    echo "Please ensure you're in /root/code and all files are uploaded."
    echo "Current directory: $(pwd)"
    echo "Files in directory:"
    ls -la
    exit 1
fi

echo "✅ Found vm_requirements.txt"

# Check Python version
echo "🐍 Checking Python version..."
python3 --version

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf venv
fi

# Create virtual environment
echo "🏗️  Creating Python virtual environment..."
python3 -m venv venv

if [ ! -d "venv" ]; then
    echo "❌ Failed to create virtual environment!"
    exit 1
fi

echo "✅ Virtual environment created"

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing Python packages..."
pip install -r vm_requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements!"
    echo "Check vm_requirements.txt for issues"
    exit 1
fi

echo "✅ All packages installed successfully"

# Create data directory
echo "📁 Creating data directories..."
mkdir -p /root/data/chroma_db
mkdir -p /var/log

echo "✅ Data directories created"

# Set permissions for scripts
echo "🔐 Setting script permissions..."
chmod +x *.sh

echo "✅ Script permissions set"

# Test imports
echo "🧪 Testing Python imports..."
python3 -c "
import sys
sys.path.append('/root/code')

try:
    import chromadb
    print('✅ ChromaDB import successful')
except ImportError as e:
    print(f'❌ ChromaDB import failed: {e}')

try:
    import sentence_transformers
    print('✅ SentenceTransformers import successful')
except ImportError as e:
    print(f'❌ SentenceTransformers import failed: {e}')

try:
    import fastapi
    print('✅ FastAPI import successful')
except ImportError as e:
    print(f'❌ FastAPI import failed: {e}')

try:
    import uvicorn
    print('✅ Uvicorn import successful')
except ImportError as e:
    print(f'❌ Uvicorn import failed: {e}')
"

echo "=========================================="
echo "           SETUP COMPLETE!"
echo "=========================================="
echo "✅ Python virtual environment: /root/code/venv"
echo "✅ Dependencies installed"
echo "✅ Data directories created"
echo "✅ Script permissions set"
echo ""
echo "🚀 Ready to start services!"
echo "Run: ./start_services.sh"
echo "=========================================="

# Show installed packages
echo "📋 Installed packages:"
pip list | grep -E "(chromadb|sentence-transformers|fastapi|uvicorn)"

echo ""
echo "🎯 Next steps:"
echo "1. Run: ./start_services.sh"
echo "2. Check health: curl http://1.32.228.33:8000/health"
echo "3. Check health: curl http://1.32.228.33:8001/health"
