#!/bin/bash
# Auto-trigger script for Rory's AI Assistant Services
# Starts ChromaDB and Embedding API services simultaneously
# Author: Rory Chen
# Usage: ./start_services.sh

echo "=========================================="
echo "  Rory's AI Assistant Services Startup"
echo "=========================================="
echo "Server: 1.32.228.33"
echo "ChromaDB Service: Port 8000"
echo "Embedding API: Port 8001"
echo "=========================================="

# Set working directory
cd /root

# Check if virtual environment exists
if [ ! -d "/root/code/venv" ]; then
    echo "❌ Virtual environment not found at /root/code/venv"
    echo "Please run setup first:"
    echo "  cd /root/code"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r vm_requirements.txt"
    exit 1
fi

# Check if service files exist
if [ ! -f "/root/code/chroma_service.py" ]; then
    echo "❌ chroma_service.py not found in /root/code/"
    exit 1
fi

if [ ! -f "/root/code/embedding_api.py" ]; then
    echo "❌ embedding_api.py not found in /root/code/"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p /root/data/chroma_db
echo "✅ Data directory ready: /root/data/chroma_db"

# Create log directory
mkdir -p /var/log
echo "✅ Log directory ready: /var/log"

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $port is already in use"
        echo "Killing existing process on port $port..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

# Check and free ports
echo "🔍 Checking ports..."
check_port 8000
check_port 8001

# Activate virtual environment
echo "🐍 Activating Python virtual environment..."
source /root/code/venv/bin/activate

# Start ChromaDB Service in background
echo "🚀 Starting ChromaDB Service (Port 8000)..."
nohup python3 /root/code/chroma_service.py > /var/log/chroma_service.log 2>&1 &
CHROMA_PID=$!
echo "ChromaDB Service PID: $CHROMA_PID"

# Wait a moment for ChromaDB to start
sleep 3

# Start Embedding API Service in background
echo "🚀 Starting Embedding API Service (Port 8001)..."
nohup python3 /root/code/embedding_api.py > /var/log/embedding_service.log 2>&1 &
EMBEDDING_PID=$!
echo "Embedding API Service PID: $EMBEDDING_PID"

# Wait for services to initialize
echo "⏳ Waiting for services to initialize..."
sleep 5

# Health check function
health_check() {
    local service_name=$1
    local url=$2
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo "✅ $service_name is healthy"
            return 0
        else
            echo "⏳ $service_name health check attempt $attempt/$max_attempts..."
            sleep 2
            ((attempt++))
        fi
    done
    
    echo "❌ $service_name failed health check"
    return 1
}

# Perform health checks
echo "🏥 Performing health checks..."
health_check "ChromaDB Service" "http://localhost:8000/health"
CHROMA_HEALTH=$?

health_check "Embedding API Service" "http://localhost:8001/health"
EMBEDDING_HEALTH=$?

# Display status
echo "=========================================="
echo "           SERVICE STATUS"
echo "=========================================="

if [ $CHROMA_HEALTH -eq 0 ]; then
    echo "✅ ChromaDB Service: RUNNING (PID: $CHROMA_PID)"
    echo "   URL: http://1.32.228.33:8000"
    echo "   Health: http://1.32.228.33:8000/health"
else
    echo "❌ ChromaDB Service: FAILED"
fi

if [ $EMBEDDING_HEALTH -eq 0 ]; then
    echo "✅ Embedding API Service: RUNNING (PID: $EMBEDDING_PID)"
    echo "   URL: http://1.32.228.33:8001"
    echo "   Health: http://1.32.228.33:8001/health"
else
    echo "❌ Embedding API Service: FAILED"
fi

echo "=========================================="
echo "           MANAGEMENT COMMANDS"
echo "=========================================="
echo "View logs:"
echo "  tail -f /var/log/chroma_service.log"
echo "  tail -f /var/log/embedding_service.log"
echo ""
echo "Stop services:"
echo "  kill $CHROMA_PID $EMBEDDING_PID"
echo "  # Or use: ./stop_services.sh"
echo ""
echo "Check processes:"
echo "  ps aux | grep python3"
echo ""
echo "Check ports:"
echo "  netstat -tlnp | grep :800"
echo "=========================================="

# Save PIDs for stop script
echo "$CHROMA_PID" > /tmp/chroma_service.pid
echo "$EMBEDDING_PID" > /tmp/embedding_service.pid

# Final status
if [ $CHROMA_HEALTH -eq 0 ] && [ $EMBEDDING_HEALTH -eq 0 ]; then
    echo "🎉 All services started successfully!"
    echo "Your AI Assistant is ready to use!"
    exit 0
else
    echo "⚠️  Some services failed to start. Check logs for details."
    exit 1
fi
