#!/bin/bash
# Stop script for Rory's AI Assistant Services
# Stops ChromaDB and Embedding API services gracefully
# Author: Rory Chen
# Usage: ./stop_services.sh

echo "=========================================="
echo "  Rory's AI Assistant Services Shutdown"
echo "=========================================="

# Function to stop service by PID
stop_service() {
    local service_name=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "🛑 Stopping $service_name (PID: $pid)..."
            kill $pid
            
            # Wait for graceful shutdown
            local count=0
            while ps -p $pid > /dev/null 2>&1 && [ $count -lt 10 ]; do
                sleep 1
                ((count++))
            done
            
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo "⚠️  Force killing $service_name..."
                kill -9 $pid
            fi
            
            echo "✅ $service_name stopped"
        else
            echo "⚠️  $service_name was not running"
        fi
        rm -f "$pid_file"
    else
        echo "⚠️  No PID file found for $service_name"
    fi
}

# Stop services using PID files
stop_service "ChromaDB Service" "/tmp/chroma_service.pid"
stop_service "Embedding API Service" "/tmp/embedding_service.pid"

# Also kill any remaining python processes on ports 8000 and 8001
echo "🔍 Checking for remaining processes on ports 8000 and 8001..."

if lsof -ti:8000 >/dev/null 2>&1; then
    echo "🛑 Killing remaining processes on port 8000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
fi

if lsof -ti:8001 >/dev/null 2>&1; then
    echo "🛑 Killing remaining processes on port 8001..."
    lsof -ti:8001 | xargs kill -9 2>/dev/null || true
fi

# Verify ports are free
sleep 2
echo "🔍 Verifying ports are free..."

if ! lsof -ti:8000 >/dev/null 2>&1; then
    echo "✅ Port 8000 is free"
else
    echo "❌ Port 8000 is still in use"
fi

if ! lsof -ti:8001 >/dev/null 2>&1; then
    echo "✅ Port 8001 is free"
else
    echo "❌ Port 8001 is still in use"
fi

echo "=========================================="
echo "🎯 All services stopped successfully!"
echo "=========================================="
echo "To restart services, run:"
echo "  ./start_services.sh"
echo "=========================================="
