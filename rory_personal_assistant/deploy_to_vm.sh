#!/bin/bash

# Deploy Streamlit App to VM (1.32.228.33)
# ========================================
# 
# This script deploys the full-featured Streamlit app to your VM
# with all dependencies and proper configuration

echo "🚀 Deploying Rory's AI Assistant to VM..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root or with sudo
if [[ $EUID -eq 0 ]]; then
    print_warning "Running as root. This is fine for VM deployment."
fi

# Update system packages
print_status "Updating system packages..."
apt update && apt upgrade -y

# Install Python 3.10+ and pip if not already installed
print_status "Installing Python and dependencies..."
apt install -y python3 python3-pip python3-venv git curl wget nginx

# Create application directory
APP_DIR="/opt/rory_assistant"
print_status "Creating application directory: $APP_DIR"
mkdir -p $APP_DIR
cd $APP_DIR

# Clone or update the repository
if [ -d ".git" ]; then
    print_status "Updating existing repository..."
    git pull origin main
else
    print_status "Cloning repository..."
    git clone https://github.com/rorycgy823/genai.git .
fi

# Create Python virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies (use the full enhanced version)
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r rory_personal_assistant/enhanced_requirements.txt

# Create systemd service file for Streamlit
print_status "Creating systemd service..."
cat > /etc/systemd/system/rory-assistant.service << EOF
[Unit]
Description=Rory's AI Assistant - Streamlit App
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
Environment=PATH=$APP_DIR/venv/bin
Environment=QWEN_API_KEY=sk-015ea57c8b254c4181d30b2de4259d8b
ExecStart=$APP_DIR/venv/bin/streamlit run rory_personal_assistant/enhanced_streamlit_assistant.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create Nginx configuration for reverse proxy
print_status "Configuring Nginx reverse proxy..."
cat > /etc/nginx/sites-available/rory-assistant << EOF
server {
    listen 80;
    server_name 1.32.228.33;  # Replace with your domain if you have one
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        
        # WebSocket support
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Increase timeout for long-running requests
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
EOF

# Enable Nginx site
ln -sf /etc/nginx/sites-available/rory-assistant /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
print_status "Testing Nginx configuration..."
nginx -t

if [ $? -eq 0 ]; then
    print_success "Nginx configuration is valid"
else
    print_error "Nginx configuration has errors"
    exit 1
fi

# Create startup script
print_status "Creating startup script..."
cat > $APP_DIR/start_web_app.sh << 'EOF'
#!/bin/bash

# Start all services for Rory's AI Assistant
echo "🚀 Starting Rory's AI Assistant services..."

# Start ChromaDB and Embedding services
cd /opt/rory_assistant/rory_personal_assistant
./start_services.sh

# Wait for services to start
sleep 5

# Start Streamlit app service
systemctl start rory-assistant
systemctl enable rory-assistant

# Start Nginx
systemctl start nginx
systemctl enable nginx

echo "✅ All services started!"
echo "🌐 Web app available at: http://1.32.228.33"
echo "📊 ChromaDB API: http://1.32.228.33:8000"
echo "🔗 Embedding API: http://1.32.228.33:8001"

# Show service status
systemctl status rory-assistant --no-pager -l
EOF

chmod +x $APP_DIR/start_web_app.sh

# Create stop script
cat > $APP_DIR/stop_web_app.sh << 'EOF'
#!/bin/bash

# Stop all services for Rory's AI Assistant
echo "🛑 Stopping Rory's AI Assistant services..."

# Stop Streamlit app service
systemctl stop rory-assistant

# Stop ChromaDB and Embedding services
cd /opt/rory_assistant/rory_personal_assistant
./stop_services.sh

# Stop Nginx (optional - comment out if you use Nginx for other sites)
# systemctl stop nginx

echo "✅ All services stopped!"
EOF

chmod +x $APP_DIR/stop_web_app.sh

# Create health check script
cat > $APP_DIR/health_check.sh << 'EOF'
#!/bin/bash

echo "🔍 Health Check - Rory's AI Assistant"
echo "======================================"

# Check Streamlit service
echo "📱 Streamlit App:"
if systemctl is-active --quiet rory-assistant; then
    echo "  ✅ Running"
    echo "  🌐 URL: http://1.32.228.33"
else
    echo "  ❌ Not running"
fi

# Check ChromaDB
echo "🗄️  ChromaDB:"
if curl -s http://localhost:8000/health > /dev/null; then
    echo "  ✅ Running on port 8000"
else
    echo "  ❌ Not responding on port 8000"
fi

# Check Embedding API
echo "🔗 Embedding API:"
if curl -s http://localhost:8001/health > /dev/null; then
    echo "  ✅ Running on port 8001"
else
    echo "  ❌ Not responding on port 8001"
fi

# Check Nginx
echo "🌐 Nginx:"
if systemctl is-active --quiet nginx; then
    echo "  ✅ Running"
else
    echo "  ❌ Not running"
fi

# Check disk space
echo "💾 Disk Space:"
df -h / | tail -1 | awk '{print "  📊 Used: " $3 "/" $2 " (" $5 ")"}'

# Check memory
echo "🧠 Memory:"
free -h | grep Mem | awk '{print "  📊 Used: " $3 "/" $2}'

echo ""
echo "🔗 Quick Links:"
echo "  • Web App: http://1.32.228.33"
echo "  • ChromaDB: http://1.32.228.33:8000"
echo "  • Embedding API: http://1.32.228.33:8001"
EOF

chmod +x $APP_DIR/health_check.sh

# Reload systemd and start services
print_status "Reloading systemd daemon..."
systemctl daemon-reload

# Start the application
print_status "Starting services..."
$APP_DIR/start_web_app.sh

# Final status check
sleep 10
print_status "Final health check..."
$APP_DIR/health_check.sh

print_success "🎉 Deployment completed!"
print_success "🌐 Your AI Assistant is now available at: http://1.32.228.33"
print_success "📚 Management commands:"
print_success "  • Start:  $APP_DIR/start_web_app.sh"
print_success "  • Stop:   $APP_DIR/stop_web_app.sh"
print_success "  • Health: $APP_DIR/health_check.sh"
print_success "  • Logs:   journalctl -u rory-assistant -f"
