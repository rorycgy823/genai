#!/bin/bash

# 🚀 Deploy Streamlit App to Private Cloud VM
# Usage: ./deploy_streamlit_to_vm.sh

echo "🚀 Deploying Streamlit App to Private Cloud VM (1.32.228.33)..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VM_IP="1.32.228.33"
APP_DIR="/root/rory_assistant"
APP_FILE="enhanced_streamlit_assistant.py"
REQUIREMENTS_FILE="enhanced_requirements.txt"
SERVICE_NAME="streamlit-assistant"

echo -e "${BLUE}📋 Deployment Configuration:${NC}"
echo -e "   VM IP: ${VM_IP}"
echo -e "   App Directory: ${APP_DIR}"
echo -e "   App File: ${APP_FILE}"
echo -e "   Service: ${SERVICE_NAME}"
echo ""

# Check if we can connect to VM
echo -e "${YELLOW}🔍 Checking VM connectivity...${NC}"
if ping -c 1 ${VM_IP} &> /dev/null; then
    echo -e "${GREEN}✅ VM is reachable${NC}"
else
    echo -e "${RED}❌ Cannot reach VM ${VM_IP}${NC}"
    exit 1
fi

# Check if required files exist locally
echo -e "${YELLOW}📁 Checking local files...${NC}"
if [[ -f "${APP_FILE}" ]]; then
    echo -e "${GREEN}✅ Found ${APP_FILE}${NC}"
else
    echo -e "${RED}❌ ${APP_FILE} not found in current directory${NC}"
    exit 1
fi

if [[ -f "${REQUIREMENTS_FILE}" ]]; then
    echo -e "${GREEN}✅ Found ${REQUIREMENTS_FILE}${NC}"
else
    echo -e "${RED}❌ ${REQUIREMENTS_FILE} not found in current directory${NC}"
    exit 1
fi

# Function to run commands on VM
run_on_vm() {
    ssh root@${VM_IP} "$1"
}

# Function to copy files to VM
copy_to_vm() {
    scp "$1" root@${VM_IP}:"$2"
}

echo ""
echo -e "${YELLOW}📦 Installing dependencies on VM...${NC}"
run_on_vm "pip install streamlit requests pandas plotly networkx python-dateutil"

echo -e "${YELLOW}📂 Creating app directory on VM...${NC}"
run_on_vm "mkdir -p ${APP_DIR}"

echo -e "${YELLOW}📤 Copying files to VM...${NC}"
copy_to_vm "${APP_FILE}" "${APP_DIR}/"
copy_to_vm "${REQUIREMENTS_FILE}" "${APP_DIR}/"

echo -e "${YELLOW}🔧 Installing app requirements on VM...${NC}"
run_on_vm "cd ${APP_DIR} && pip install -r ${REQUIREMENTS_FILE}"

echo -e "${YELLOW}🔥 Configuring firewall...${NC}"
run_on_vm "ufw allow 8501/tcp"

echo -e "${YELLOW}⚙️ Creating systemd service...${NC}"
run_on_vm "cat > /etc/systemd/system/${SERVICE_NAME}.service << 'EOF'
[Unit]
Description=Rory's AI Assistant - Streamlit App
After=network.target
Requires=chroma-service.service embedding-api.service

[Service]
Type=simple
User=root
WorkingDirectory=${APP_DIR}
Environment=QWEN_API_KEY=sk-015ea57c8b254c4181d30b2de4259d8b
Environment=PYTHONPATH=${APP_DIR}
ExecStart=/usr/local/bin/streamlit run ${APP_FILE} --server.port 8501 --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF"

echo -e "${YELLOW}🔄 Enabling and starting service...${NC}"
run_on_vm "systemctl daemon-reload"
run_on_vm "systemctl enable ${SERVICE_NAME}"
run_on_vm "systemctl start ${SERVICE_NAME}"

# Wait a moment for service to start
sleep 5

echo -e "${YELLOW}📊 Checking service status...${NC}"
SERVICE_STATUS=$(run_on_vm "systemctl is-active ${SERVICE_NAME}")

if [[ "${SERVICE_STATUS}" == "active" ]]; then
    echo -e "${GREEN}✅ Service is running successfully!${NC}"
    echo ""
    echo -e "${BLUE}🌐 Your Streamlit app is now available at:${NC}"
    echo -e "${GREEN}   http://${VM_IP}:8501${NC}"
    echo ""
    echo -e "${BLUE}📋 Service Management Commands:${NC}"
    echo -e "   Check status: ssh root@${VM_IP} 'systemctl status ${SERVICE_NAME}'"
    echo -e "   View logs:    ssh root@${VM_IP} 'journalctl -u ${SERVICE_NAME} -f'"
    echo -e "   Restart:      ssh root@${VM_IP} 'systemctl restart ${SERVICE_NAME}'"
    echo -e "   Stop:         ssh root@${VM_IP} 'systemctl stop ${SERVICE_NAME}'"
    echo ""
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
else
    echo -e "${RED}❌ Service failed to start${NC}"
    echo -e "${YELLOW}📋 Checking logs...${NC}"
    run_on_vm "journalctl -u ${SERVICE_NAME} --no-pager -n 20"
    exit 1
fi

echo ""
echo -e "${BLUE}🔍 Final System Check:${NC}"
echo -e "${YELLOW}Checking all services...${NC}"
run_on_vm "systemctl status chroma-service --no-pager -l"
run_on_vm "systemctl status embedding-api --no-pager -l"
run_on_vm "systemctl status ${SERVICE_NAME} --no-pager -l"

echo ""
echo -e "${GREEN}✅ All done! Your enhanced AI assistant is now running on your private cloud!${NC}"
