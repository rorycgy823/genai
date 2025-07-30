# VM Web Deployment Guide - Rory's AI Assistant

## 🚀 Deploy Your AI Assistant to Your Own Cloud VM

Since Streamlit Cloud has too many dependency restrictions, hosting on your own VM (1.32.228.33) is the perfect solution!

## ✅ **Advantages of VM Deployment**

### **🎯 Full Control**
- ✅ No dependency restrictions
- ✅ All libraries available (NetworkX, Plotly, ChromaDB, etc.)
- ✅ Custom configurations
- ✅ Better performance

### **🔧 Professional Setup**
- ✅ Nginx reverse proxy
- ✅ Systemd service management
- ✅ Auto-restart on failure
- ✅ Production-ready configuration

### **📊 Complete Features**
- ✅ Full GraphRAG with NetworkX
- ✅ Interactive Plotly visualizations
- ✅ Cost analytics dashboard
- ✅ ChromaDB integration
- ✅ Document processing

## 📋 **Deployment Steps**

### **Step 1: Connect to Your VM**
```bash
# SSH into your VM
ssh root@1.32.228.33
```

### **Step 2: Run the Deployment Script**
```bash
# Download and run the deployment script
curl -sSL https://raw.githubusercontent.com/rorycgy823/genai/main/rory_personal_assistant/deploy_to_vm.sh | bash

# Or if you prefer to download first:
wget https://raw.githubusercontent.com/rorycgy823/genai/main/rory_personal_assistant/deploy_to_vm.sh
chmod +x deploy_to_vm.sh
./deploy_to_vm.sh
```

### **Step 3: Verify Deployment**
The script will automatically:
1. ✅ Install all dependencies
2. ✅ Clone your GitHub repository
3. ✅ Set up Python virtual environment
4. ✅ Configure Nginx reverse proxy
5. ✅ Create systemd services
6. ✅ Start all services
7. ✅ Run health checks

## 🌐 **Access Your Web App**

After deployment, your AI assistant will be available at:
- **🌐 Main Web App**: `http://1.32.228.33`
- **📊 ChromaDB API**: `http://1.32.228.33:8000`
- **🔗 Embedding API**: `http://1.32.228.33:8001`

## 🛠️ **Service Management**

### **Start Services**
```bash
cd /opt/rory_assistant
./start_web_app.sh
```

### **Stop Services**
```bash
cd /opt/rory_assistant
./stop_web_app.sh
```

### **Health Check**
```bash
cd /opt/rory_assistant
./health_check.sh
```

### **View Logs**
```bash
# Streamlit app logs
journalctl -u rory-assistant -f

# Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

## 📊 **Expected Output After Deployment**

```
🔍 Health Check - Rory's AI Assistant
======================================
📱 Streamlit App:
  ✅ Running
  🌐 URL: http://1.32.228.33
🗄️  ChromaDB:
  ✅ Running on port 8000
🔗 Embedding API:
  ✅ Running on port 8001
🌐 Nginx:
  ✅ Running
💾 Disk Space:
  📊 Used: 2.1G/20G (11%)
🧠 Memory:
  📊 Used: 1.2G/4.0G

🔗 Quick Links:
  • Web App: http://1.32.228.33
  • ChromaDB: http://1.32.228.33:8000
  • Embedding API: http://1.32.228.33:8001
```

## 🔧 **Architecture Overview**

```
Internet → Nginx (Port 80) → Streamlit (Port 8501)
                          ↓
                    ChromaDB (Port 8000)
                          ↓
                   Embedding API (Port 8001)
                          ↓
                   Documents (/root/data/)
```

## 🎯 **Features Available**

### **✅ Full-Featured Web Interface**
- 💬 Interactive chat with AI assistant
- 📊 Real-time cost tracking and analytics
- 🧠 GraphRAG entity extraction with NetworkX
- 📈 Interactive Plotly visualizations
- 🎨 Professional UI with responsive design

### **✅ Backend Services**
- 🗄️ ChromaDB vector database
- 🔗 Embedding API for document processing
- 📄 Document processing from `/root/data/`
- 💰 Cost calculation and tracking

### **✅ Production Features**
- 🔄 Auto-restart on failure
- 📝 Comprehensive logging
- 🔍 Health monitoring
- 🌐 Nginx reverse proxy
- 🔒 Systemd service management

## 🔍 **Verify Document Processing**

### **Check Documents in /root/data/**
```bash
ls -la /root/data/
# Should show:
# CV_Rory_2501.pdf
# CV_Rory_2507_DS_Analytics.txt  
# Project experience highlight - Rory.docx
```

### **Test Document Processing**
```bash
cd /opt/rory_assistant/rory_personal_assistant
python process_root_documents.py
```

### **Check ChromaDB Collection**
```bash
curl http://localhost:8000/collections
# Should show knowledge_base collection with documents
```

## 🚀 **Performance Optimization**

### **For Better Performance**
```bash
# Increase Nginx worker processes
sudo nano /etc/nginx/nginx.conf
# Set: worker_processes auto;

# Restart Nginx
sudo systemctl restart nginx
```

### **Monitor Resource Usage**
```bash
# Check CPU and memory
htop

# Check disk usage
df -h

# Check network connections
netstat -tulpn | grep -E "(8000|8001|8501|80)"
```

## 🔒 **Security Considerations**

### **Firewall Configuration**
```bash
# Allow HTTP traffic
ufw allow 80/tcp

# Allow SSH (if not already allowed)
ufw allow 22/tcp

# Enable firewall
ufw enable
```

### **SSL Certificate (Optional)**
If you have a domain name, you can add SSL:
```bash
# Install Certbot
apt install certbot python3-certbot-nginx

# Get SSL certificate
certbot --nginx -d yourdomain.com
```

## 🎉 **Success Indicators**

After successful deployment, you should see:

1. **✅ Web App Running**: Visit `http://1.32.228.33` and see the AI assistant interface
2. **✅ Chat Working**: Ask questions and get responses with cost tracking
3. **✅ Analytics Dashboard**: View cost analytics with interactive charts
4. **✅ Document Retrieval**: Questions answered using your documents from `/root/data/`
5. **✅ GraphRAG Active**: Entity extraction working with NetworkX
6. **✅ All Services Healthy**: Health check shows all green checkmarks

## 📞 **Support & Troubleshooting**

### **Common Issues**

**Issue**: Web app not accessible
```bash
# Check if services are running
systemctl status rory-assistant
systemctl status nginx

# Check ports
netstat -tulpn | grep -E "(80|8501)"
```

**Issue**: ChromaDB connection failed
```bash
# Restart ChromaDB service
cd /opt/rory_assistant/rory_personal_assistant
./stop_services.sh
./start_services.sh
```

**Issue**: Documents not found
```bash
# Verify documents exist
ls -la /root/data/

# Reprocess documents
cd /opt/rory_assistant/rory_personal_assistant
python process_root_documents.py
```

### **Contact Information**
- **Email**: chengy823@gmail.com
- **GitHub**: https://github.com/rorycgy823/genai
- **VM**: 1.32.228.33 (Ubuntu-22.04-x64)

---

## 🎊 **Congratulations!**

Your AI Assistant is now running on your own cloud infrastructure with:
- ✅ **Full-featured web interface** at `http://1.32.228.33`
- ✅ **No dependency restrictions** - all libraries working
- ✅ **Professional production setup** with Nginx + systemd
- ✅ **Complete GraphRAG and analytics** with interactive visualizations
- ✅ **Automatic service management** and health monitoring
- ✅ **Document processing** from your `/root/data/` folder
- ✅ **Cost tracking and analytics** dashboard

**🌟 This is a much better solution than Streamlit Cloud!**

### **Next Steps:**
1. **Test the web interface**: Visit `http://1.32.228.33`
2. **Ask sample questions** to verify document retrieval
3. **Check cost analytics** in the dashboard
4. **Monitor services** with the health check script
5. **Optional**: Set up a domain name and SSL certificate

### **Key Benefits Achieved:**
- 🚀 **Better Performance**: No cloud platform limitations
- 🔧 **Full Control**: Complete access to all features
- 💰 **Cost Effective**: Use your existing VM infrastructure
- 🛡️ **Reliable**: Production-grade setup with auto-restart
- 📊 **Complete Analytics**: Full Plotly visualizations working
- 🧠 **Advanced AI**: GraphRAG with NetworkX fully functional

**Your AI Assistant is now production-ready and accessible to anyone on the internet!**
