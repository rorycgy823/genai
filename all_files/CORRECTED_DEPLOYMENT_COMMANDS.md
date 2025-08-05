# Corrected Deployment Commands
## File Upload and Server Setup

**IMPORTANT:** These commands must be run from your LOCAL machine (Windows), not from the remote server.

## Step 1: Upload Files from Local Machine

### From Windows Command Prompt or PowerShell (in the project directory):

```cmd
# Navigate to your project directory first
cd G:\Data_Science\github_project

# Upload application files to /root/code/
scp -P 64713 rory_personal_assistant\chroma_service.py root@1.32.228.33:/root/code/
scp -P 64713 rory_personal_assistant\embedding_api.py root@1.32.228.33:/root/code/
scp -P 64713 rory_personal_assistant\enhanced_streamlit_assistant.py root@1.32.228.33:/root/code/
scp -P 64713 rory_personal_assistant\process_root_documents.py root@1.32.228.33:/root/code/
scp -P 64713 rory_personal_assistant\remote_config_review.py root@1.32.228.33:/root/code/
scp -P 64713 rory_personal_assistant\vm_requirements.txt root@1.32.228.33:/root/code/
scp -P 64713 rory_personal_assistant\setup_vm.sh root@1.32.228.33:/root/code/
scp -P 64713 rory_personal_assistant\start_services.sh root@1.32.228.33:/root/code/
scp -P 64713 rory_personal_assistant\stop_services.sh root@1.32.228.33:/root/code/

# Create data directory and upload data files
ssh -p 64713 root@1.32.228.33 "mkdir -p /root/data"
scp -P 64713 "rory_personal_assistant\CV_Rory_2507_DS_Analytics.txt" root@1.32.228.33:/root/data/
scp -P 64713 "rory_personal_assistant\CV_Rory_2501.pdf" root@1.32.228.33:/root/data/
scp -P 64713 "rory_personal_assistant\Project experience highlight - Rory.docx" root@1.32.228.33:/root/data/
```

## Step 2: Connect to Server and Setup Environment

```cmd
# Connect to the server
ssh -p 64713 root@1.32.228.33

# Once connected to the server, run these commands:
cd /root/code
chmod +x setup_vm.sh
chmod +x start_services.sh
chmod +x stop_services.sh

# Run the setup script
./setup_vm.sh
```

## Step 3: Start Services

```bash
# On the remote server:
./start_services.sh
```

## Step 4: Process Documents

```bash
# On the remote server:
python3 process_root_documents.py
```

## Step 5: Verify Services

```bash
# Check if services are running:
curl http://localhost:8000/health    # ChromaDB
curl http://localhost:8001/health    # Embedding API
curl http://localhost:8501           # Streamlit (should show HTML)

# Check processes:
ps aux | grep python
netstat -tlnp | grep -E "(8000|8001|8501)"
```

## Alternative: Use the Batch Script

You can also use the pre-configured batch script:

```cmd
# From Windows, in the project directory:
cloud_server_config\file_transfer_tools.bat
```

## Troubleshooting

### If files don't exist locally:
Check which files are actually present:
```cmd
dir rory_personal_assistant\*.py
dir rory_personal_assistant\*.txt
dir rory_personal_assistant\*.docx
dir rory_personal_assistant\*.pdf
```

### If connection fails:
Test the connection first:
```cmd
cloud_server_config\test_connection.bat
```

### If directories don't exist on server:
Create them manually:
```cmd
ssh -p 64713 root@1.32.228.33 "mkdir -p /root/code /root/data /root/logs"
```

## Expected File Structure on Server

After successful upload, the server should have:

```
/root/
├── code/
│   ├── chroma_service.py
│   ├── embedding_api.py
│   ├── enhanced_streamlit_assistant.py
│   ├── process_root_documents.py
│   ├── remote_config_review.py
│   ├── vm_requirements.txt
│   ├── setup_vm.sh
│   ├── start_services.sh
│   ├── stop_services.sh
│   └── venv/ (created by setup_vm.sh)
├── data/
│   ├── CV_Rory_2507_DS_Analytics.txt
│   ├── CV_Rory_2501.pdf
│   ├── Project experience highlight - Rory.docx
│   └── chroma_db/ (created by services)
└── logs/ (created by services)
```

## Service URLs (once running)

- **ChromaDB API:** http://1.32.228.33:8000
- **Embedding API:** http://1.32.228.33:8001  
- **Streamlit Web App:** http://1.32.228.33:8501

The key issue was running the scp commands from the wrong location. These must be run from your local Windows machine, not from the remote server.
