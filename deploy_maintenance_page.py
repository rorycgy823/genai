#!/usr/bin/env python3
"""
Deploy Maintenance Page
======================

Deploy a simple maintenance page to show users that the GraphRAG Assistant
is temporarily under maintenance and will be back soon.

Author: Rory Chen
"""

import paramiko
import time
from datetime import datetime

def run_ssh_command(command, timeout=60):
    """Execute command via SSH"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('1.32.228.33', port=64713, username='root', password='nJPoZDv0JBw2', timeout=15)
        
        stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
        
        stdout_result = stdout.read().decode('utf-8', errors='ignore').strip()
        stderr_result = stderr.read().decode('utf-8', errors='ignore').strip()
        exit_status = stdout.channel.recv_exit_status()
        
        ssh.close()
        
        return exit_status == 0, stdout_result, stderr_result
        
    except Exception as e:
        return False, "", str(e)

def create_maintenance_page():
    """Create a simple maintenance page"""
    
    maintenance_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GraphRAG Assistant - Under Maintenance</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        
        .maintenance-container {
            text-align: center;
            max-width: 600px;
            padding: 40px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .maintenance-icon {
            font-size: 80px;
            margin-bottom: 20px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
            font-weight: 300;
        }
        
        .subtitle {
            font-size: 1.2em;
            margin-bottom: 30px;
            opacity: 0.9;
        }
        
        .features {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            margin: 30px 0;
            text-align: left;
        }
        
        .features h3 {
            color: #ffd700;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .features ul {
            list-style: none;
            padding: 0;
        }
        
        .features li {
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .features li:last-child {
            border-bottom: none;
        }
        
        .features li::before {
            content: "🧠 ";
            margin-right: 10px;
        }
        
        .status {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .eta {
            font-size: 1.1em;
            color: #ffd700;
            font-weight: bold;
        }
        
        .contact {
            margin-top: 30px;
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top: 3px solid #ffd700;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="maintenance-container">
        <div class="maintenance-icon">🔧</div>
        <h1>GraphRAG Assistant</h1>
        <div class="subtitle">Under Maintenance</div>
        
        <div class="status">
            <p>We're currently upgrading our GraphRAG Assistant to provide you with an even better experience.</p>
            <div class="spinner"></div>
            <div class="eta">Expected completion: Within 24 hours</div>
        </div>
        
        <div class="features">
            <h3>🚀 What's Coming:</h3>
            <ul>
                <li>Enhanced GraphRAG functionality with cross-document analysis</li>
                <li>Improved multi-source knowledge synthesis</li>
                <li>Advanced entity extraction and relationship mapping</li>
                <li>Better contextual query enhancement</li>
                <li>Optimized ChromaDB performance</li>
                <li>Fresh deployment with clean environment</li>
            </ul>
        </div>
        
        <div class="contact">
            <p>Thank you for your patience!</p>
            <p>The GraphRAG Assistant will be back online soon with enhanced capabilities.</p>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 minutes to check if service is back
        setTimeout(function() {
            window.location.reload();
        }, 1800000); // 30 minutes
    </script>
</body>
</html>'''
    
    return maintenance_html

def deploy_maintenance_page():
    """Deploy the maintenance page to the server"""
    
    print("🔧 DEPLOYING MAINTENANCE PAGE")
    print("=" * 50)
    
    # Create maintenance HTML
    maintenance_html = create_maintenance_page()
    
    # Kill existing services
    print("   🛑 Stopping existing services...")
    kill_commands = [
        "pkill -9 -f 'streamlit'",
        "pkill -9 -f 'chroma'",
        "pkill -9 -f 'embedding'",
        "fuser -k 8502/tcp 2>/dev/null || true"
    ]
    
    for cmd in kill_commands:
        run_ssh_command(cmd)
        time.sleep(1)
    
    # Create maintenance directory
    print("   📁 Creating maintenance directory...")
    run_ssh_command("mkdir -p /root/maintenance")
    
    # Upload maintenance page
    print("   📄 Uploading maintenance page...")
    success, _, _ = run_ssh_command(f"cat > /root/maintenance/index.html << 'EOF'\n{maintenance_html}\nEOF")
    
    if success:
        print("   ✅ Maintenance page created")
    else:
        print("   ❌ Failed to create maintenance page")
        return False
    
    # Start simple HTTP server for maintenance page
    print("   🌐 Starting maintenance server on port 8502...")
    success, stdout, stderr = run_ssh_command(
        "cd /root/maintenance && nohup python3 -m http.server 8502 > /root/logs/maintenance.log 2>&1 &"
    )
    
    time.sleep(3)
    
    # Verify maintenance page is accessible
    print("   ✅ Verifying maintenance page...")
    success, stdout, stderr = run_ssh_command("curl -s http://localhost:8502 | head -5")
    
    if success and "GraphRAG Assistant" in stdout:
        print("   ✅ Maintenance page is live!")
        return True
    else:
        print("   ⚠️ Maintenance page may not be fully accessible")
        return False

def main():
    """Main function"""
    
    print("🔧 DEPLOY MAINTENANCE PAGE FOR GRAPHRAG ASSISTANT")
    print("=" * 60)
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Target: 1.32.228.33:8502")
    print("=" * 60)
    
    try:
        if deploy_maintenance_page():
            print("\n" + "=" * 60)
            print("✅ MAINTENANCE PAGE DEPLOYED SUCCESSFULLY")
            print("=" * 60)
            
            print("🌐 Maintenance page is now live at:")
            print("   http://1.32.228.33:8502")
            
            print("\n📋 What users will see:")
            print("   • Professional maintenance notice")
            print("   • Information about GraphRAG upgrades")
            print("   • Expected completion time (24 hours)")
            print("   • List of upcoming improvements")
            print("   • Auto-refresh every 30 minutes")
            
            print("\n🔄 Next steps:")
            print("   1. Server reboot when ready")
            print("   2. Fresh deployment tomorrow")
            print("   3. GraphRAG Assistant back online")
            
            return True
        else:
            print("❌ Failed to deploy maintenance page")
            return False
            
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
