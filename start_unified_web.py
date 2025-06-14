#!/usr/bin/env python3
"""
Simple Unified Web Interface Starter
Starts both backend and frontend servers automatically.
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def main():
    print("🚀 Starting Unified Web Scraper Interface...")
    print("=" * 50)
    
    # Configuration
    backend_port = 8001
    frontend_port = 8000
    host = "localhost"
    
    # Check if backend script exists
    backend_script = Path("web/api/main.py")
    if not backend_script.exists():
        print("❌ Backend API server not found!")
        print(f"   Expected: {backend_script}")
        return False
    
    # Check if frontend script exists
    frontend_script = Path("web/frontend/test_server.py")
    if not frontend_script.exists():
        print("❌ Frontend server not found!")
        print(f"   Expected: {frontend_script}")
        return False
    
    print("✅ Both server scripts found")
    
    try:
        # Start backend server
        print(f"\n🔧 Starting backend API server on port {backend_port}...")
        
        backend_env = {
            **dict(os.environ),
            'WEB_HOST': host,
            'WEB_PORT': str(backend_port),
            'WEB_DEBUG': 'false'
        }
        
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "web.api.main"],
            env=backend_env
        )
        
        print("⏳ Waiting for backend to start...")
        time.sleep(8)  # Give backend time to start
        
        # Check if backend is still running
        if backend_process.poll() is not None:
            print("❌ Backend failed to start!")
            return False
        
        print(f"✅ Backend started on http://{host}:{backend_port}")
        
        # Create frontend configuration
        config_content = f"""
// Auto-generated configuration for unified web interface
const CONFIG = {{
    API_BASE_URL: 'http://{host}:{backend_port}/api/v1',
    WS_URL: 'ws://{host}:{backend_port}/ws',
    REFRESH_INTERVAL: 5000,
    USE_REAL_DATA: true,
    BACKEND_HOST: '{host}',
    BACKEND_PORT: {backend_port},
    CHART_COLORS: {{
        primary: '#0d6efd',
        success: '#198754',
        warning: '#ffc107',
        danger: '#dc3545',
        info: '#0dcaf0',
        secondary: '#6c757d'
    }}
}};

// Override the default CONFIG in common.js
if (typeof window !== 'undefined') {{
    window.UNIFIED_CONFIG = CONFIG;
}}
"""
        
        config_path = Path("web/frontend/static/js/unified_config.js")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Frontend configuration created")
        
        # Start frontend server
        print(f"\n🎨 Starting frontend server on port {frontend_port}...")
        
        frontend_env = {
            **dict(os.environ),
            'WEB_HOST': host,
            'WEB_PORT': str(frontend_port),
            'WEB_DEBUG': 'false'
        }
        
        frontend_process = subprocess.Popen(
            [sys.executable, str(frontend_script)],
            env=frontend_env
        )
        
        print("⏳ Waiting for frontend to start...")
        time.sleep(5)  # Give frontend time to start
        
        # Check if frontend is still running
        if frontend_process.poll() is not None:
            print("❌ Frontend failed to start!")
            backend_process.terminate()
            return False
        
        print(f"✅ Frontend started on http://{host}:{frontend_port}")
        
        # Success message
        print("\n" + "=" * 50)
        print("🎉 Unified Web Interface Started Successfully!")
        print("=" * 50)
        print(f"📊 Frontend Dashboard: http://{host}:{frontend_port}/app")
        print(f"🔧 Backend API Docs:   http://{host}:{backend_port}/docs")
        print(f"💚 Health Check:       http://{host}:{backend_port}/health")
        print("\n🌟 Features Available:")
        print("   • Real-time data (no mock data)")
        print("   • WebSocket connections")
        print("   • Live job management")
        print("   • Agent monitoring")
        print("   • Data export tools")
        print("\n⚠️  Press Ctrl+C to stop both servers")
        print("=" * 50)
        
        # Open browser
        time.sleep(2)
        url = f"http://{host}:{frontend_port}/app"
        print(f"🌍 Opening browser: {url}")
        webbrowser.open(url)
        
        # Keep running
        try:
            while True:
                time.sleep(1)
                
                # Check if processes are still running
                if backend_process.poll() is not None:
                    print("\n❌ Backend process died unexpectedly")
                    break
                    
                if frontend_process.poll() is not None:
                    print("\n❌ Frontend process died unexpectedly")
                    break
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down servers...")
            
        finally:
            # Cleanup
            print("🧹 Cleaning up...")
            if backend_process:
                backend_process.terminate()
                try:
                    backend_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    backend_process.kill()
            
            if frontend_process:
                frontend_process.terminate()
                try:
                    frontend_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    frontend_process.kill()
            
            print("✅ Cleanup completed")
            print("👋 Goodbye!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(0)
