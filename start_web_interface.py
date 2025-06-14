#!/usr/bin/env python3
"""
Unified Web Interface Starter
Automatically starts both backend API and frontend servers with real data integration.
"""

import os
import sys
import time
import socket
import subprocess
import webbrowser
import signal
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class WebInterfaceManager:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.backend_port = 8001
        self.frontend_port = 8000
        self.host = "localhost"
        
    def check_port_available(self, host, port):
        """Check if a port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)  # 1 second timeout
            result = sock.connect_ex((host, port))
            sock.close()
            return result != 0
        except Exception as e:
            console.print(f"[yellow]Warning: Error checking port {port}: {e}[/yellow]")
            return False

    def find_available_port(self, start_port, max_attempts=100):
        """Find an available port starting from start_port."""
        port = start_port
        attempts = 0

        while attempts < max_attempts:
            if self.check_port_available(self.host, port):
                return port
            port += 1
            attempts += 1

        # If no port found in range, try some common alternative ports
        alternative_ports = [3000, 5000, 8080, 8888, 9000]
        for alt_port in alternative_ports:
            if alt_port not in range(start_port, start_port + max_attempts):
                if self.check_port_available(self.host, alt_port):
                    console.print(f"[yellow]Using alternative port {alt_port}[/yellow]")
                    return alt_port

        raise Exception(f"No available ports found starting from {start_port} after {max_attempts} attempts")

    def kill_process_on_port(self, port):
        """Kill any process running on the specified port (Windows-specific)."""
        try:
            import subprocess
            import platform

            if platform.system() == "Windows":
                # Find process using the port
                result = subprocess.run(
                    ["netstat", "-ano"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                for line in result.stdout.split('\n'):
                    if f":{port}" in line and "LISTENING" in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            try:
                                subprocess.run(["taskkill", "/F", "/PID", pid],
                                             capture_output=True, timeout=5)
                                console.print(f"[yellow]Killed process {pid} on port {port}[/yellow]")
                                return True
                            except Exception:
                                pass
            else:
                # Unix-like systems
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.stdout.strip():
                    pid = result.stdout.strip()
                    subprocess.run(["kill", "-9", pid], timeout=5)
                    console.print(f"[yellow]Killed process {pid} on port {port}[/yellow]")
                    return True

        except Exception as e:
            console.print(f"[yellow]Could not kill process on port {port}: {e}[/yellow]")

        return False
    
    def install_missing_dependencies(self):
        """Install missing dependencies."""
        console.print("[yellow]📦 Installing missing dependencies...[/yellow]")
        
        dependencies = [
            "pydantic-settings",
            "pydantic[email]",
            "fastapi",
            "uvicorn[standard]",
            "websockets",
            "jinja2"
        ]
        
        for dep in dependencies:
            try:
                console.print(f"[dim]Installing {dep}...[/dim]")
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                             check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Failed to install {dep}: {e}[/red]")
                return False
        
        console.print("[green]✅ Dependencies installed successfully[/green]")
        return True
    
    def start_backend_server(self):
        """Start the backend API server."""
        backend_script = Path("web/api/main.py")
        
        if not backend_script.exists():
            console.print("[red]❌ Backend API server not found![/red]")
            return False
        
        # Find available port for backend
        self.backend_port = self.find_available_port(8001)
        
        console.print(f"[blue]🔧 Starting backend API server on port {self.backend_port}...[/blue]")
        
        env = {
            **dict(os.environ),
            'WEB_HOST': self.host,
            'WEB_PORT': str(self.backend_port),
            'WEB_DEBUG': 'false'
        }
        
        try:
            self.backend_process = subprocess.Popen(
                [sys.executable, "-m", "web.api.main"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for backend to start
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Starting backend server...", total=None)
                
                for i in range(10):  # Wait up to 10 seconds
                    time.sleep(1)
                    if self.backend_process.poll() is not None:
                        stdout, stderr = self.backend_process.communicate()
                        console.print(f"[red]❌ Backend failed to start: {stderr}[/red]")
                        return False
                    
                    # Check if server is responding
                    if not self.check_port_available(self.host, self.backend_port):
                        break
                else:
                    console.print("[red]❌ Backend server failed to start within timeout[/red]")
                    return False
            
            console.print(f"[green]✅ Backend API server started on http://{self.host}:{self.backend_port}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to start backend server: {e}[/red]")
            return False
    
    def create_unified_frontend_config(self):
        """Create a configuration file for the frontend to use real backend."""
        config_content = f"""
// Auto-generated configuration for unified web interface
const CONFIG = {{
    API_BASE_URL: 'http://{self.host}:{self.backend_port}/api/v1',
    WS_URL: 'ws://{self.host}:{self.backend_port}/ws',
    REFRESH_INTERVAL: 5000,
    USE_REAL_DATA: true,
    BACKEND_HOST: '{self.host}',
    BACKEND_PORT: {self.backend_port},
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
        
        console.print(f"[green]✅ Frontend configuration created: {config_path}[/green]")
    
    def start_frontend_server(self):
        """Start the frontend server."""
        frontend_script = Path("web/frontend/test_server.py")
        
        if not frontend_script.exists():
            console.print("[red]❌ Frontend server not found![/red]")
            return False
        
        # Find available port for frontend
        self.frontend_port = self.find_available_port(8000)
        
        console.print(f"[blue]🎨 Starting frontend server on port {self.frontend_port}...[/blue]")
        
        # Create unified configuration
        self.create_unified_frontend_config()
        
        env = {
            **dict(os.environ),
            'WEB_HOST': self.host,
            'WEB_PORT': str(self.frontend_port),
            'WEB_DEBUG': 'false',
            'API_BASE_URL': f'http://{self.host}:{self.backend_port}/api/v1',
            'WS_URL': f'ws://{self.host}:{self.backend_port}/ws',
            'USE_REAL_DATA': 'true'
        }
        
        try:
            self.frontend_process = subprocess.Popen(
                [sys.executable, str(frontend_script)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for frontend to start
            time.sleep(3)
            
            if self.frontend_process.poll() is not None:
                stdout, stderr = self.frontend_process.communicate()
                console.print(f"[red]❌ Frontend failed to start: {stderr}[/red]")
                return False
            
            console.print(f"[green]✅ Frontend server started on http://{self.host}:{self.frontend_port}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to start frontend server: {e}[/red]")
            return False
    
    def open_browser(self):
        """Open the web interface in the default browser."""
        url = f"http://{self.host}:{self.frontend_port}/app"
        console.print(f"[blue]🌍 Opening browser: {url}[/blue]")
        webbrowser.open(url)
    
    def cleanup(self):
        """Clean up processes."""
        console.print("\n[yellow]🧹 Cleaning up...[/yellow]")
        
        if self.backend_process:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        
        if self.frontend_process:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
        
        console.print("[green]✅ Cleanup completed[/green]")
    
    def start(self, open_browser=True, install_deps=True):
        """Start the unified web interface."""
        console.print(Panel.fit(
            "[bold blue]🚀 Unified Web Scraper Interface[/bold blue]\n\n"
            "[bold]Features:[/bold]\n"
            "• 🔧 Automatic backend + frontend startup\n"
            "• 📊 Real-time data integration\n"
            "• 🌐 WebSocket support\n"
            "• 💾 Live data management\n"
            "• 🤖 Agent monitoring\n"
            "• 📈 System metrics",
            title="Starting Unified Interface",
            border_style="blue"
        ))
        
        try:
            # Install dependencies if needed
            if install_deps:
                if not self.install_missing_dependencies():
                    return False
            
            # Start backend server
            if not self.start_backend_server():
                return False
            
            # Start frontend server
            if not self.start_frontend_server():
                self.cleanup()
                return False
            
            # Display success information
            console.print(Panel.fit(
                f"[bold green]🎉 Unified Web Interface Started Successfully![/bold green]\n\n"
                f"[bold]Access URLs:[/bold]\n"
                f"• Frontend: [link]http://{self.host}:{self.frontend_port}/app[/link]\n"
                f"• Backend API: [link]http://{self.host}:{self.backend_port}/docs[/link]\n"
                f"• Health Check: [link]http://{self.host}:{self.backend_port}/health[/link]\n\n"
                f"[bold]Features Available:[/bold]\n"
                f"• 📊 Dashboard with real-time metrics\n"
                f"• 🔧 Job management and creation\n"
                f"• 🤖 Agent monitoring and control\n"
                f"• 💾 Data management and export\n"
                f"• 📈 System performance monitoring\n\n"
                f"[yellow]Press Ctrl+C to stop all servers[/yellow]",
                title="🌐 Web Interface Ready",
                border_style="green"
            ))
            
            # Open browser
            if open_browser:
                time.sleep(2)  # Give servers a moment to fully start
                self.open_browser()
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to start unified interface: {e}[/red]")
            self.cleanup()
            return False

def main():
    """Main entry point."""
    manager = WebInterfaceManager()
    
    def signal_handler(sig, frame):
        console.print("\n[yellow]🛑 Received interrupt signal[/yellow]")
        manager.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if manager.start():
            # Keep the script running
            while True:
                time.sleep(1)
                
                # Check if processes are still running
                if manager.backend_process and manager.backend_process.poll() is not None:
                    console.print("[red]❌ Backend process died unexpectedly[/red]")
                    break
                    
                if manager.frontend_process and manager.frontend_process.poll() is not None:
                    console.print("[red]❌ Frontend process died unexpectedly[/red]")
                    break
        else:
            console.print("[red]❌ Failed to start unified web interface[/red]")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Shutting down...[/yellow]")
    finally:
        manager.cleanup()

if __name__ == "__main__":
    main()
