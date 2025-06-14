#!/usr/bin/env python3
"""
Demo script to showcase the web interface improvements.
This script starts the enhanced web interface and provides a guided tour.
"""

import webbrowser
import time
import subprocess
import sys
import os
from pathlib import Path

def print_banner():
    """Print a welcome banner."""
    print("=" * 80)
    print("🚀 WEB SCRAPER INTERFACE IMPROVEMENTS DEMO")
    print("=" * 80)
    print()
    print("This demo showcases the enhanced web interface with:")
    print("✨ Modern UI design with smooth animations")
    print("🎨 Enhanced visual hierarchy and typography")
    print("🧭 Fixed navigation with smooth transitions")
    print("📊 Real data integration with animated charts")
    print("📱 Responsive design for all devices")
    print()

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import jinja2
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install fastapi uvicorn jinja2")
        return False

def start_server():
    """Start the demo server."""
    print("🔧 Starting the enhanced web interface server...")
    
    # Change to the project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    try:
        # Start the server
        server_script = project_root / "web" / "frontend" / "test_server.py"
        process = subprocess.Popen([
            sys.executable, str(server_script)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Check if server is running
        if process.poll() is None:
            print("✅ Server started successfully!")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def open_demo_pages():
    """Open demo pages in the browser."""
    base_url = "http://localhost:8000"
    
    pages = [
        ("/app", "📊 Dashboard - Main overview with animated metrics"),
        ("/app/jobs", "🔧 Jobs - Job management interface"),
        ("/app/agents", "🤖 Agents - Agent monitoring and control"),
        ("/app/monitoring", "📈 Monitoring - System performance metrics"),
        ("/app/data", "💾 Data - Data management and export")
    ]
    
    print("\n🌐 Opening demo pages in your browser...")
    print("Navigate between pages to see the smooth transitions and animations!")
    print()
    
    for path, description in pages:
        url = base_url + path
        print(f"   {description}")
        print(f"   URL: {url}")
    
    print()
    
    # Open the main dashboard
    dashboard_url = base_url + "/app"
    print(f"🚀 Opening dashboard: {dashboard_url}")
    webbrowser.open(dashboard_url)

def show_improvement_highlights():
    """Show key improvement highlights."""
    print("\n" + "=" * 80)
    print("🎯 KEY IMPROVEMENTS DEMONSTRATED")
    print("=" * 80)
    
    improvements = [
        ("🎨 Visual Design", [
            "Modern color scheme with CSS custom properties",
            "Enhanced typography with Google Fonts (Inter)",
            "Improved card designs with gradients and shadows",
            "Better spacing and visual hierarchy"
        ]),
        ("✨ Animations", [
            "Smooth page transitions with loading overlays",
            "Staggered animations for metric cards",
            "Animated counters for real-time data",
            "Hover effects and micro-interactions"
        ]),
        ("🧭 Navigation", [
            "Fixed tab navigation with proper active states",
            "Smooth transitions between pages",
            "Enhanced sidebar with hover effects",
            "Improved mobile navigation"
        ]),
        ("📊 Data Integration", [
            "Real API data instead of mock data",
            "Enhanced charts with gradients and tooltips",
            "Live system metrics with animated progress bars",
            "Error handling with fallback data"
        ]),
        ("📱 Responsive Design", [
            "Mobile-first responsive layout",
            "Touch-friendly interface elements",
            "Optimized performance for all devices",
            "Accessibility improvements"
        ])
    ]
    
    for category, features in improvements:
        print(f"\n{category}:")
        for feature in features:
            print(f"   ✓ {feature}")

def interactive_demo():
    """Run an interactive demo."""
    print("\n" + "=" * 80)
    print("🎮 INTERACTIVE DEMO GUIDE")
    print("=" * 80)
    
    demo_steps = [
        "1. 📊 Dashboard Overview:",
        "   • Notice the animated metric cards loading with staggered delays",
        "   • Observe the enhanced charts with gradients and smooth animations",
        "   • Check the real-time system performance indicators",
        "",
        "2. 🧭 Navigation Testing:",
        "   • Click on different navigation tabs (Jobs, Agents, Monitoring, Data)",
        "   • Notice the smooth page transitions with loading overlays",
        "   • Observe the active state highlighting in the sidebar",
        "",
        "3. 🎨 Visual Enhancements:",
        "   • Hover over cards to see elevation effects",
        "   • Notice the improved typography and spacing",
        "   • Check the responsive design by resizing your browser",
        "",
        "4. 📊 Data Visualization:",
        "   • Interact with the charts to see enhanced tooltips",
        "   • Watch the animated progress bars in system performance",
        "   • Notice the real data integration in all components",
        "",
        "5. ✨ Animation Details:",
        "   • Refresh the page to see the loading animations",
        "   • Hover over navigation items for smooth transitions",
        "   • Notice the subtle animations throughout the interface"
    ]
    
    for step in demo_steps:
        print(step)
    
    print("\n" + "=" * 80)
    print("💡 TIP: Open browser developer tools to see the enhanced CSS and animations!")
    print("=" * 80)

def main():
    """Main demo function."""
    print_banner()
    
    if not check_dependencies():
        return
    
    server_process = start_server()
    if not server_process:
        return
    
    try:
        open_demo_pages()
        show_improvement_highlights()
        interactive_demo()
        
        print("\n🎉 Demo is now running!")
        print("Press Ctrl+C to stop the server and exit the demo.")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping demo server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Demo completed successfully!")
        print("Thank you for exploring the web interface improvements!")

if __name__ == "__main__":
    main()
