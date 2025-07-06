"""
Setup script for the LangChain & Pydantic AI enhanced web scraping system.
This script helps users install dependencies and configure the system.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Dict, Any


class EnhancedSystemSetup:
    """Setup manager for the enhanced web scraping system."""
    
    def __init__(self):
        """Initialize the setup manager."""
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.requirements_file = self.project_root / "requirements.txt"
        self.env_file = self.project_root / ".env"
        
    def print_banner(self):
        """Print setup banner."""
        print("=" * 70)
        print("[SETUP] LangChain & Pydantic AI Enhanced Web Scraping System Setup")
        print("=" * 70)
        print("This setup will install and configure:")
        print("• LangChain for AI-powered agent reasoning")
        print("• Pydantic AI for structured data validation")
        print("• LangGraph for complex workflow orchestration")
        print("• Enhanced multi-agent communication protocols")
        print("• Monitoring and observability tools")
        print("=" * 70)
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        print("\n[PYTHON] Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 9):
            print(f"[ERROR] Python {version.major}.{version.minor} is not supported.")
            print("   Please install Python 3.9 or higher.")
            return False
        
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro} is compatible.")
        return True
    
    def create_virtual_environment(self) -> bool:
        """Create a virtual environment."""
        print("\n[ENV] Setting up virtual environment...")
        
        if self.venv_path.exists():
            print("[OK] Virtual environment already exists.")
            return True
        
        try:
            subprocess.run([
                sys.executable, "-m", "venv", str(self.venv_path)
            ], check=True)
            print("[OK] Virtual environment created successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to create virtual environment: {e}")
            return False
    
    def get_pip_command(self) -> str:
        """Get the pip command for the current platform."""
        if platform.system() == "Windows":
            return str(self.venv_path / "Scripts" / "pip.exe")
        else:
            return str(self.venv_path / "bin" / "pip")
    
    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        print("\n[DEPENDENCIES] Installing dependencies...")
        
        if not self.requirements_file.exists():
            print(f"[ERROR] Requirements file not found: {self.requirements_file}")
            return False
        
        pip_cmd = self.get_pip_command()
        
        try:
            # Upgrade pip first
            subprocess.run([
                pip_cmd, "install", "--upgrade", "pip"
            ], check=True)
            
            # Install requirements
            subprocess.run([
                pip_cmd, "install", "-r", str(self.requirements_file)
            ], check=True)
            
            print("[OK] Dependencies installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install dependencies: {e}")
            return False
    
    def create_env_file(self) -> bool:
        """Create environment configuration file."""
        print("\n⚙️  Creating environment configuration...")
        
        if self.env_file.exists():
            print("✅ Environment file already exists.")
            return True
        
        env_template = """# LangChain & Pydantic AI Enhanced System Configuration

# OpenAI Configuration (Required for full functionality)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=2000

# Anthropic Configuration (Alternative to OpenAI)
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# ANTHROPIC_MODEL=claude-3-sonnet-20240229

# LangChain Settings
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=web-scraping-agents

# Redis Configuration (for distributed agent communication)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
# REDIS_PASSWORD=your_redis_password

# PostgreSQL Configuration (for data storage)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=web_scraping
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# Agent System Configuration
MAX_CONCURRENT_WORKFLOWS=10
DEFAULT_TIMEOUT=300
MAX_RETRY_ATTEMPTS=3
MIN_AGENTS_PER_TYPE=1
MAX_AGENTS_PER_TYPE=5

# Monitoring Configuration
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8000
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security Configuration
API_KEY_REQUIRED=false
RATE_LIMIT_ENABLED=true
RATE_LIMIT_RPM=60
ENCRYPT_SENSITIVE_DATA=true
DATA_RETENTION_DAYS=90

# Performance Settings
ENABLE_PERFORMANCE_OPTIMIZATION=true
ADAPTIVE_RATE_LIMITING=true
INTELLIGENT_RETRY=true
"""
        
        try:
            with open(self.env_file, 'w') as f:
                f.write(env_template)
            print("✅ Environment file created successfully.")
            print(f"   📝 Please edit {self.env_file} with your configuration.")
            return True
        except Exception as e:
            print(f"❌ Failed to create environment file: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        print("\n📁 Creating project directories...")
        
        directories = [
            "data",
            "logs",
            "output",
            "screenshots",
            "temp"
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(exist_ok=True)
            
            print("✅ Project directories created successfully.")
            return True
        except Exception as e:
            print(f"❌ Failed to create directories: {e}")
            return False
    
    def test_installation(self) -> bool:
        """Test the installation by importing key modules."""
        print("\n🧪 Testing installation...")
        
        test_imports = [
            ("langchain", "LangChain"),
            ("langgraph", "LangGraph"),
            ("pydantic", "Pydantic"),
            ("redis", "Redis"),
            ("sqlalchemy", "SQLAlchemy"),
            ("prometheus_client", "Prometheus Client")
        ]
        
        failed_imports = []
        
        for module, name in test_imports:
            try:
                __import__(module)
                print(f"   ✅ {name}")
            except ImportError:
                print(f"   ❌ {name}")
                failed_imports.append(name)
        
        if failed_imports:
            print(f"\n⚠️  Some modules failed to import: {', '.join(failed_imports)}")
            print("   This may be normal if you haven't configured all services.")
            return False
        
        print("\n✅ All core modules imported successfully!")
        return True
    
    def print_next_steps(self):
        """Print next steps for the user."""
        print("\n" + "=" * 70)
        print("[NEXT] SETUP COMPLETED! Next Steps:")
        print("=" * 70)
        
        steps = [
            "1. Configure API Keys:",
            f"   • Edit {self.env_file}",
            "   • Add your OpenAI API key (required for full functionality)",
            "   • Optionally add Anthropic API key as alternative",
            "",
            "2. Set up External Services (Optional):",
            "   • Install and configure Redis for distributed communication",
            "   • Install and configure PostgreSQL for data persistence",
            "   • Set up monitoring with Prometheus/Grafana",
            "",
            "3. Test the System:",
            "   • Run: python examples/langchain_enhanced_example.py",
            "   • Check the output for successful operation",
            "",
            "4. Start Development:",
            "   • Create custom agents in the agents/ directory",
            "   • Define new workflows in the workflows/ directory",
            "   • Add custom tools and integrations",
            "",
            "5. Production Deployment:",
            "   • Configure proper logging and monitoring",
            "   • Set up load balancing and scaling",
            "   • Implement proper security measures"
        ]
        
        for step in steps:
            print(step)
        
        print("\n" + "=" * 70)
        print("[DOCS] Documentation and Examples:")
        print("   • README.md - System overview and basic usage")
        print("   • examples/ - Example scripts and use cases")
        print("   • docs/ - Detailed documentation")
        print("   • config/ - Configuration templates and examples")
        print("=" * 70)
    
    def run_setup(self) -> bool:
        """Run the complete setup process."""
        self.print_banner()
        
        # Check prerequisites
        if not self.check_python_version():
            return False
        
        # Setup steps
        steps = [
            ("Creating virtual environment", self.create_virtual_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Creating environment file", self.create_env_file),
            ("Creating directories", self.create_directories),
            ("Testing installation", self.test_installation)
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                print(f"\n[ERROR] Setup failed at: {step_name}")
                return False
        
        self.print_next_steps()
        return True


def main():
    """Main setup function."""
    setup = EnhancedSystemSetup()
    
    try:
        success = setup.run_setup()
        if success:
            print("\n[OK] Setup completed successfully!")
            sys.exit(0)
        else:
            print("\n[ERROR] Setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[STOP] Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
