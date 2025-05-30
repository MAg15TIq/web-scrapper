#!/usr/bin/env python3
"""
System validation and setup script for the multi-agent web scraping system.
This script tests the core components and provides setup guidance.
"""

import sys
import os
import importlib.util
import subprocess
import json
from pathlib import Path

def check_python_installation():
    """Check Python installation and version."""
    print("🐍 Checking Python installation...")

    python_version = sys.version_info
    print(f"  ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("  ❌ Python 3.8+ is required")
        return False

    print(f"  ✅ Python version is compatible")
    return True

def check_virtual_environment():
    """Check if we're in a virtual environment."""
    print("\n🔧 Checking virtual environment...")

    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

    if in_venv:
        print(f"  ✅ Running in virtual environment: {sys.prefix}")
    else:
        print(f"  ⚠️  Not in virtual environment (using system Python)")
        print(f"  📍 Python location: {sys.executable}")

    return in_venv

def install_missing_dependencies():
    """Install missing dependencies."""
    print("\n📦 Checking and installing dependencies...")

    # Core dependencies that should be available
    core_deps = [
        "requests", "beautifulsoup4", "pandas", "rich", "typer",
        "click", "pyyaml", "pydantic", "asyncio"
    ]

    missing_deps = []

    for dep in core_deps:
        try:
            if dep == "asyncio":
                import asyncio
            else:
                __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            missing_deps.append(dep)
            print(f"  ❌ {dep} - missing")

    if missing_deps:
        print(f"\n🔧 Installing missing dependencies: {', '.join(missing_deps)}")
        try:
            for dep in missing_deps:
                if dep != "asyncio":  # asyncio is built-in
                    subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                    print(f"  ✅ Installed {dep}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install dependencies: {e}")
            return False

    return True

def test_imports():
    """Test if all core modules can be imported."""
    print("🔍 Testing module imports...")

    # Core modules to test
    modules_to_test = [
        "agents.base",
        "agents.coordinator",
        "agents.scraper",
        "agents.parser",
        "agents.storage",
        "models.task",
        "models.message",
        "cli.interface"
    ]

    results = {}

    for module_name in modules_to_test:
        try:
            # Try to import the module
            spec = importlib.util.spec_from_file_location(
                module_name,
                module_name.replace(".", "/") + ".py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                results[module_name] = "✅ SUCCESS"
                print(f"  ✅ {module_name}")
            else:
                results[module_name] = "❌ SPEC_ERROR"
                print(f"  ❌ {module_name} - Could not create spec")
        except Exception as e:
            results[module_name] = f"❌ ERROR: {str(e)}"
            print(f"  ❌ {module_name} - {str(e)}")

    return results

def test_file_structure():
    """Test if all required files and directories exist."""
    print("\n📁 Testing file structure...")

    required_paths = [
        "agents/",
        "agents/__init__.py",
        "agents/base.py",
        "agents/coordinator.py",
        "agents/scraper.py",
        "agents/parser.py",
        "agents/storage.py",
        "models/",
        "models/__init__.py",
        "models/task.py",
        "models/message.py",
        "cli/",
        "cli/__init__.py",
        "cli/interface.py",
        "main.py",
        "requirements.txt",
        "README.md"
    ]

    results = {}

    for path in required_paths:
        if os.path.exists(path):
            results[path] = "✅ EXISTS"
            print(f"  ✅ {path}")
        else:
            results[path] = "❌ MISSING"
            print(f"  ❌ {path}")

    return results

def test_dependencies():
    """Test if required dependencies are available."""
    print("\n📦 Testing dependencies...")

    # Core dependencies
    dependencies = [
        "asyncio",
        "logging",
        "typing",
        "queue",
        "json",
        "os",
        "sys"
    ]

    # Optional dependencies
    optional_deps = [
        "requests",
        "beautifulsoup4",
        "pandas",
        "rich",
        "typer",
        "click"
    ]

    results = {"core": {}, "optional": {}}

    # Test core dependencies
    for dep in dependencies:
        try:
            __import__(dep)
            results["core"][dep] = "✅ AVAILABLE"
            print(f"  ✅ {dep} (core)")
        except ImportError:
            results["core"][dep] = "❌ MISSING"
            print(f"  ❌ {dep} (core)")

    # Test optional dependencies
    for dep in optional_deps:
        try:
            __import__(dep)
            results["optional"][dep] = "✅ AVAILABLE"
            print(f"  ✅ {dep} (optional)")
        except ImportError:
            results["optional"][dep] = "❌ MISSING"
            print(f"  ⚠️  {dep} (optional) - not installed")

    return results

def test_basic_functionality():
    """Test basic system functionality."""
    print("\n⚙️  Testing basic functionality...")

    try:
        # Test task creation
        sys.path.insert(0, os.path.abspath("."))
        from models.task import Task, TaskType, TaskStatus

        # Create a simple task
        task = Task(
            type=TaskType.FETCH_URL,
            parameters={"url": "https://example.com"}
        )

        print(f"  ✅ Task creation: {task.id}")
        print(f"  ✅ Task type: {task.type}")
        print(f"  ✅ Task status: {task.status}")

        return {"task_creation": "✅ SUCCESS"}

    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {str(e)}")
        return {"task_creation": f"❌ ERROR: {str(e)}"}

def generate_report(import_results, file_results, dep_results, func_results):
    """Generate a comprehensive system report."""
    print("\n" + "="*60)
    print("🎯 SYSTEM VALIDATION REPORT")
    print("="*60)

    # Import results
    import_success = sum(1 for r in import_results.values() if r == "✅ SUCCESS")
    import_total = len(import_results)
    print(f"\n📦 Module Imports: {import_success}/{import_total} successful")

    # File structure results
    file_success = sum(1 for r in file_results.values() if r == "✅ EXISTS")
    file_total = len(file_results)
    print(f"📁 File Structure: {file_success}/{file_total} files found")

    # Dependencies results
    core_success = sum(1 for r in dep_results["core"].values() if r == "✅ AVAILABLE")
    core_total = len(dep_results["core"])
    opt_success = sum(1 for r in dep_results["optional"].values() if r == "✅ AVAILABLE")
    opt_total = len(dep_results["optional"])
    print(f"🔧 Core Dependencies: {core_success}/{core_total} available")
    print(f"⚡ Optional Dependencies: {opt_success}/{opt_total} available")

    # Overall status
    overall_score = (import_success + file_success + core_success) / (import_total + file_total + core_total) * 100

    print(f"\n🎯 Overall System Health: {overall_score:.1f}%")

    if overall_score >= 80:
        print("✅ System is ready for use!")
        print("\n🚀 Next steps:")
        print("  1. Install missing optional dependencies if needed")
        print("  2. Run: python main.py agents")
        print("  3. Try: python main.py scrape --interactive")
    elif overall_score >= 60:
        print("⚠️  System has some issues but may be functional")
        print("\n🔧 Recommended actions:")
        print("  1. Fix missing core dependencies")
        print("  2. Check file permissions")
        print("  3. Reinstall virtual environment if needed")
    else:
        print("❌ System has significant issues")
        print("\n🆘 Required actions:")
        print("  1. Reinstall Python environment")
        print("  2. Reinstall all dependencies")
        print("  3. Check system requirements")

def create_simple_test():
    """Create a simple test to verify the system works."""
    print("\n🧪 Creating simple system test...")

    try:
        # Add current directory to Python path
        sys.path.insert(0, os.path.abspath("."))

        # Test basic imports
        from models.task import Task, TaskType, TaskStatus
        from models.message import Message, TaskMessage

        # Create a simple task
        task = Task(
            type=TaskType.FETCH_URL,
            parameters={"url": "https://httpbin.org/get"}
        )

        print(f"  ✅ Created task: {task.id}")
        print(f"  ✅ Task type: {task.type}")
        print(f"  ✅ Task status: {task.status}")

        # Test task status updates
        task.update_status(TaskStatus.RUNNING)
        print(f"  ✅ Updated status to: {task.status}")

        # Test task completion
        task.complete({"status": "success", "data": "test"})
        print(f"  ✅ Task completed with result")

        return True

    except Exception as e:
        print(f"  ❌ Simple test failed: {str(e)}")
        return False

def run_example_scrape():
    """Run a simple example scrape to test the system."""
    print("\n🕷️ Testing simple scrape functionality...")

    try:
        # Add current directory to Python path
        sys.path.insert(0, os.path.abspath("."))

        # Import required modules
        from agents.coordinator import CoordinatorAgent
        from agents.scraper import ScraperAgent
        from models.task import Task, TaskType

        print("  ✅ Successfully imported agent modules")

        # Create coordinator
        coordinator = CoordinatorAgent()
        print(f"  ✅ Created coordinator: {coordinator.agent_id}")

        # Create scraper agent
        scraper = ScraperAgent(coordinator_id=coordinator.agent_id)
        print(f"  ✅ Created scraper: {scraper.agent_id}")

        # Register agent
        coordinator.register_agent(scraper)
        print("  ✅ Registered scraper with coordinator")

        return True

    except Exception as e:
        print(f"  ❌ Example scrape test failed: {str(e)}")
        import traceback
        print(f"  📋 Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run the complete system validation and setup."""
    print("🚀 Multi-Agent Web Scraping System Setup & Validation")
    print("=" * 60)

    # Step 1: Check Python
    python_ok = check_python_installation()
    if not python_ok:
        print("\n❌ Python installation issues detected. Please install Python 3.8+")
        return

    # Step 2: Check virtual environment
    venv_status = check_virtual_environment()

    # Step 3: Install dependencies
    deps_ok = install_missing_dependencies()
    if not deps_ok:
        print("\n❌ Dependency installation failed")
        return

    # Step 4: Test file structure
    file_results = test_file_structure()

    # Step 5: Test imports
    import_results = test_imports()

    # Step 6: Test basic functionality
    func_results = test_basic_functionality()

    # Step 7: Run simple test
    simple_test_ok = create_simple_test()

    # Step 8: Test example scrape
    example_ok = run_example_scrape()

    # Generate comprehensive report
    print("\n" + "="*60)
    print("🎯 FINAL SYSTEM REPORT")
    print("="*60)

    # Calculate overall health
    checks = [python_ok, deps_ok, simple_test_ok, example_ok]
    passed = sum(checks)
    total = len(checks)
    health_score = (passed / total) * 100

    print(f"\n📊 System Health: {health_score:.1f}% ({passed}/{total} checks passed)")

    if health_score >= 75:
        print("\n✅ System is ready for use!")
        print("\n🚀 Next steps:")
        print("  1. Run: python main.py agents")
        print("  2. Try: python main.py scrape --interactive")
        print("  3. Test: python examples/simple_scrape.py")

        # Create a quick start script
        create_quick_start_script()

    elif health_score >= 50:
        print("\n⚠️  System has some issues but may be functional")
        print("\n🔧 Recommended actions:")
        print("  1. Check the error messages above")
        print("  2. Try reinstalling dependencies")
        print("  3. Run this script again")
    else:
        print("\n❌ System has significant issues")
        print("\n🆘 Required actions:")
        print("  1. Check Python installation")
        print("  2. Reinstall virtual environment")
        print("  3. Install all dependencies manually")

def create_quick_start_script():
    """Create a quick start script for easy system usage."""
    print("\n📝 Creating quick start script...")

    quick_start_content = '''#!/usr/bin/env python3
"""
Quick start script for the Multi-Agent Web Scraping System
"""

import sys
import os

def main():
    print("🚀 Multi-Agent Web Scraping System - Quick Start")
    print("=" * 50)

    print("\\nAvailable commands:")
    print("  1. List agents: python main.py agents")
    print("  2. Interactive mode: python main.py scrape --interactive")
    print("  3. Simple scrape: python main.py scrape --url https://httpbin.org/get")
    print("  4. Dashboard: python main.py dashboard")
    print("  5. Run example: python examples/simple_scrape.py")

    print("\\n📚 Documentation:")
    print("  - README.md - Main documentation")
    print("  - examples/ - Example scripts")
    print("  - docs/ - Additional documentation")

    choice = input("\\nEnter command number (1-5) or 'q' to quit: ")

    if choice == "1":
        os.system("python main.py agents")
    elif choice == "2":
        os.system("python main.py scrape --interactive")
    elif choice == "3":
        os.system("python main.py scrape --url https://httpbin.org/get")
    elif choice == "4":
        os.system("python main.py dashboard")
    elif choice == "5":
        os.system("python examples/simple_scrape.py")
    elif choice.lower() == "q":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
'''

    try:
        with open("quick_start.py", "w") as f:
            f.write(quick_start_content)
        print("  ✅ Created quick_start.py")
        print("  💡 Run: python quick_start.py")
    except Exception as e:
        print(f"  ❌ Failed to create quick start script: {e}")

if __name__ == "__main__":
    main()
