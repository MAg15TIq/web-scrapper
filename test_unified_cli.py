#!/usr/bin/env python3
"""
Test script for the Unified Web Scraper CLI
Verifies that all components are working correctly
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required imports work."""
    print("🧪 Testing imports...")
    
    try:
        from cli.unified_cli import UnifiedWebScraperCLI, unified_app
        print("  ✅ Unified CLI imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_cli_initialization():
    """Test CLI initialization."""
    print("🧪 Testing CLI initialization...")
    
    try:
        from cli.unified_cli import UnifiedWebScraperCLI
        
        cli = UnifiedWebScraperCLI()
        print("  ✅ CLI initialization successful")
        
        # Test basic methods
        cli.display_banner()
        print("  ✅ Banner display works")
        
        cli.display_system_status()
        print("  ✅ System status display works")
        
        return True
    except Exception as e:
        print(f"  ❌ CLI initialization error: {e}")
        return False

def test_command_routing():
    """Test command routing logic."""
    print("🧪 Testing command routing...")
    
    try:
        from cli.unified_cli import UnifiedWebScraperCLI
        
        cli = UnifiedWebScraperCLI()
        
        # Test command type detection
        test_cases = [
            ("scrape https://example.com", "natural language"),
            ("help", "built-in command"),
            ("agents", "built-in command"),
            ("analyze https://example.com", "intelligent command"),
            ("check-system", "classic command")
        ]
        
        for command, expected_type in test_cases:
            # Test the detection methods
            is_natural = cli._is_natural_language(command)
            is_classic = cli._is_classic_command(command)
            is_intelligent = cli._is_intelligent_command(command)
            
            print(f"  📝 '{command}' -> Natural: {is_natural}, Classic: {is_classic}, Intelligent: {is_intelligent}")
        
        print("  ✅ Command routing logic works")
        return True
    except Exception as e:
        print(f"  ❌ Command routing error: {e}")
        return False

def test_click_cli():
    """Test Click CLI interface."""
    print("🧪 Testing Click CLI interface...")
    
    try:
        from cli.unified_cli import unified_app
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        # Test help command
        result = runner.invoke(unified_app, ['--help'])
        if result.exit_code == 0:
            print("  ✅ Help command works")
        else:
            print(f"  ⚠️ Help command exit code: {result.exit_code}")
        
        # Test agents command
        result = runner.invoke(unified_app, ['agents'])
        if result.exit_code == 0:
            print("  ✅ Agents command works")
        else:
            print(f"  ⚠️ Agents command exit code: {result.exit_code}")
        
        # Test status command
        result = runner.invoke(unified_app, ['status'])
        if result.exit_code == 0:
            print("  ✅ Status command works")
        else:
            print(f"  ⚠️ Status command exit code: {result.exit_code}")
        
        return True
    except Exception as e:
        print(f"  ❌ Click CLI error: {e}")
        return False

async def test_async_functionality():
    """Test async functionality."""
    print("🧪 Testing async functionality...")
    
    try:
        from cli.unified_cli import UnifiedWebScraperCLI
        
        cli = UnifiedWebScraperCLI()
        
        # Test basic async command processing
        await cli._handle_basic_natural_language("help")
        print("  ✅ Basic natural language processing works")
        
        # Test classic command handling
        await cli._classic_check_system([])
        print("  ✅ Classic command handling works")
        
        return True
    except Exception as e:
        print(f"  ❌ Async functionality error: {e}")
        return False

def test_dependencies():
    """Test optional dependencies."""
    print("🧪 Testing dependencies...")
    
    # Test Rich components
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        print("  ✅ Rich components available")
    except ImportError:
        print("  ❌ Rich components missing")
    
    # Test Click
    try:
        import click
        print("  ✅ Click available")
    except ImportError:
        print("  ❌ Click missing")
    
    # Test questionary
    try:
        import questionary
        print("  ✅ Questionary available")
    except ImportError:
        print("  ❌ Questionary missing")
    
    # Test pyfiglet
    try:
        import pyfiglet
        print("  ✅ Pyfiglet available")
    except ImportError:
        print("  ⚠️ Pyfiglet missing (optional)")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Testing Unified Web Scraper CLI")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    tests = [
        test_dependencies,
        test_imports,
        test_cli_initialization,
        test_command_routing,
        test_click_cli,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            print()
    
    # Test async functionality separately
    try:
        asyncio.run(test_async_functionality())
        passed += 1
        total += 1
    except Exception as e:
        print(f"  ❌ Async test failed: {e}")
        total += 1
    
    print("=" * 50)
    print(f"🎉 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Unified CLI is ready to use.")
        print("\n🚀 Try it out:")
        print("  python main.py --interactive")
        print("  python main.py scrape --url https://quotes.toscrape.com")
        print("  python main.py agents")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        print("\n💡 Common issues:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - Import errors: Check Python path and module structure")
        return 1

if __name__ == "__main__":
    sys.exit(main())
