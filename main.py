#!/usr/bin/env python
"""
Main entry point for the Unified Web Scraping System.
All CLI interfaces combined into one beautiful, comprehensive interface.
"""
import os
import sys
import logging

# Import the unified CLI
try:
    from cli.unified_cli import unified_app
    print("🚀 Loading Unified Web Scraper CLI...")
except ImportError as e:
    print(f"❌ Error importing unified CLI: {e}")
    print("📦 Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Launch the unified CLI
    try:
        unified_app()
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Thanks for using the Unified Web Scraper!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.exception("Fatal error in main application")
        sys.exit(1)
