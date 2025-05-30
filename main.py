#!/usr/bin/env python
"""
Main entry point for the web scraping system.
"""
import os
import sys
import logging
import argparse

# Try to import the modern CLI first, fall back to the standard interface if not available
try:
    from cli.modern_cli import cli as modern_app
    USE_MODERN_CLI = True
except ImportError:
    from cli.interface import app
    USE_MODERN_CLI = False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Web Scraping System")
    parser.add_argument("--classic", action="store_true", help="Use classic CLI interface")
    args, remaining = parser.parse_known_args()

    # Determine which CLI to use
    if args.classic or not USE_MODERN_CLI:
        # Use classic CLI
        sys.argv = [sys.argv[0]] + remaining
        app()
    else:
        # Use modern CLI
        sys.argv = [sys.argv[0]] + remaining
        modern_app()
