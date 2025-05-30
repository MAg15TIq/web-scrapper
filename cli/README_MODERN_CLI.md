# Modern CLI for Intelligent Web Scraping System

This directory contains the modern, colorful CLI interface for the Intelligent Web Scraping System.

## Features

- **Rich Colorful Interface**: Vibrant colors and modern design
- **Agent-Specific Themes**: Each agent has its own color scheme and icon
- **Real-Time Progress Visualization**: Live progress bars and status updates
- **Interactive Dashboard**: Live monitoring of agent activities
- **Intelligent Input Analysis**: Automatic detection of input types and agent selection

## Usage

### Basic Commands

```bash
# Start interactive scraping
python main.py scrape --interactive

# Scrape a specific URL
python main.py scrape --url "https://example.com"

# Process a document
python main.py scrape --file "document.pdf"

# Launch the live dashboard
python main.py dashboard

# List all available agents
python main.py agents
```

### Requirements

The modern CLI requires the following packages:
- rich
- click
- pyfiglet

These are automatically installed when you run `pip install -r requirements.txt`.

## Implementation Details

The modern CLI is implemented in `modern_cli.py` and uses the following components:

1. **ModernCLI Class**: Main class that handles the CLI interface
2. **Agent Themes**: Color schemes and icons for each agent type
3. **Click Commands**: Command-line interface using Click
4. **Rich Components**: Tables, panels, progress bars, and other rich components

## Agent Visualization

Each agent is represented with a unique icon and color scheme:

- 🧠 Master Intelligence Agent (Gold)
- 🎯 Coordinator Agent (Blue)
- 🕷️ Scraper Agent (Green)
- 🔍 Parser Agent (Red)
- 💾 Storage Agent (Purple)
- ⚡ JavaScript Agent (Yellow)
- 🔐 Authentication Agent (Bright Red)
- 🥷 Anti-Detection Agent (Dark Gray)
- 🔄 Data Transformation Agent (Cyan)
- 🔗 API Integration Agent (Magenta)
- 🧮 NLP Processing Agent (Bright Cyan)
- 🖼️ Image Processing Agent (Bright Magenta)
- 📄 Document Intelligence Agent (Bright Yellow)
- 🌐 URL Intelligence Agent (Bright Blue)
- 🎬 Media Processing Agent (Bright Red)
- ✅ Quality Assurance Agent (Bright Green)
- ⚡ Performance Optimization Agent (Bright Blue)

## Dashboard

The live dashboard provides real-time monitoring of:

1. Agent activities and statuses
2. System performance metrics
3. Task progress and completion rates
4. Resource utilization

## Customization

You can customize the CLI by modifying the `AGENT_THEMES` dictionary in `modern_cli.py`.
