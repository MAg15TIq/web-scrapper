# 🚀 Unified Web Scraper CLI - All-in-One Interface

The **Unified Web Scraper CLI** combines all four different CLI interfaces (Modern, Classic, Enhanced, and Intelligent) into one comprehensive, beautiful, and feature-rich command-line interface.

## 🌟 What Makes It Unified?

### 🎯 **All CLI Capabilities in One Place**
- 🎨 **Modern CLI**: Beautiful visual interface with rich colors and animations
- 🛠️ **Classic CLI**: Complete command suite with 18+ advanced commands
- 🧠 **Enhanced CLI**: AI-powered natural language processing with LangChain
- 🔬 **Intelligent CLI**: Advanced AI analysis and document processing

### 🎨 **Beautiful & Intuitive Interface**
- Rich colors, animations, and progress bars
- Agent-specific themes with unique icons
- Interactive prompts and guided workflows
- Real-time monitoring and live dashboards

### 🧠 **Intelligent Command Routing**
- Automatically detects command type and routes to appropriate handler
- Supports natural language, classic commands, and AI analysis
- Seamless switching between interaction modes
- Context-aware suggestions and help

## 🚀 Quick Start

### Installation
```bash
# The unified CLI is automatically available after installing dependencies
pip install -r requirements.txt

# Launch the unified interface
python main.py
```

### Basic Usage
```bash
# Interactive mode (recommended for beginners)
python main.py --interactive

# Quick scraping
python main.py scrape --url https://example.com

# Show all available agents
python main.py agents

# System status
python main.py status

# Get help
python main.py --help
```

## 🎯 Available Commands

### 🌟 **Core Commands**
```bash
# Interactive mode with all capabilities
python main.py --interactive
python main.py interactive

# Scraping with advanced options
python main.py scrape --url <URL> [options]

# System information
python main.py agents          # List all agents
python main.py status          # System status
python main.py dashboard       # Live monitoring
python main.py config          # Configuration
python main.py history         # Command history
python main.py modes           # Interaction modes
```

### 🕷️ **Scraping Commands**
```bash
# Basic scraping
python main.py scrape --url https://example.com

# Advanced scraping with options
python main.py scrape \
  --url https://example.com \
  --selectors "title:h1,price:.price" \
  --format csv \
  --max-pages 5 \
  --render-js \
  --anti-detection \
  --clean-data

# Interactive scraping setup
python main.py scrape --interactive
```

### 🧠 **AI Analysis Commands**
```bash
# AI-powered analysis
python main.py analyze https://example.com --type url
python main.py analyze document.pdf --type document
python main.py analyze "sample text" --type text
```

### ⚙️ **Configuration Commands**
```bash
# Configuration management
python main.py config
python main.py --config custom_config.yaml
python main.py --profile production
python main.py --verbose
```

## 🎨 Interaction Modes

### 1. 🗣️ **Natural Language Mode**
Use conversational commands:
```
🤖 What would you like to do?
> "Scrape product prices from amazon.com"
> "Extract news articles from techcrunch.com"  
> "Monitor job queue status every 5 minutes"
> "Configure anti-detection settings"
```

### 2. 💻 **Classic Command Mode**
Traditional CLI commands:
```
🤖 What would you like to do?
> scrape https://example.com
> preview output.json
> check-system
> analyze-text "sample text"
> clean-data data.json
```

### 3. 🧠 **Intelligent Analysis Mode**
AI-powered analysis:
```
🤖 What would you like to do?
> analyze https://example.com
> document process.pdf
> optimize performance
> quality assessment data.json
```

### 4. 🎯 **Visual Interactive Mode**
Guided prompts and menus for easy navigation.

## 🤖 Available Agents

### 🧠 **Intelligence Agents**
- 🧠 Master Intelligence Agent - AI orchestration
- 🌐 URL Intelligence Agent - Smart URL analysis
- 👁️ Content Recognition Agent - Content understanding
- 📄 Document Intelligence Agent - Document processing
- 🗣️ NLP Processing Agent - Natural language processing
- ⚡ Performance Optimization Agent - System optimization
- ✅ Quality Assurance Agent - Data quality control

### 🛠️ **Core Agents**
- 🎯 Coordinator Agent - Task coordination
- 🕷️ Scraper Agent - Web scraping
- 📝 Parser Agent - Data parsing
- 💾 Storage Agent - Data storage
- ⛏️ Data Extractor Agent - Data extraction

### 🔧 **Specialized Agents**
- ⚡ JavaScript Agent - Browser automation
- 🔐 Authentication Agent - Login handling
- 🛡️ Anti Detection Agent - Stealth measures
- 🔄 Data Transformation Agent - Data processing
- 🚑 Error Recovery Agent - Error handling
- 🖼️ Image Processing Agent - Image analysis
- 🔗 API Integration Agent - API connectivity

## 🎨 Features Showcase

### 🌈 **Beautiful Visual Interface**
- Rich colors and animations
- Progress bars and spinners
- Agent-specific themes and icons
- Interactive tables and panels
- Live status updates

### 🧠 **AI-Powered Intelligence**
- Natural language command processing
- Intelligent command routing
- Context-aware suggestions
- Automatic agent selection
- Smart error recovery

### 🛠️ **Comprehensive Functionality**
- All features from 4 different CLIs
- 18+ interactive commands
- Multiple output formats
- Advanced configuration options
- Session management and history

### 🔧 **Advanced Capabilities**
- JavaScript rendering
- Anti-detection measures
- Data validation and cleaning
- Performance optimization
- Quality assessment
- Document processing
- Real-time monitoring

## 💡 Pro Tips

### 🎯 **Getting Started**
1. Start with `python main.py --interactive` for guided experience
2. Use `help` command to see all available options
3. Try natural language commands for intuitive interaction
4. Use `agents` to see what's available in your system

### 🚀 **Power User Tips**
1. Mix different command styles freely
2. Use configuration profiles for different environments
3. Leverage session history for repeated tasks
4. Monitor system status for optimal performance

### 🧠 **AI Features**
1. Use natural language for complex operations
2. Let the system auto-select appropriate agents
3. Take advantage of intelligent error recovery
4. Use document processing for various file types

## 🔧 Configuration

### 📋 **Profiles**
- `default` - Standard settings
- `development` - Development environment
- `production` - Production environment  
- `testing` - Testing environment

### ⚙️ **Settings**
- Log levels and output formats
- Agent configurations
- Performance tuning
- Security settings

## 🎉 Benefits of Unified CLI

### ✅ **For Users**
- **Single Interface**: No need to remember different CLI commands
- **Intelligent Routing**: System automatically chooses the best approach
- **Beautiful Experience**: Rich visual interface with helpful guidance
- **All Features**: Access to every capability from one place

### ✅ **For Developers**
- **Consistent API**: Unified interface for all functionality
- **Easy Integration**: Single entry point for automation
- **Comprehensive Logging**: Centralized logging and monitoring
- **Extensible Design**: Easy to add new features

### ✅ **For Operations**
- **Simplified Deployment**: One CLI to deploy and manage
- **Unified Configuration**: Single configuration system
- **Centralized Monitoring**: All operations visible in one place
- **Consistent Behavior**: Predictable interface across environments

---

🎉 **The Unified Web Scraper CLI brings together the best of all worlds - beautiful design, comprehensive functionality, AI intelligence, and ease of use - all in one powerful interface!**
