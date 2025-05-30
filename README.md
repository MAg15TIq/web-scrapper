# 🚀 Multi-Agent Web Scraping System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, AI-powered web scraping system featuring multiple specialized agents, intelligent orchestration, and both CLI and Web interfaces. Built with LangChain for AI capabilities and Pydantic for data validation.

## 🌟 Features

### 🤖 Multi-Agent Architecture
- **5 Specialized Agents**: Orchestrator, Web Scraping, Document Processing, Data Transformation, and Data Output
- **AI-Powered Coordination**: LangChain integration for intelligent decision making
- **Scalable Design**: Easily extensible with new agent types

### 🖥️ Dual Interface System
- **Enhanced CLI**: Natural language processing, interactive modes, real-time progress
- **Web API**: RESTful endpoints, WebSocket support, comprehensive documentation
- **Real-time Monitoring**: Live agent status, job progress, system metrics

### 🛡️ Enterprise Features
- **Authentication & Authorization**: JWT-based security with role management
- **Rate Limiting**: Configurable limits to prevent abuse
- **Job Scheduling**: Priority-based queue with recurring tasks
- **Comprehensive Logging**: Detailed audit trails and error tracking

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Clone Repository

```bash
git clone https://github.com/MAg15TIq/web-scrapper.git
cd web-scrapper
```

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
# Add your API keys and configuration
```

## ⚡ Quick Start

### 1. Start the Enhanced CLI

```bash
# Interactive mode with AI assistance
python -m cli.enhanced_interface --interactive

# Use natural language commands
> "Scrape product prices from amazon.com"
> "Extract news articles from techcrunch.com"
> "Monitor system status"
```

### 2. Launch Web API Server

```bash
# Development mode
python web/api/main.py

# The API will be available at:
# - Main API: http://localhost:8000
# - Documentation: http://localhost:8000/api/docs
# - Alternative docs: http://localhost:8000/api/redoc
```

### 3. Basic Web API Usage

```bash
# Health check
curl http://localhost:8000/health

# List agents
curl http://localhost:8000/api/v1/agents

# Create a scraping job
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Sample Scraping Job",
    "job_type": "scrape",
    "parameters": {
      "url": "https://example.com",
      "selectors": ["title", "description"]
    }
  }'
```

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Enhanced CLI  │    │   Web Frontend  │    │   Mobile App    │
│  (Natural Lang) │    │   (React/Vue)   │    │   (Future)      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      FastAPI Server       │
                    │   • Authentication        │
                    │   • Rate Limiting         │
                    │   • WebSocket Support     │
                    │   • Job Management        │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │   LangChain Framework     │
                    │   • AI Orchestration      │
                    │   • Natural Language      │
                    │   • Decision Making       │
                    └─────────────┬─────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────┴───────┐    ┌───────────┴────────────┐    ┌───────┴───────┐
│ Orchestrator  │    │    Specialized         │    │  Data Output  │
│    Agent      │    │      Agents            │    │    Agent      │
│               │    │ • Web Scraping         │    │               │
│ • Coordinates │    │ • Document Processing  │    │ • JSON Export │
│ • Plans       │    │ • Data Transformation  │    │ • CSV Export  │
│ • Monitors    │    │ • Error Recovery       │    │ • Database    │
└───────────────┘    └────────────────────────┘    └───────────────┘
```

### Agent Types

1. **Orchestrator Agent** 🎯
   - Coordinates all scraping operations
   - Plans execution strategies
   - Monitors system health

2. **Web Scraping Agent** 🕷️
   - Handles HTTP requests
   - Manages sessions and cookies
   - Implements anti-detection measures

3. **Document Processing Agent** 📄
   - Processes PDFs and documents
   - Extracts text and metadata
   - Handles various file formats

4. **Data Transformation Agent** 🔄
   - Cleans and normalizes data
   - Applies transformation rules
   - Validates data quality

5. **Data Output Agent** 💾
   - Manages data export
   - Supports multiple formats
   - Handles database operations

## 📖 Usage

### Enhanced CLI

#### Interactive Mode
```bash
# Start interactive session
python -m cli.enhanced_interface -i

# Natural language commands
> "Scrape all product reviews from the first 5 pages of amazon.com"
> "Extract contact information from company websites"
> "Set up scheduled scraping for news sites every hour"
```

#### Direct Commands
```bash
# List available agents
python -m cli.enhanced_interface agents

# Check system status
python -m cli.enhanced_interface status

# View configuration
python -m cli.enhanced_interface config

# Show command history
python -m cli.enhanced_interface history
```

### Web API

#### Authentication
```bash
# Login to get access token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"

# Use token in subsequent requests
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/agents
```

#### Job Management
```bash
# Create a scraping job
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "E-commerce Product Scraping",
    "job_type": "scrape",
    "priority": "high",
    "parameters": {
      "url": "https://example-store.com/products",
      "selectors": {
        "title": ".product-title",
        "price": ".price",
        "description": ".product-description"
      },
      "max_pages": 10,
      "delay": 2
    },
    "tags": ["ecommerce", "products"]
  }'

# Monitor job progress
curl http://localhost:8000/api/v1/jobs/{job_id}

# Get job statistics
curl http://localhost:8000/api/v1/jobs/stats/summary
```

#### Real-time Monitoring
```javascript
// WebSocket connection for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/client_123');

// Subscribe to updates
ws.send(JSON.stringify({
    type: 'subscribe',
    payload: {
        channels: ['agent_status', 'job_updates', 'system_metrics']
    }
}));

// Handle incoming messages
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};
```

## 📚 API Documentation

### Core Endpoints

#### Agents
- `GET /api/v1/agents` - List all agents
- `GET /api/v1/agents/{agent_id}` - Get agent details
- `PUT /api/v1/agents/{agent_id}/configure` - Configure agent
- `POST /api/v1/agents/{agent_id}/restart` - Restart agent

#### Jobs
- `POST /api/v1/jobs` - Create new job
- `GET /api/v1/jobs` - List jobs with filtering
- `GET /api/v1/jobs/{job_id}` - Get job details
- `POST /api/v1/jobs/{job_id}/cancel` - Cancel job

#### Monitoring
- `GET /api/v1/monitoring/health` - System health
- `GET /api/v1/monitoring/metrics/system` - System metrics
- `GET /api/v1/monitoring/metrics/agents` - Agent metrics
- `GET /api/v1/monitoring/alerts` - System alerts

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **OpenAPI Schema**: http://localhost:8000/api/openapi.json

## ⚙️ Configuration

### Environment Variables

```bash
# Web Server
WEB_HOST=0.0.0.0
WEB_PORT=8000
WEB_DEBUG=false

# Security
SECRET_KEY=your-super-secret-key-here
OPENAI_API_KEY=your-openai-api-key

# Database
DATABASE_URL=sqlite:///./webscraper.db
REDIS_URL=redis://localhost:6379/0

# Features
WEB_ENABLE_AUTH=true
WEB_ENABLE_RATE_LIMITING=true
WEB_ENABLE_WEBSOCKETS=true
```

### CLI Configuration

```yaml
# ~/.webscraper_cli/config.yaml
version: "1.0.0"
default_profile: "default"

profiles:
  default:
    name: "default"
    default_output_format: "json"
    auto_confirm: false
    theme: "default"

logging:
  level: "INFO"
  file: "logs/cli.log"
```

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest tests/test_cli/
pytest tests/test_web_api/
pytest tests/test_integration/
```

### Manual Testing
```bash
# Test CLI
python -m cli.enhanced_interface --help

# Test API health
curl http://localhost:8000/health

# Test WebSocket
# Use a WebSocket client to connect to ws://localhost:8000/ws/test
```

## 🛠️ Development

### Project Structure
```
web-scrapper/
├── agents/                 # Agent implementations
├── cli/                   # Enhanced CLI components
├── web/                   # Web API and dashboard
├── config/                # Configuration files
├── models/                # Data models
├── monitoring/            # Monitoring components
├── tests/                 # Test suites
├── logs/                  # Log files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Adding New Features

1. **New Agent Type**
   - Create agent class in `agents/`
   - Register with coordinator
   - Add configuration options

2. **New API Endpoint**
   - Add route in `web/api/routes/`
   - Update dependencies if needed
   - Add tests

3. **CLI Enhancement**
   - Extend command parser
   - Add new commands
   - Update help documentation

### Code Style
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy .
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation
- Ensure all tests pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for AI orchestration
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Rich](https://rich.readthedocs.io/) for beautiful CLI interfaces
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation

## 📞 Support

- 📧 Email: support@webscraper.com
- 🐛 Issues: [GitHub Issues](https://github.com/MAg15TIq/web-scrapper/issues)
- 📖 Documentation: [Wiki](https://github.com/MAg15TIq/web-scrapper/wiki)

---

**Made with ❤️ by the Web Scraper Team**
