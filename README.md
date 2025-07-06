# 🚀 Unified Multi-Agent Web Scraping System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-brightgreen.svg)](https://github.com/MAg15TIq/web-scrapper)
[![Enterprise Ready](https://img.shields.io/badge/Enterprise-Ready-blue.svg)](https://github.com/MAg15TIq/web-scrapper)
[![Unified System](https://img.shields.io/badge/Unified-System-purple.svg)](https://github.com/MAg15TIq/web-scrapper)
[![Phase 5 Complete](https://img.shields.io/badge/Phase%205-Complete-success.svg)](https://github.com/MAg15TIq/web-scrapper)
[![All Phases Complete](https://img.shields.io/badge/All%20Phases-Complete-gold.svg)](https://github.com/MAg15TIq/web-scrapper)

A comprehensive, enterprise-grade AI-powered web scraping platform featuring 30+ specialized agents, advanced security & compliance, and intelligent AI orchestration. Built for CLI-first workflows with LangChain for AI reasoning, Pydantic for data validation, and modern Python technologies for enterprise scalability.

---

## 🆕 New Features

- **Anti-Detection Agent**: Browser fingerprint randomization, request pattern variation, blocking detection, and adaptive request optimization.
- **Data Transformation Agent**: Data cleaning, normalization, schema transformation, entity extraction, and text analysis (sentiment, keywords, etc).
- **Enhanced Error Recovery Agent**: Pattern-based failure prediction, adaptive retry strategies, self-healing, and proactive error prevention.
- **Enhanced Quality Assurance Agent**: AI-powered quality assessment, anomaly detection, intelligent data validation, and adaptive quality thresholds.
- **Enhanced Content Recognition Agent**: ML-based content classification, pattern recognition, and content quality assessment.
- **Enhanced NLP Agent**: Advanced NLP tasks including entity extraction, sentiment analysis, summarization, translation, and emotion analysis.
- **Advanced Pagination Engine**: Hybrid detection (URL, DOM, JS), browser support, and seamless integration with scraping workflows.
- **Distributed Result Aggregation**: Efficient merging, conflict resolution, and deduplication using Apache Arrow and semantic similarity.
- **API Integration Agent**: Seamless interaction with external APIs, supporting authentication, pagination, and schema transformation.
- **Live Monitoring Dashboard**: Real-time scraping progress, agent/task stats, and visual metrics using the `rich` library.

See `docs/new_features.md` and `docs/specialized_agents.md` for more details and usage examples.

---

## 🌟 Features

- **30+ Specialized Agents**: Intelligence, Security, Data Processing, Enterprise Scalability, Advanced NLP, Computer Vision.
- **Fully CLI-Based**: All operations, configuration, and monitoring are performed via a unified command-line interface.
- **Centralized Configuration**: Single YAML configuration system managing all components.
- **Unified Data Layer**: Centralized SQLite database with real-time synchronization.
- **Enterprise Security**: ML anti-detection, GDPR compliance, advanced encryption, audit logging.
- **Advanced Data Processing**: Enhanced NLP, computer vision, geocoding, and data enrichment.
- **Distributed Architecture**: Horizontal scaling, load balancing, multi-tenant support.
- **Plugin System**: Extensible architecture with plugin marketplace and custom processors.

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Agent System](#agent-system)
- [CLI Usage](#cli-usage)
- [Configuration](#configuration)
- [Security & Compliance](#security--compliance)
- [Performance & Scalability](#performance--scalability)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

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
python -m venv venv
# Activate virtual environment
# On Windows (cmd):
venv\Scripts\activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install  # For JavaScript rendering
python -m spacy download en_core_web_sm  # Basic NLP
# Optional: For advanced NLP, ML, and OCR features
pip install scikit-learn textblob transformers langdetect googletrans==4.0.0rc1
# For OCR (Image Processing Agent):
# On Windows: Download and install Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH
# On Linux: sudo apt-get install tesseract-ocr libtesseract-dev
```

### Environment Setup
```bash
cp .env.example .env
# Edit .env with your configuration
```

---

## ⚡ Quick Start

### 1. System Initialization
```bash
python unified_system_simple.py
```

### 2. CLI Usage
```bash
python cli.py --help
python cli.py agents list
python cli.py jobs create --url https://example.com
python cli.py monitoring status
```

---

## 🤖 Agent System

- **Coordinator Agent**: Orchestrates all scraping operations.
- **Scraper Agent**: Handles HTTP requests and content fetching.
- **JavaScript Agent**: Renders JavaScript-heavy pages.
- **Parser Agent**: Extracts structured data from HTML/XML.
- **Storage Agent**: Manages data export and storage.
- **Authentication Agent**: Handles login and session management.
- **Anti-Detection Agent**: Implements stealth and anti-detection measures.
- **Data Transformation Agent**: Cleans and transforms extracted data.
- **Error Recovery Agent**: Handles errors and implements recovery strategies.
- **Monitoring Agent**: Tracks system performance and health.
- **Enhanced Error Recovery Agent**: Predictive, self-healing error handling.
- **Enhanced Quality Assurance Agent**: AI-powered data quality and anomaly detection.
- **Enhanced Content Recognition Agent**: ML-based content classification and quality.
- **Enhanced NLP Agent**: Advanced NLP, translation, and emotion analysis.
- **API Integration Agent**: External API data integration.
- ...and many more specialized agents for NLP, computer vision, geocoding, and more.

---

## 🖥️ CLI Usage

All system operations are performed via the CLI. Example commands:

```bash
python cli.py agents list
python cli.py jobs create --url https://example.com
python cli.py monitoring status
python cli.py plugins list
```

See `python cli.py --help` for all available commands.

---

## ⚙️ Configuration

- All configuration is managed via a single YAML file: `config/unified_config.yaml`.
- Environment variables can be set in `.env`.

---

## 🛡️ Security & Compliance

- **JWT Authentication**: Secure login and CLI access.
- **Role-Based Access Control**: Admin, user, and viewer roles.
- **ML Anti-Detection**: Behavioral mimicking, fingerprint randomization, proxy rotation.
- **GDPR Compliance**: PII detection, data anonymization, retention policies, audit logging.
- **Advanced Encryption**: End-to-end encryption with key rotation.

---

## 📈 Performance & Scalability

- **Distributed Architecture**: Horizontal scaling, load balancing, multi-tenant support.
- **Health Monitoring**: Automatic failover, node health monitoring, performance metrics.
- **Plugin System**: Extensible with plugin marketplace and custom processors.

---

## 🧪 Testing

```bash
pip install pytest pytest-cov pytest-asyncio
pytest
```

> **Note:** Some advanced features require optional dependencies (see Install Dependencies above).

---

## 🛠️ Troubleshooting

- **Authentication Issues**: Use correct credentials, check backend status.
- **Port Conflicts**: The system auto-detects free ports, but you can manually specify ports if needed.

---

## 🏗️ Development

- **Project Structure**:
  - `agents/` - Agent implementations
  - `config/` - Configuration files
  - `models/` - Data models
  - `monitoring/` - Monitoring components
  - `tests/` - Test suites
  - `logs/` - Log files
  - `requirements.txt` - Python dependencies

- **Code Style**:
```bash
black .
isort .
flake8 .
mypy .
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 👥 Contributors

- **WebScraper Team**
- _Your Name Here_

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for AI orchestration and intelligent agent coordination
- [Rich](https://rich.readthedocs.io/) for beautiful and interactive CLI interfaces
- [Pydantic](https://pydantic-docs.helpmanual.io/) for robust data validation and settings management
- [Playwright](https://playwright.dev/) for reliable browser automation and JavaScript rendering
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing and data extraction

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/MAg15TIq/web-scrapper/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MAg15TIq/web-scrapper/discussions)
- **Documentation**: [Project Wiki](https://github.com/MAg15TIq/web-scrapper/wiki)
