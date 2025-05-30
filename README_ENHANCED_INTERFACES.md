# Enhanced CLI and Web Interface System

## 🚀 Overview

This document describes the enhanced Command Line Interface (CLI) and Web interface components for the Multi-Agent Web Scraping System. The system now features advanced AI-powered interfaces with LangChain integration, real-time monitoring, and comprehensive job management.

## 📋 System Architecture

### Enhanced CLI Components

```
cli/
├── enhanced_interface.py          # Main enhanced CLI with AI integration
├── command_parser.py              # Advanced command parsing with Pydantic
├── config_manager.py              # Configuration management system
├── session_manager.py             # Session persistence and history
├── progress_manager.py            # Advanced progress tracking
├── langchain_cli_adapter.py       # LangChain integration for NLP
└── agent_communication.py         # Agent communication layer
```

### Web Interface Components

```
web/
├── api/
│   ├── main.py                    # FastAPI application entry point
│   ├── dependencies.py            # Common API dependencies
│   ├── routes/
│   │   ├── agents.py              # Agent management endpoints
│   │   ├── jobs.py                # Job management endpoints
│   │   ├── monitoring.py          # System monitoring endpoints
│   │   ├── auth.py                # Authentication endpoints
│   │   └── websockets.py          # WebSocket real-time communication
│   └── middleware/
│       ├── auth.py                # Authentication middleware
│       └── rate_limiting.py       # Rate limiting middleware
├── dashboard/
│   └── agent_monitor.py           # Real-time agent monitoring
├── scheduler/
│   └── job_manager.py             # Job scheduling and execution
└── frontend/                      # React frontend (to be implemented)
```

## 🎯 Key Features

### Enhanced CLI Features

1. **Natural Language Processing**
   - AI-powered command interpretation using LangChain
   - Context-aware command suggestions
   - Intelligent parameter extraction

2. **Advanced Progress Tracking**
   - Real-time progress indicators with Rich library
   - Agent-specific progress visualization
   - Desktop notifications for long-running tasks

3. **Session Management**
   - Command history persistence
   - Session state management
   - Context preservation across sessions

4. **Configuration Management**
   - Profile-based configurations
   - Environment-specific settings
   - Dynamic configuration updates

### Web Interface Features

1. **RESTful API**
   - Comprehensive REST endpoints for all operations
   - OpenAPI/Swagger documentation
   - Pydantic data validation

2. **Real-time Communication**
   - WebSocket support for live updates
   - Real-time agent status monitoring
   - Live job progress tracking

3. **Authentication & Security**
   - JWT-based authentication
   - Role-based access control
   - Rate limiting and security headers

4. **Job Management**
   - Priority-based job queuing
   - Scheduled and recurring jobs
   - Comprehensive job monitoring

## 🛠️ Installation and Setup

### Prerequisites

```bash
# Install required dependencies
pip install -r requirements.txt

# Additional dependencies for enhanced features
pip install fastapi uvicorn websockets
pip install rich questionary textual
pip install psutil
```

### Configuration

1. **Environment Variables**
```bash
# Web configuration
export WEB_HOST=0.0.0.0
export WEB_PORT=8000
export WEB_DEBUG=false

# Security
export SECRET_KEY=your-secret-key-here
export OPENAI_API_KEY=your-openai-key-here

# Database
export DATABASE_URL=sqlite:///./webscraper.db
export REDIS_URL=redis://localhost:6379/0
```

2. **Configuration Files**
```bash
# Create configuration directory
mkdir -p ~/.webscraper_cli

# Copy default configurations
cp config/default_config.yaml ~/.webscraper_cli/config.yaml
```

## 🚀 Usage

### Enhanced CLI

1. **Interactive Mode**
```bash
# Start interactive mode with AI assistance
python -m cli.enhanced_interface --interactive

# Use natural language commands
> "Scrape product prices from amazon.com"
> "Extract news articles from techcrunch.com"
> "Monitor job queue status"
```

2. **Direct Commands**
```bash
# List available agents
python -m cli.enhanced_interface agents

# Check system status
python -m cli.enhanced_interface status

# View command history
python -m cli.enhanced_interface history
```

### Web Interface

1. **Start the Web Server**
```bash
# Development mode
python web/api/main.py

# Production mode with Gunicorn
gunicorn web.api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

2. **API Documentation**
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`
- OpenAPI Schema: `http://localhost:8000/api/openapi.json`

3. **WebSocket Connection**
```javascript
// Connect to WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/client_123');

// Subscribe to agent status updates
ws.send(JSON.stringify({
    type: 'subscribe',
    payload: { channels: ['agent_status', 'job_updates'] }
}));
```

## 📊 API Endpoints

### Agent Management
- `GET /api/v1/agents` - List all agents
- `GET /api/v1/agents/{agent_id}` - Get agent details
- `PUT /api/v1/agents/{agent_id}/configure` - Configure agent
- `POST /api/v1/agents/{agent_id}/tasks` - Submit task to agent
- `POST /api/v1/agents/{agent_id}/restart` - Restart agent

### Job Management
- `POST /api/v1/jobs` - Create new job
- `GET /api/v1/jobs` - List jobs with filtering
- `GET /api/v1/jobs/{job_id}` - Get job details
- `POST /api/v1/jobs/{job_id}/cancel` - Cancel job
- `GET /api/v1/jobs/stats/summary` - Get job statistics

### Monitoring
- `GET /api/v1/monitoring/health` - System health check
- `GET /api/v1/monitoring/metrics/system` - System metrics
- `GET /api/v1/monitoring/metrics/agents` - Agent metrics
- `GET /api/v1/monitoring/alerts` - System alerts
- `GET /api/v1/monitoring/logs` - System logs

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/register` - User registration
- `GET /api/v1/auth/profile` - Get user profile
- `POST /api/v1/auth/logout` - User logout

## 🔧 Configuration

### CLI Configuration

```yaml
# ~/.webscraper_cli/config.yaml
version: "1.0.0"
default_profile: "default"

profiles:
  default:
    name: "default"
    description: "Default CLI profile"
    default_output_format: "json"
    default_agents: ["Scraper Agent", "Parser Agent", "Storage Agent"]
    auto_confirm: false
    verbose_logging: false
    theme: "default"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/cli.log"
```

### Web Configuration

```python
# config/web_config.py
class WebConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str = "your-secret-key"
    enable_auth: bool = True
    enable_rate_limiting: bool = True
    
    # Database
    database_url: str = "sqlite:///./webscraper.db"
    redis_url: str = "redis://localhost:6379/0"
```

## 🔍 Monitoring and Metrics

### Real-time Monitoring

The system provides comprehensive monitoring capabilities:

1. **Agent Metrics**
   - CPU and memory usage per agent
   - Task completion rates
   - Response times and throughput
   - Error rates and failure analysis

2. **System Metrics**
   - Overall system performance
   - Resource utilization
   - Network I/O statistics
   - Active connections and load

3. **Job Metrics**
   - Job queue sizes by priority
   - Success/failure rates
   - Average execution times
   - Scheduling efficiency

### Alerting

Configurable alerts for:
- High CPU/memory usage
- Agent failures or timeouts
- Job queue backlogs
- System errors and exceptions

## 🧪 Testing

### CLI Testing
```bash
# Run CLI tests
python -m pytest tests/test_cli/ -v

# Test interactive mode
python -m cli.enhanced_interface --interactive --config tests/test_config.yaml
```

### API Testing
```bash
# Run API tests
python -m pytest tests/test_web_api/ -v

# Test with coverage
python -m pytest tests/ --cov=web --cov-report=html
```

### Integration Testing
```bash
# Run full integration tests
python -m pytest tests/test_integration/ -v
```

## 🚀 Next Steps

### Phase 1: Frontend Development
1. Create React frontend application
2. Implement dashboard components
3. Add real-time data visualization
4. Create agent management interface

### Phase 2: Advanced Features
1. Implement workflow builder
2. Add data export capabilities
3. Create scheduling interface
4. Implement user management

### Phase 3: Production Deployment
1. Docker containerization
2. Kubernetes deployment manifests
3. CI/CD pipeline setup
4. Production monitoring setup

## 📚 Documentation

- [API Documentation](http://localhost:8000/api/docs)
- [CLI User Guide](docs/cli_guide.md)
- [Web Interface Guide](docs/web_guide.md)
- [Developer Documentation](docs/developer_guide.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
