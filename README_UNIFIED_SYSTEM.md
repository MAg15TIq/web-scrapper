# 🚀 Unified Web Scraper System

A comprehensive, integrated web scraping system that combines all CLI interfaces, web interface, and backend services into one cohesive platform.

## 🌟 What's New in the Unified System

### ✅ **Phase 1: Unified Configuration System**
- **Single configuration file** (`config/unified_config.yaml`) for all components
- **Environment variable support** for deployment flexibility
- **Configuration validation** and backup system
- **Real-time configuration updates** across all components

### ✅ **Phase 2: Shared Authentication & Session Management**
- **Unified authentication** across CLI and Web interfaces
- **JWT token-based** session management
- **Role-based access control** (Admin, User, Viewer, API User)
- **Session persistence** and automatic cleanup

### ✅ **Phase 3: Centralized Data Layer**
- **Unified data storage** with SQLite, JSON, or in-memory backends
- **Shared data models** for jobs, tasks, agents, and results
- **Real-time data synchronization** between CLI and Web
- **Advanced querying** and search capabilities

### ✅ **Phase 4: Enhanced Integration Points**
- **Seamless CLI-Web integration** - jobs started in CLI visible in Web interface
- **Real-time event system** for component communication
- **Shared job queue** and monitoring
- **Cross-component notifications**

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified System                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Unified CLI │  │ Web Interface│  │   API       │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Config    │  │    Auth     │  │ Integration │         │
│  │  Manager    │  │  Manager    │  │   Manager   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Unified Data Layer                           │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │ │
│  │  │ SQLite  │  │  JSON   │  │ Memory  │  │  Redis  │   │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                Agent System                             │ │
│  │  Scraper • Parser • Storage • JavaScript • Auth        │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. **Migration from Old System**
```bash
# Migrate your existing configuration and data
python migration/migrate_to_unified.py
```

### 2. **Initialize the Unified System**
```bash
# Initialize all components
python unified_system.py
```

### 3. **Start the Web Interface**
```bash
# Start both backend and frontend
python start_web_interface.py
```

### 4. **Use the Unified CLI**
```bash
# Access all CLI functionality
python main.py --help

# Interactive mode with all features
python main.py interactive

# Direct scraping with unified config
python main.py scrape --url https://example.com
```

## 🔧 Configuration

### **Unified Configuration File**
All system settings are now in `config/unified_config.yaml`:

```yaml
version: "2.0.0"

# Web interface settings
web:
  host: "0.0.0.0"
  port: 8000
  debug: false
  title: "Unified Web Scraper System"

# CLI settings
cli:
  default_profile: "default"
  profiles:
    default:
      name: "default"
      theme: "modern"
      default_output_format: "json"

# Global settings
global_settings:
  log_level: "INFO"
  max_workers: 5
  cache_enabled: true
  output_dir: "output"
  logs_dir: "logs"
```

### **Environment Variables**
```bash
# Web configuration
export WEB_HOST=localhost
export WEB_PORT=8000
export WEB_DEBUG=false

# Authentication
export JWT_SECRET_KEY=your-secret-key

# Global settings
export LOG_LEVEL=INFO
export MAX_WORKERS=10
export OUTPUT_DIR=./output
```

## 🔐 Authentication

### **Default Users**
- **Admin**: `admin` / `admin123` (change immediately!)
- **Demo**: `demo` / `demo123`

### **Creating Users**
```python
from auth.unified_auth import get_unified_auth_manager

auth_manager = get_unified_auth_manager()
user = auth_manager.create_user(
    username="newuser",
    password="secure_password",
    email="user@example.com",
    role=UserRole.USER
)
```

### **CLI Authentication**
```bash
# Login to CLI
python main.py login --username admin

# Check current session
python main.py status --auth
```

## 💾 Data Management

### **Unified Data Layer**
All data (jobs, tasks, results, logs) is stored in a unified data layer:

```python
from data.unified_data_layer import get_unified_data_layer, EntityType

data_layer = get_unified_data_layer()

# Create a job
job_entity = data_layer.create_entity(
    entity_type=EntityType.JOB,
    data={
        "name": "My Scraping Job",
        "status": "running",
        "progress": 50
    }
)

# Query jobs
jobs = data_layer.list_entities(EntityType.JOB)
```

### **Data Backends**
- **SQLite** (default): `data/unified_data.db`
- **JSON**: Individual files in `data/json/`
- **Memory**: In-memory storage for testing

## 🔗 Integration Features

### **CLI-Web Integration**
- Jobs started in CLI appear in Web interface
- Web interface can trigger CLI commands
- Shared configuration and authentication
- Real-time status synchronization

### **Event System**
```python
from integration.unified_integration import get_unified_integration_manager

integration = get_unified_integration_manager()

# Publish events
await integration.publish_event(
    event_type=IntegrationEvent.JOB_CREATED,
    source_component=ComponentType.CLI,
    data={"job_id": "123", "name": "New Job"}
)

# Subscribe to events
integration.subscribe(
    IntegrationEvent.JOB_UPDATED,
    lambda msg: print(f"Job updated: {msg.data}")
)
```

## 📊 Monitoring & Logging

### **System Status**
```bash
# Check system health
python main.py status --system

# View component status
python main.py status --components

# Integration statistics
python main.py status --integration
```

### **Web Dashboard**
- Real-time system metrics
- Job monitoring and management
- Agent status and control
- Configuration management
- User session management

## 🛠️ Development

### **Adding New Components**
1. **Register with Integration Manager**:
```python
from integration.unified_integration import get_unified_integration_manager

integration = get_unified_integration_manager()
integration.register_handler(event_type, your_handler)
```

2. **Use Unified Configuration**:
```python
from config.unified_config import get_unified_config_manager

config = get_unified_config_manager()
my_setting = config.get_global_setting("my_setting", default_value)
```

3. **Store Data in Unified Layer**:
```python
from data.unified_data_layer import get_unified_data_layer

data_layer = get_unified_data_layer()
entity = data_layer.create_entity(EntityType.CUSTOM, your_data)
```

## 🔄 Migration Guide

### **From Old System**
1. **Backup**: Migration automatically creates backups
2. **Configuration**: Old configs are merged into unified config
3. **Data**: Existing jobs and results are migrated
4. **Users**: Default users are created

### **Manual Migration Steps**
```bash
# 1. Run migration
python migration/migrate_to_unified.py

# 2. Verify migration
python unified_system.py

# 3. Test components
python main.py status
curl http://localhost:8000/health

# 4. Update any custom scripts to use unified APIs
```

## 📚 API Reference

### **Unified Configuration**
- `get_unified_config_manager()` - Get config manager
- `get_unified_config()` - Get current configuration
- `get_component_config(component)` - Get component-specific config

### **Unified Authentication**
- `get_unified_auth_manager()` - Get auth manager
- `authenticate_user(username, password)` - Authenticate user
- `create_session(user)` - Create user session
- `validate_token(token)` - Validate JWT token

### **Unified Data Layer**
- `get_unified_data_layer()` - Get data layer
- `create_entity(type, data)` - Create data entity
- `list_entities(type, filters)` - Query entities
- `search_entities(query)` - Search entities

### **Unified Integration**
- `get_unified_integration_manager()` - Get integration manager
- `publish_event(event, component, data)` - Publish event
- `subscribe(event, callback)` - Subscribe to events

## 🎯 Benefits of the Unified System

### **For Users**
- ✅ **Single interface** for all functionality
- ✅ **Consistent authentication** across CLI and Web
- ✅ **Shared job history** and results
- ✅ **Real-time synchronization** between interfaces

### **For Developers**
- ✅ **Unified configuration** management
- ✅ **Shared data models** and storage
- ✅ **Event-driven architecture** for loose coupling
- ✅ **Comprehensive logging** and monitoring

### **For Operations**
- ✅ **Single deployment** process
- ✅ **Centralized monitoring** and health checks
- ✅ **Unified backup** and recovery
- ✅ **Consistent security** model

## 🆘 Troubleshooting

### **Common Issues**

1. **Migration Failed**
   ```bash
   # Check migration logs
   cat logs/migration.log
   
   # Restore from backup if needed
   cp migration/backups/latest/* config/
   ```

2. **Authentication Issues**
   ```bash
   # Reset admin password
   python -c "
   from auth.unified_auth import get_unified_auth_manager
   auth = get_unified_auth_manager()
   admin = auth.get_user_by_username('admin')
   admin.set_password('newpassword')
   auth._save_users()
   "
   ```

3. **Configuration Problems**
   ```bash
   # Validate configuration
   python -c "
   from config.unified_config import get_unified_config_manager
   config = get_unified_config_manager()
   errors = config.validate_configuration()
   print(errors)
   "
   ```

4. **Data Layer Issues**
   ```bash
   # Check data layer status
   python -c "
   from data.unified_data_layer import get_unified_data_layer
   data = get_unified_data_layer()
   print(data.get_statistics())
   "
   ```

## 📞 Support

- **Documentation**: Check this README and inline code documentation
- **Logs**: Check `logs/unified_system.log` for detailed information
- **Migration Report**: Check `migration/backups/migration_report.json`
- **System Status**: Run `python unified_system.py` for health check

---

🎉 **Welcome to the Unified Web Scraper System!** 🎉

The system is now fully integrated and ready for production use. All components work together seamlessly, providing a consistent and powerful web scraping platform.
