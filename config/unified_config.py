"""
Unified Configuration System for Web Scraper
Combines all configuration management into a single, coherent system.
"""
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from pydantic import BaseModel, Field
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings
from enum import Enum

# Import existing config models
from config.web_config import WebConfig, DatabaseConfig, SecurityConfig
from config.langchain_config import LangChainConfig, RedisConfig, MonitoringConfig, AgentSystemConfig
from cli.config_manager import CLIProfile, AgentConfiguration


class ConfigSource(str, Enum):
    """Configuration source types."""
    FILE = "file"
    ENVIRONMENT = "environment"
    DEFAULT = "default"
    OVERRIDE = "override"


class ComponentType(str, Enum):
    """System component types."""
    CLI = "cli"
    WEB = "web"
    API = "api"
    AGENT = "agent"
    DATABASE = "database"
    SECURITY = "security"
    MONITORING = "monitoring"


class UnifiedConfig(BaseModel):
    """Main unified configuration model."""
    
    # Metadata
    version: str = Field("2.0.0", description="Configuration version")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    config_source: ConfigSource = ConfigSource.DEFAULT
    
    # Component configurations
    web: WebConfig = Field(default_factory=WebConfig)
    langchain: LangChainConfig = Field(default_factory=LangChainConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    agent_system: AgentSystemConfig = Field(default_factory=AgentSystemConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # CLI configuration
    cli: Dict[str, Any] = Field(default_factory=dict)
    
    # Global settings
    global_settings: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        validate_assignment = True


class UnifiedConfigManager:
    """Unified configuration manager for all system components."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the unified configuration manager."""
        self.logger = logging.getLogger("unified_config")
        
        # Configuration paths
        self.config_dir = Path(config_dir or "config")
        self.config_file = self.config_dir / "unified_config.yaml"
        self.env_file = Path(".env")
        self.backup_dir = self.config_dir / "backups"
        
        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Configuration state
        self._config: Optional[UnifiedConfig] = None
        self._watchers: Dict[str, List[callable]] = {}
        self._component_configs: Dict[ComponentType, Dict[str, Any]] = {}
        
        # Load configuration
        self._load_configuration()
        
        self.logger.info(f"Unified configuration manager initialized (config: {self.config_file})")
    
    def _load_configuration(self) -> None:
        """Load configuration from all sources."""
        try:
            # Start with defaults
            config_data = self._get_default_config()
            
            # Load from file if exists
            if self.config_file.exists():
                file_config = self._load_from_file()
                config_data = self._deep_merge(config_data, file_config)
            
            # Override with environment variables
            env_config = self._load_from_environment()
            config_data = self._deep_merge(config_data, env_config)
            
            # Create unified config object
            self._config = UnifiedConfig(**config_data)
            self._config.updated_at = datetime.now()
            
            # Update component configurations
            self._update_component_configs()
            
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Fall back to defaults
            self._config = UnifiedConfig()
            self._update_component_configs()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "version": "2.0.0",
            "web": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
                "title": "Unified Web Scraper System"
            },
            "cli": {
                "default_profile": "default",
                "profiles": {
                    "default": {
                        "name": "default",
                        "description": "Default unified profile",
                        "default_output_format": "json",
                        "default_agents": ["scraper", "parser", "storage"],
                        "theme": "modern"
                    }
                },
                "agents": {
                    "scraper": {
                        "agent_type": "scraper",
                        "config": {"timeout": 30, "retries": 3}
                    },
                    "parser": {
                        "agent_type": "parser", 
                        "config": {"normalize_data": True}
                    },
                    "storage": {
                        "agent_type": "storage",
                        "config": {"default_format": "json"}
                    }
                }
            },
            "global_settings": {
                "log_level": "INFO",
                "max_workers": 5,
                "cache_enabled": True,
                "output_dir": "output",
                "logs_dir": "logs"
            }
        }

    def _load_from_file(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.suffix.lower() == '.yaml' or self.config_file.suffix.lower() == '.yml':
                    return yaml.safe_load(f) or {}
                else:
                    return json.load(f) or {}
        except Exception as e:
            self.logger.error(f"Failed to load config file {self.config_file}: {e}")
            return {}

    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}

        # Web configuration
        if os.getenv("WEB_HOST"):
            env_config.setdefault("web", {})["host"] = os.getenv("WEB_HOST")
        if os.getenv("WEB_PORT"):
            env_config.setdefault("web", {})["port"] = int(os.getenv("WEB_PORT"))
        if os.getenv("WEB_DEBUG"):
            env_config.setdefault("web", {})["debug"] = os.getenv("WEB_DEBUG").lower() == "true"

        # Global settings
        if os.getenv("LOG_LEVEL"):
            env_config.setdefault("global_settings", {})["log_level"] = os.getenv("LOG_LEVEL")
        if os.getenv("MAX_WORKERS"):
            env_config.setdefault("global_settings", {})["max_workers"] = int(os.getenv("MAX_WORKERS"))
        if os.getenv("OUTPUT_DIR"):
            env_config.setdefault("global_settings", {})["output_dir"] = os.getenv("OUTPUT_DIR")

        return env_config

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _update_component_configs(self) -> None:
        """Update component-specific configurations."""
        if not self._config:
            return

        # Update component configs
        self._component_configs[ComponentType.WEB] = self._config.web.model_dump()
        self._component_configs[ComponentType.CLI] = self._config.cli
        self._component_configs[ComponentType.DATABASE] = self._config.database.model_dump()
        self._component_configs[ComponentType.SECURITY] = self._config.security.model_dump()
        self._component_configs[ComponentType.MONITORING] = self._config.monitoring.model_dump()
        self._component_configs[ComponentType.AGENT] = self._config.agent_system.model_dump()

    def get_config(self) -> UnifiedConfig:
        """Get the unified configuration."""
        if not self._config:
            self._load_configuration()
        return self._config

    def get_component_config(self, component: ComponentType) -> Dict[str, Any]:
        """Get configuration for a specific component."""
        return self._component_configs.get(component, {})

    def get_web_config(self) -> WebConfig:
        """Get web configuration."""
        return self._config.web if self._config else WebConfig()

    def get_cli_config(self) -> Dict[str, Any]:
        """Get CLI configuration."""
        return self._config.cli if self._config else {}

    def get_global_setting(self, key: str, default: Any = None) -> Any:
        """Get a global setting value."""
        if not self._config:
            return default
        return self._config.global_settings.get(key, default)

    def set_global_setting(self, key: str, value: Any) -> None:
        """Set a global setting value."""
        if not self._config:
            self._load_configuration()

        self._config.global_settings[key] = value
        self._config.updated_at = datetime.now()

        # Notify watchers
        self._notify_watchers("global_settings", key, value)

    def save_configuration(self, backup: bool = True) -> bool:
        """Save configuration to file."""
        try:
            if backup and self.config_file.exists():
                self._create_backup()

            # Convert config to dict
            config_dict = self._config.model_dump() if self._config else self._get_default_config()

            # Remove computed fields
            config_dict.pop('created_at', None)
            config_dict.pop('updated_at', None)

            # Save to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            self.logger.info(f"Configuration saved to {self.config_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False

    def _create_backup(self) -> None:
        """Create a backup of the current configuration."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"unified_config_{timestamp}.yaml"

            import shutil
            shutil.copy2(self.config_file, backup_file)

            self.logger.info(f"Configuration backup created: {backup_file}")

        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")

    def add_watcher(self, component: str, callback: callable) -> None:
        """Add a configuration change watcher."""
        if component not in self._watchers:
            self._watchers[component] = []
        self._watchers[component].append(callback)

    def remove_watcher(self, component: str, callback: callable) -> None:
        """Remove a configuration change watcher."""
        if component in self._watchers:
            try:
                self._watchers[component].remove(callback)
            except ValueError:
                pass

    def _notify_watchers(self, component: str, key: str, value: Any) -> None:
        """Notify watchers of configuration changes."""
        if component in self._watchers:
            for callback in self._watchers[component]:
                try:
                    callback(key, value)
                except Exception as e:
                    self.logger.error(f"Error in config watcher callback: {e}")

    def reload_configuration(self) -> bool:
        """Reload configuration from all sources."""
        try:
            self._load_configuration()
            self.logger.info("Configuration reloaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            return False

    def validate_configuration(self) -> List[str]:
        """Validate the current configuration and return any errors."""
        errors = []

        if not self._config:
            errors.append("No configuration loaded")
            return errors

        # Validate web config
        try:
            if self._config.web.port < 1 or self._config.web.port > 65535:
                errors.append("Web port must be between 1 and 65535")
        except Exception as e:
            errors.append(f"Invalid web configuration: {e}")

        # Validate global settings
        try:
            if self._config.global_settings.get("max_workers", 1) < 1:
                errors.append("max_workers must be at least 1")
        except Exception as e:
            errors.append(f"Invalid global settings: {e}")

        return errors


# Global unified configuration manager instance
_unified_config_manager: Optional[UnifiedConfigManager] = None


def get_unified_config_manager() -> UnifiedConfigManager:
    """Get the global unified configuration manager instance."""
    global _unified_config_manager
    if _unified_config_manager is None:
        _unified_config_manager = UnifiedConfigManager()
    return _unified_config_manager


def get_unified_config() -> UnifiedConfig:
    """Get the unified configuration."""
    return get_unified_config_manager().get_config()


def get_component_config(component: ComponentType) -> Dict[str, Any]:
    """Get configuration for a specific component."""
    return get_unified_config_manager().get_component_config(component)
