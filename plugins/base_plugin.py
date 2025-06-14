"""
Base Plugin Class for Web Scraper Plugin System
"""

import abc
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from pydantic import BaseModel, Field


class PluginType(str, Enum):
    """Types of plugins supported by the system."""
    AGENT = "agent"
    PROCESSOR = "processor"
    EXTRACTOR = "extractor"
    TRANSFORMER = "transformer"
    VALIDATOR = "validator"
    EXPORTER = "exporter"
    MIDDLEWARE = "middleware"
    INTEGRATION = "integration"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True
    priority: int = 100  # Lower numbers = higher priority


class PluginConfig(BaseModel):
    """Configuration for a plugin instance."""
    enabled: bool = True
    settings: Dict[str, Any] = Field(default_factory=dict)
    environment: str = "production"
    debug: bool = False


class PluginResult(BaseModel):
    """Result from plugin execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: Optional[float] = None


class BasePlugin(abc.ABC):
    """
    Base class for all plugins in the web scraper system.
    
    Plugins extend the functionality of the web scraper by providing
    custom agents, processors, extractors, and other components.
    """
    
    def __init__(self, config: Optional[PluginConfig] = None):
        """Initialize the plugin."""
        self.config = config or PluginConfig()
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self._initialized = False
        self._running = False
        
    @property
    @abc.abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abc.abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the plugin.
        
        Returns:
            True if initialization was successful, False otherwise.
        """
        pass
    
    @abc.abstractmethod
    async def execute(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> PluginResult:
        """
        Execute the plugin's main functionality.
        
        Args:
            input_data: Input data for the plugin
            context: Optional execution context
            
        Returns:
            PluginResult with execution results
        """
        pass
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        self._running = False
        self.logger.info(f"Plugin {self.metadata.name} cleaned up")
    
    async def validate_config(self) -> bool:
        """
        Validate the plugin configuration.
        
        Returns:
            True if configuration is valid, False otherwise.
        """
        try:
            if self.metadata.config_schema:
                # TODO: Implement JSON schema validation
                pass
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the plugin.
        
        Returns:
            Health status information
        """
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "running": self._running,
            "metadata": {
                "name": self.metadata.name,
                "version": self.metadata.version,
                "type": self.metadata.plugin_type.value
            }
        }
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.settings.get(key, default)
    
    def set_config_value(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self.config.settings[key] = value
    
    def is_enabled(self) -> bool:
        """Check if the plugin is enabled."""
        return self.config.enabled and self.metadata.enabled
    
    def __str__(self) -> str:
        """String representation of the plugin."""
        return f"{self.metadata.name} v{self.metadata.version} ({self.metadata.plugin_type.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the plugin."""
        return (
            f"<{self.__class__.__name__}("
            f"name='{self.metadata.name}', "
            f"version='{self.metadata.version}', "
            f"type='{self.metadata.plugin_type.value}', "
            f"enabled={self.is_enabled()}"
            f")>"
        )


class AgentPlugin(BasePlugin):
    """Base class for agent plugins."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description="Agent plugin",
            author="Unknown",
            plugin_type=PluginType.AGENT
        )


class ProcessorPlugin(BasePlugin):
    """Base class for processor plugins."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0", 
            description="Processor plugin",
            author="Unknown",
            plugin_type=PluginType.PROCESSOR
        )


class ExtractorPlugin(BasePlugin):
    """Base class for extractor plugins."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description="Extractor plugin", 
            author="Unknown",
            plugin_type=PluginType.EXTRACTOR
        )


class TransformerPlugin(BasePlugin):
    """Base class for transformer plugins."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description="Transformer plugin",
            author="Unknown", 
            plugin_type=PluginType.TRANSFORMER
        )
