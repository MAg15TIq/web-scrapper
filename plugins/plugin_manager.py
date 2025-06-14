"""
Plugin Manager for Web Scraper Plugin System
Handles plugin discovery, loading, and lifecycle management.
"""

import os
import sys
import importlib
import importlib.util
import logging
import asyncio
from typing import Dict, List, Optional, Type, Any, Set
from pathlib import Path
import json
import yaml

from .base_plugin import BasePlugin, PluginType, PluginMetadata, PluginConfig, PluginResult
from .plugin_registry import PluginRegistry


class PluginManager:
    """
    Manages the lifecycle of plugins in the web scraper system.
    
    Handles plugin discovery, loading, initialization, execution, and cleanup.
    """
    
    def __init__(self, plugin_directories: Optional[List[str]] = None):
        """
        Initialize the plugin manager.
        
        Args:
            plugin_directories: List of directories to search for plugins
        """
        self.logger = logging.getLogger("plugin_manager")
        self.registry = PluginRegistry()
        self.loaded_plugins: Dict[str, BasePlugin] = {}
        self.plugin_configs: Dict[str, PluginConfig] = {}
        
        # Default plugin directories
        self.plugin_directories = plugin_directories or [
            "plugins/builtin",
            "plugins/custom", 
            "plugins/marketplace",
            os.path.expanduser("~/.web-scraper/plugins")
        ]
        
        # Ensure plugin directories exist
        for directory in self.plugin_directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def discover_plugins(self) -> List[PluginMetadata]:
        """
        Discover all available plugins in the plugin directories.
        
        Returns:
            List of discovered plugin metadata
        """
        discovered = []
        
        for directory in self.plugin_directories:
            if not os.path.exists(directory):
                continue
                
            self.logger.info(f"Discovering plugins in: {directory}")
            
            for item in os.listdir(directory):
                plugin_path = os.path.join(directory, item)
                
                if os.path.isdir(plugin_path):
                    # Check for plugin.yaml or plugin.json
                    metadata = await self._load_plugin_metadata(plugin_path)
                    if metadata:
                        discovered.append(metadata)
                        self.registry.register_metadata(metadata)
                
                elif item.endswith('.py') and not item.startswith('_'):
                    # Single file plugin
                    metadata = await self._load_plugin_from_file(plugin_path)
                    if metadata:
                        discovered.append(metadata)
                        self.registry.register_metadata(metadata)
        
        self.logger.info(f"Discovered {len(discovered)} plugins")
        return discovered
    
    async def load_plugin(self, plugin_name: str, config: Optional[PluginConfig] = None) -> Optional[BasePlugin]:
        """
        Load a specific plugin by name.
        
        Args:
            plugin_name: Name of the plugin to load
            config: Optional plugin configuration
            
        Returns:
            Loaded plugin instance or None if loading failed
        """
        if plugin_name in self.loaded_plugins:
            self.logger.warning(f"Plugin {plugin_name} is already loaded")
            return self.loaded_plugins[plugin_name]
        
        metadata = self.registry.get_plugin_metadata(plugin_name)
        if not metadata:
            self.logger.error(f"Plugin metadata not found: {plugin_name}")
            return None
        
        try:
            # Load the plugin class
            plugin_class = await self._import_plugin_class(metadata)
            if not plugin_class:
                return None
            
            # Create plugin instance
            plugin_config = config or self.plugin_configs.get(plugin_name, PluginConfig())
            plugin_instance = plugin_class(plugin_config)
            
            # Validate configuration
            if not await plugin_instance.validate_config():
                self.logger.error(f"Plugin configuration validation failed: {plugin_name}")
                return None
            
            # Initialize the plugin
            if await plugin_instance.initialize():
                self.loaded_plugins[plugin_name] = plugin_instance
                self.plugin_configs[plugin_name] = plugin_config
                self.logger.info(f"Plugin loaded successfully: {plugin_name}")
                return plugin_instance
            else:
                self.logger.error(f"Plugin initialization failed: {plugin_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_name}: {e}")
            return None
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a specific plugin.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if unloading was successful, False otherwise
        """
        if plugin_name not in self.loaded_plugins:
            self.logger.warning(f"Plugin {plugin_name} is not loaded")
            return False
        
        try:
            plugin = self.loaded_plugins[plugin_name]
            await plugin.cleanup()
            del self.loaded_plugins[plugin_name]
            self.logger.info(f"Plugin unloaded successfully: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    async def load_all_plugins(self) -> Dict[str, BasePlugin]:
        """
        Load all discovered plugins.
        
        Returns:
            Dictionary of loaded plugins
        """
        await self.discover_plugins()
        
        for metadata in self.registry.get_all_metadata():
            if metadata.enabled:
                await self.load_plugin(metadata.name)
        
        return self.loaded_plugins.copy()
    
    async def execute_plugin(self, plugin_name: str, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Optional[PluginResult]:
        """
        Execute a specific plugin.
        
        Args:
            plugin_name: Name of the plugin to execute
            input_data: Input data for the plugin
            context: Optional execution context
            
        Returns:
            Plugin execution result or None if execution failed
        """
        if plugin_name not in self.loaded_plugins:
            self.logger.error(f"Plugin not loaded: {plugin_name}")
            return None
        
        plugin = self.loaded_plugins[plugin_name]
        
        if not plugin.is_enabled():
            self.logger.warning(f"Plugin is disabled: {plugin_name}")
            return None
        
        try:
            return await plugin.execute(input_data, context)
        except Exception as e:
            self.logger.error(f"Error executing plugin {plugin_name}: {e}")
            return PluginResult(success=False, error=str(e))
    
    def get_loaded_plugins(self) -> Dict[str, BasePlugin]:
        """Get all loaded plugins."""
        return self.loaded_plugins.copy()
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all loaded plugins of a specific type."""
        return [
            plugin for plugin in self.loaded_plugins.values()
            if plugin.metadata.plugin_type == plugin_type
        ]
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform health checks on all loaded plugins.
        
        Returns:
            Health status for all plugins
        """
        health_status = {}
        
        for name, plugin in self.loaded_plugins.items():
            try:
                health_status[name] = await plugin.health_check()
            except Exception as e:
                health_status[name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return health_status
    
    async def _load_plugin_metadata(self, plugin_path: str) -> Optional[PluginMetadata]:
        """Load plugin metadata from plugin.yaml or plugin.json."""
        metadata_files = ['plugin.yaml', 'plugin.yml', 'plugin.json']
        
        for filename in metadata_files:
            metadata_file = os.path.join(plugin_path, filename)
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        if filename.endswith('.json'):
                            data = json.load(f)
                        else:
                            data = yaml.safe_load(f)
                    
                    return PluginMetadata(**data)
                    
                except Exception as e:
                    self.logger.error(f"Error loading metadata from {metadata_file}: {e}")
        
        return None
    
    async def _load_plugin_from_file(self, plugin_file: str) -> Optional[PluginMetadata]:
        """Load plugin metadata from a Python file."""
        try:
            spec = importlib.util.spec_from_file_location("temp_plugin", plugin_file)
            if not spec or not spec.loader:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for plugin classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePlugin) and 
                    attr != BasePlugin):
                    
                    # Create temporary instance to get metadata
                    temp_instance = attr()
                    return temp_instance.metadata
            
        except Exception as e:
            self.logger.error(f"Error loading plugin from {plugin_file}: {e}")
        
        return None
    
    async def _import_plugin_class(self, metadata: PluginMetadata) -> Optional[Type[BasePlugin]]:
        """Import the plugin class from its module."""
        # This is a simplified implementation
        # In a real system, you'd need more sophisticated module loading
        try:
            # For now, assume plugins are in the plugins package
            module_name = f"plugins.{metadata.name.lower()}"
            module = importlib.import_module(module_name)
            
            # Look for the plugin class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePlugin) and 
                    attr != BasePlugin):
                    return attr
            
        except ImportError as e:
            self.logger.error(f"Could not import plugin module: {e}")
        
        return None
