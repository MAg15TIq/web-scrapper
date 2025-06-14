"""
Plugin Registry for Web Scraper Plugin System
Maintains a registry of available plugins and their metadata.
"""

import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict

from .base_plugin import PluginMetadata, PluginType


class PluginRegistry:
    """
    Registry for managing plugin metadata and discovery.
    
    Provides a centralized location for plugin information and
    supports querying plugins by various criteria.
    """
    
    def __init__(self):
        """Initialize the plugin registry."""
        self.logger = logging.getLogger("plugin_registry")
        self._plugins: Dict[str, PluginMetadata] = {}
        self._plugins_by_type: Dict[PluginType, List[str]] = defaultdict(list)
        self._plugins_by_author: Dict[str, List[str]] = defaultdict(list)
        self._plugins_by_tag: Dict[str, List[str]] = defaultdict(list)
    
    def register_metadata(self, metadata: PluginMetadata) -> bool:
        """
        Register plugin metadata in the registry.
        
        Args:
            metadata: Plugin metadata to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            plugin_name = metadata.name
            
            # Check if plugin already exists
            if plugin_name in self._plugins:
                existing = self._plugins[plugin_name]
                if existing.version == metadata.version:
                    self.logger.warning(f"Plugin {plugin_name} v{metadata.version} already registered")
                    return False
                else:
                    self.logger.info(f"Updating plugin {plugin_name} from v{existing.version} to v{metadata.version}")
                    self._unregister_metadata(plugin_name)
            
            # Register the plugin
            self._plugins[plugin_name] = metadata
            
            # Index by type
            self._plugins_by_type[metadata.plugin_type].append(plugin_name)
            
            # Index by author
            self._plugins_by_author[metadata.author].append(plugin_name)
            
            # Index by tags
            for tag in metadata.tags:
                self._plugins_by_tag[tag].append(plugin_name)
            
            self.logger.info(f"Plugin registered: {plugin_name} v{metadata.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering plugin metadata: {e}")
            return False
    
    def unregister_metadata(self, plugin_name: str) -> bool:
        """
        Unregister plugin metadata from the registry.
        
        Args:
            plugin_name: Name of the plugin to unregister
            
        Returns:
            True if unregistration was successful, False otherwise
        """
        return self._unregister_metadata(plugin_name)
    
    def _unregister_metadata(self, plugin_name: str) -> bool:
        """Internal method to unregister plugin metadata."""
        if plugin_name not in self._plugins:
            self.logger.warning(f"Plugin not found in registry: {plugin_name}")
            return False
        
        try:
            metadata = self._plugins[plugin_name]
            
            # Remove from main registry
            del self._plugins[plugin_name]
            
            # Remove from type index
            if plugin_name in self._plugins_by_type[metadata.plugin_type]:
                self._plugins_by_type[metadata.plugin_type].remove(plugin_name)
            
            # Remove from author index
            if plugin_name in self._plugins_by_author[metadata.author]:
                self._plugins_by_author[metadata.author].remove(plugin_name)
            
            # Remove from tag indices
            for tag in metadata.tags:
                if plugin_name in self._plugins_by_tag[tag]:
                    self._plugins_by_tag[tag].remove(plugin_name)
            
            self.logger.info(f"Plugin unregistered: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unregistering plugin {plugin_name}: {e}")
            return False
    
    def get_plugin_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """
        Get metadata for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin metadata or None if not found
        """
        return self._plugins.get(plugin_name)
    
    def get_all_metadata(self) -> List[PluginMetadata]:
        """
        Get metadata for all registered plugins.
        
        Returns:
            List of all plugin metadata
        """
        return list(self._plugins.values())
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginMetadata]:
        """
        Get all plugins of a specific type.
        
        Args:
            plugin_type: Type of plugins to retrieve
            
        Returns:
            List of plugin metadata for the specified type
        """
        plugin_names = self._plugins_by_type.get(plugin_type, [])
        return [self._plugins[name] for name in plugin_names if name in self._plugins]
    
    def get_plugins_by_author(self, author: str) -> List[PluginMetadata]:
        """
        Get all plugins by a specific author.
        
        Args:
            author: Author name
            
        Returns:
            List of plugin metadata for the specified author
        """
        plugin_names = self._plugins_by_author.get(author, [])
        return [self._plugins[name] for name in plugin_names if name in self._plugins]
    
    def get_plugins_by_tag(self, tag: str) -> List[PluginMetadata]:
        """
        Get all plugins with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of plugin metadata with the specified tag
        """
        plugin_names = self._plugins_by_tag.get(tag, [])
        return [self._plugins[name] for name in plugin_names if name in self._plugins]
    
    def search_plugins(self, query: str) -> List[PluginMetadata]:
        """
        Search for plugins by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of matching plugin metadata
        """
        query_lower = query.lower()
        matches = []
        
        for metadata in self._plugins.values():
            # Search in name
            if query_lower in metadata.name.lower():
                matches.append(metadata)
                continue
            
            # Search in description
            if query_lower in metadata.description.lower():
                matches.append(metadata)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in metadata.tags):
                matches.append(metadata)
                continue
        
        return matches
    
    def get_enabled_plugins(self) -> List[PluginMetadata]:
        """
        Get all enabled plugins.
        
        Returns:
            List of enabled plugin metadata
        """
        return [metadata for metadata in self._plugins.values() if metadata.enabled]
    
    def get_plugin_count(self) -> int:
        """
        Get the total number of registered plugins.
        
        Returns:
            Number of registered plugins
        """
        return len(self._plugins)
    
    def get_plugin_count_by_type(self) -> Dict[PluginType, int]:
        """
        Get plugin count by type.
        
        Returns:
            Dictionary mapping plugin types to counts
        """
        return {
            plugin_type: len(plugin_names)
            for plugin_type, plugin_names in self._plugins_by_type.items()
        }
    
    def list_plugin_names(self) -> List[str]:
        """
        Get a list of all registered plugin names.
        
        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())
    
    def list_authors(self) -> List[str]:
        """
        Get a list of all plugin authors.
        
        Returns:
            List of unique author names
        """
        return list(self._plugins_by_author.keys())
    
    def list_tags(self) -> List[str]:
        """
        Get a list of all plugin tags.
        
        Returns:
            List of unique tags
        """
        return list(self._plugins_by_tag.keys())
    
    def validate_dependencies(self, plugin_name: str) -> Dict[str, bool]:
        """
        Validate that all dependencies for a plugin are available.
        
        Args:
            plugin_name: Name of the plugin to validate
            
        Returns:
            Dictionary mapping dependency names to availability status
        """
        metadata = self.get_plugin_metadata(plugin_name)
        if not metadata:
            return {}
        
        dependency_status = {}
        for dependency in metadata.dependencies:
            dependency_status[dependency] = dependency in self._plugins
        
        return dependency_status
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get the dependency graph for all plugins.
        
        Returns:
            Dictionary mapping plugin names to their dependencies
        """
        dependency_graph = {}
        for name, metadata in self._plugins.items():
            dependency_graph[name] = metadata.dependencies.copy()
        
        return dependency_graph
    
    def clear(self) -> None:
        """Clear all registered plugins from the registry."""
        self._plugins.clear()
        self._plugins_by_type.clear()
        self._plugins_by_author.clear()
        self._plugins_by_tag.clear()
        self.logger.info("Plugin registry cleared")
