"""
Plugin System for Web Scraper
Extensible architecture for custom agents and processors.
"""

from .base_plugin import BasePlugin, PluginType, PluginMetadata
from .plugin_manager import PluginManager
from .plugin_registry import PluginRegistry

__all__ = [
    'BasePlugin',
    'PluginType', 
    'PluginMetadata',
    'PluginManager',
    'PluginRegistry'
]

__version__ = "1.0.0"
