"""
API routes for plugin management and marketplace.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel, Field
from datetime import datetime

from plugins.plugin_manager import PluginManager
from plugins.base_plugin import PluginType, PluginConfig, PluginMetadata
from web.api.dependencies import get_agent_manager


logger = logging.getLogger("plugins_api")
router = APIRouter()

# Global plugin manager instance
plugin_manager = PluginManager()


class PluginInstallRequest(BaseModel):
    """Request to install a plugin."""
    plugin_name: str
    source: str = "marketplace"  # "marketplace", "file", "git"
    config: Optional[Dict[str, Any]] = None


class PluginConfigRequest(BaseModel):
    """Request to configure a plugin."""
    plugin_name: str
    config: Dict[str, Any]


class PluginExecutionRequest(BaseModel):
    """Request to execute a plugin."""
    plugin_name: str
    input_data: Any
    context: Optional[Dict[str, Any]] = None


@router.get("/")
async def list_plugins():
    """List all available plugins."""
    try:
        # Discover plugins
        await plugin_manager.discover_plugins()
        
        # Get plugin metadata
        plugins = []
        for metadata in plugin_manager.registry.get_all_metadata():
            plugin_info = {
                "name": metadata.name,
                "version": metadata.version,
                "description": metadata.description,
                "author": metadata.author,
                "type": metadata.plugin_type.value,
                "enabled": metadata.enabled,
                "tags": metadata.tags,
                "dependencies": metadata.dependencies,
                "loaded": metadata.name in plugin_manager.loaded_plugins
            }
            plugins.append(plugin_info)
        
        return {"plugins": plugins}
        
    except Exception as e:
        logger.error(f"Failed to list plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def get_plugin_types():
    """Get available plugin types."""
    types = [
        {
            "type": "agent",
            "name": "Agent Plugin",
            "description": "Custom agent implementations",
            "icon": "🤖"
        },
        {
            "type": "processor",
            "name": "Data Processor",
            "description": "Custom data processing logic",
            "icon": "⚙️"
        },
        {
            "type": "extractor",
            "name": "Data Extractor",
            "description": "Custom data extraction methods",
            "icon": "🔍"
        },
        {
            "type": "transformer",
            "name": "Data Transformer",
            "description": "Custom data transformation logic",
            "icon": "🔄"
        },
        {
            "type": "validator",
            "name": "Data Validator",
            "description": "Custom validation rules",
            "icon": "✅"
        },
        {
            "type": "exporter",
            "name": "Data Exporter",
            "description": "Custom export formats",
            "icon": "📤"
        },
        {
            "type": "middleware",
            "name": "Middleware",
            "description": "Request/response middleware",
            "icon": "🔗"
        },
        {
            "type": "integration",
            "name": "Integration",
            "description": "Third-party service integrations",
            "icon": "🔌"
        }
    ]
    
    return {"types": types}


@router.get("/marketplace")
async def get_marketplace_plugins():
    """Get plugins available in the marketplace."""
    # Simulated marketplace data
    marketplace_plugins = [
        {
            "name": "advanced-pdf-extractor",
            "version": "1.2.0",
            "description": "Advanced PDF text and table extraction with OCR support",
            "author": "WebScraper Team",
            "type": "extractor",
            "tags": ["pdf", "ocr", "tables"],
            "downloads": 1247,
            "rating": 4.8,
            "price": "free",
            "featured": True,
            "screenshots": ["/static/images/pdf-extractor-1.png"],
            "documentation": "https://docs.webscraper.com/plugins/pdf-extractor"
        },
        {
            "name": "social-media-scraper",
            "version": "2.1.3",
            "description": "Scrape content from major social media platforms",
            "author": "Community",
            "type": "agent",
            "tags": ["social", "twitter", "facebook", "instagram"],
            "downloads": 892,
            "rating": 4.5,
            "price": "$9.99",
            "featured": True,
            "screenshots": ["/static/images/social-scraper-1.png"],
            "documentation": "https://docs.webscraper.com/plugins/social-scraper"
        },
        {
            "name": "ai-content-classifier",
            "version": "1.0.5",
            "description": "AI-powered content classification and sentiment analysis",
            "author": "AI Labs",
            "type": "processor",
            "tags": ["ai", "nlp", "classification", "sentiment"],
            "downloads": 634,
            "rating": 4.7,
            "price": "$19.99",
            "featured": False,
            "screenshots": ["/static/images/ai-classifier-1.png"],
            "documentation": "https://docs.webscraper.com/plugins/ai-classifier"
        },
        {
            "name": "excel-advanced-exporter",
            "version": "1.1.2",
            "description": "Export data to Excel with advanced formatting and charts",
            "author": "Office Tools Inc",
            "type": "exporter",
            "tags": ["excel", "charts", "formatting"],
            "downloads": 445,
            "rating": 4.3,
            "price": "$14.99",
            "featured": False,
            "screenshots": ["/static/images/excel-exporter-1.png"],
            "documentation": "https://docs.webscraper.com/plugins/excel-exporter"
        },
        {
            "name": "proxy-rotator-pro",
            "version": "3.0.1",
            "description": "Professional proxy rotation with geo-targeting",
            "author": "Proxy Solutions",
            "type": "middleware",
            "tags": ["proxy", "rotation", "geo", "stealth"],
            "downloads": 1156,
            "rating": 4.9,
            "price": "$29.99",
            "featured": True,
            "screenshots": ["/static/images/proxy-rotator-1.png"],
            "documentation": "https://docs.webscraper.com/plugins/proxy-rotator"
        }
    ]
    
    return {"plugins": marketplace_plugins}


@router.post("/install")
async def install_plugin(request: PluginInstallRequest):
    """Install a plugin."""
    try:
        logger.info(f"Installing plugin: {request.plugin_name}")
        
        if request.source == "marketplace":
            # Simulate marketplace installation
            await asyncio.sleep(2)  # Simulate download time
            
            # Create plugin config
            config = PluginConfig(**(request.config or {}))
            
            # Load the plugin (simulated)
            plugin = await plugin_manager.load_plugin(request.plugin_name, config)
            
            if plugin:
                return {
                    "status": "success",
                    "message": f"Plugin {request.plugin_name} installed successfully",
                    "plugin": {
                        "name": plugin.metadata.name,
                        "version": plugin.metadata.version,
                        "type": plugin.metadata.plugin_type.value
                    }
                }
            else:
                raise HTTPException(status_code=400, detail="Failed to load plugin after installation")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported installation source: {request.source}")
            
    except Exception as e:
        logger.error(f"Failed to install plugin {request.plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{plugin_name}")
async def uninstall_plugin(plugin_name: str):
    """Uninstall a plugin."""
    try:
        success = await plugin_manager.unload_plugin(plugin_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Plugin {plugin_name} uninstalled successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Plugin not found or not loaded")
            
    except Exception as e:
        logger.error(f"Failed to uninstall plugin {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{plugin_name}")
async def get_plugin_details(plugin_name: str):
    """Get detailed information about a plugin."""
    try:
        # Check if plugin is loaded
        loaded_plugin = plugin_manager.loaded_plugins.get(plugin_name)
        
        if loaded_plugin:
            health_status = await loaded_plugin.health_check()
            
            return {
                "name": loaded_plugin.metadata.name,
                "version": loaded_plugin.metadata.version,
                "description": loaded_plugin.metadata.description,
                "author": loaded_plugin.metadata.author,
                "type": loaded_plugin.metadata.plugin_type.value,
                "enabled": loaded_plugin.is_enabled(),
                "tags": loaded_plugin.metadata.tags,
                "dependencies": loaded_plugin.metadata.dependencies,
                "config": loaded_plugin.config.dict(),
                "health": health_status,
                "loaded": True
            }
        
        # Check registry for metadata
        metadata = plugin_manager.registry.get_plugin_metadata(plugin_name)
        if metadata:
            return {
                "name": metadata.name,
                "version": metadata.version,
                "description": metadata.description,
                "author": metadata.author,
                "type": metadata.plugin_type.value,
                "enabled": metadata.enabled,
                "tags": metadata.tags,
                "dependencies": metadata.dependencies,
                "loaded": False
            }
        
        raise HTTPException(status_code=404, detail="Plugin not found")
        
    except Exception as e:
        logger.error(f"Failed to get plugin details for {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{plugin_name}/configure")
async def configure_plugin(plugin_name: str, request: PluginConfigRequest):
    """Configure a plugin."""
    try:
        plugin = plugin_manager.loaded_plugins.get(plugin_name)
        
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not loaded")
        
        # Update plugin configuration
        for key, value in request.config.items():
            plugin.set_config_value(key, value)
        
        # Validate new configuration
        if await plugin.validate_config():
            return {
                "status": "success",
                "message": f"Plugin {plugin_name} configured successfully",
                "config": plugin.config.dict()
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid configuration")
            
    except Exception as e:
        logger.error(f"Failed to configure plugin {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{plugin_name}/execute")
async def execute_plugin(plugin_name: str, request: PluginExecutionRequest):
    """Execute a plugin."""
    try:
        result = await plugin_manager.execute_plugin(
            plugin_name,
            request.input_data,
            request.context
        )
        
        if result:
            return {
                "status": "success",
                "result": result.dict()
            }
        else:
            raise HTTPException(status_code=400, detail="Plugin execution failed")
            
    except Exception as e:
        logger.error(f"Failed to execute plugin {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{plugin_name}/health")
async def check_plugin_health(plugin_name: str):
    """Check plugin health status."""
    try:
        plugin = plugin_manager.loaded_plugins.get(plugin_name)
        
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not loaded")
        
        health_status = await plugin.health_check()
        return health_status
        
    except Exception as e:
        logger.error(f"Failed to check plugin health for {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_plugin(file: UploadFile = File(...)):
    """Upload a custom plugin file."""
    try:
        if not file.filename.endswith(('.py', '.zip')):
            raise HTTPException(status_code=400, detail="Only .py and .zip files are supported")
        
        # Save uploaded file
        upload_path = f"plugins/custom/{file.filename}"
        
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Try to discover and load the plugin
        await plugin_manager.discover_plugins()
        
        return {
            "status": "success",
            "message": f"Plugin file {file.filename} uploaded successfully",
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Failed to upload plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def check_all_plugins_health():
    """Check health status of all loaded plugins."""
    try:
        health_status = await plugin_manager.health_check_all()
        return {"health_status": health_status}
        
    except Exception as e:
        logger.error(f"Failed to check plugins health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload")
async def reload_plugins():
    """Reload all plugins."""
    try:
        # Unload all plugins
        for plugin_name in list(plugin_manager.loaded_plugins.keys()):
            await plugin_manager.unload_plugin(plugin_name)
        
        # Discover and load plugins
        await plugin_manager.discover_plugins()
        loaded_plugins = await plugin_manager.load_all_plugins()
        
        return {
            "status": "success",
            "message": f"Reloaded {len(loaded_plugins)} plugins",
            "loaded_plugins": list(loaded_plugins.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to reload plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))
