"""
Simplified Unified System Initialization
Core system without problematic CLI dependencies.
"""
import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import unified components
from config.unified_config import get_unified_config_manager, UnifiedConfig, ComponentType
from auth.unified_auth import get_unified_auth_manager, UserRole
from data.unified_data_layer import get_unified_data_layer, EntityType
from integration.unified_integration import get_unified_integration_manager, IntegrationEvent


class UnifiedSystemStatus:
    """System status tracking."""
    
    def __init__(self):
        self.started_at = datetime.now()
        self.components_status: Dict[str, Dict[str, Any]] = {}
        self.last_health_check = None
        self.errors: List[str] = []
    
    def update_component_status(self, component: str, status: str, details: Dict[str, Any] = None):
        """Update component status."""
        self.components_status[component] = {
            "status": status,
            "last_updated": datetime.now(),
            "details": details or {}
        }
    
    def add_error(self, error: str):
        """Add system error."""
        self.errors.append(f"{datetime.now().isoformat()}: {error}")
        # Keep only last 100 errors
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]
    
    def get_overall_status(self) -> str:
        """Get overall system status."""
        if self.errors:
            return "error"
        
        statuses = [comp["status"] for comp in self.components_status.values()]
        if all(status == "healthy" for status in statuses):
            return "healthy"
        elif any(status == "error" for status in statuses):
            return "error"
        else:
            return "warning"


class SimpleUnifiedSystem:
    """Simplified unified system class that orchestrates core components."""
    
    def __init__(self):
        """Initialize the unified system."""
        self.logger = self._setup_logging()
        self.status = UnifiedSystemStatus()
        
        # Component managers
        self.config_manager = None
        self.auth_manager = None
        self.data_layer = None
        self.integration_manager = None
        
        # System state
        self.is_initialized = False
        self.is_running = False
        
        self.logger.info("Simple unified system created")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup system logging."""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / "unified_system.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger("unified_system")
    
    async def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            self.logger.info("🚀 Initializing Unified Web Scraper System...")
            
            # Phase 1: Initialize configuration
            await self._initialize_config()
            
            # Phase 2: Initialize authentication
            await self._initialize_auth()
            
            # Phase 3: Initialize data layer
            await self._initialize_data_layer()
            
            # Phase 4: Initialize integration
            await self._initialize_integration()
            
            # Run system health check
            await self._health_check()
            
            self.is_initialized = True
            self.logger.info("✅ Unified system initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize unified system: {e}")
            self.status.add_error(f"Initialization failed: {e}")
            return False
    
    async def _initialize_config(self) -> None:
        """Initialize unified configuration."""
        try:
            self.logger.info("📋 Initializing unified configuration...")
            
            self.config_manager = get_unified_config_manager()
            
            # Validate configuration
            errors = self.config_manager.validate_configuration()
            if errors:
                for error in errors:
                    self.logger.warning(f"Config validation warning: {error}")
            
            self.status.update_component_status("config", "healthy", {
                "config_file": str(self.config_manager.config_file),
                "version": self.config_manager.get_config().version
            })
            
            self.logger.info("✅ Configuration initialized")
            
        except Exception as e:
            self.status.update_component_status("config", "error", {"error": str(e)})
            raise
    
    async def _initialize_auth(self) -> None:
        """Initialize unified authentication."""
        try:
            self.logger.info("🔐 Initializing unified authentication...")
            
            self.auth_manager = get_unified_auth_manager()
            
            # Check if default admin exists
            admin_user = self.auth_manager.get_user_by_username("admin")
            if admin_user:
                self.logger.info("Default admin user found")
            
            self.status.update_component_status("auth", "healthy", {
                "users_count": len(self.auth_manager._users),
                "sessions_count": len(self.auth_manager._sessions)
            })
            
            self.logger.info("✅ Authentication initialized")
            
        except Exception as e:
            self.status.update_component_status("auth", "error", {"error": str(e)})
            raise
    
    async def _initialize_data_layer(self) -> None:
        """Initialize unified data layer."""
        try:
            self.logger.info("💾 Initializing unified data layer...")
            
            self.data_layer = get_unified_data_layer()
            
            # Get data statistics
            stats = self.data_layer.get_statistics()
            
            self.status.update_component_status("data", "healthy", stats)
            
            self.logger.info("✅ Data layer initialized")
            
        except Exception as e:
            self.status.update_component_status("data", "error", {"error": str(e)})
            raise
    
    async def _initialize_integration(self) -> None:
        """Initialize unified integration."""
        try:
            self.logger.info("🔗 Initializing unified integration...")
            
            self.integration_manager = get_unified_integration_manager()
            
            # Get integration statistics
            stats = self.integration_manager.get_integration_stats()
            
            self.status.update_component_status("integration", "healthy", stats)
            
            self.logger.info("✅ Integration initialized")
            
        except Exception as e:
            self.status.update_component_status("integration", "error", {"error": str(e)})
            raise
    
    async def _health_check(self) -> None:
        """Perform system health check."""
        try:
            self.logger.info("🏥 Performing system health check...")
            
            # Check configuration
            config = self.config_manager.get_config()
            if not config:
                raise Exception("Configuration not loaded")
            
            # Check data layer
            data_stats = self.data_layer.get_statistics()
            if data_stats["data_source"] == "sqlite":
                # Test database connection
                test_entity = self.data_layer.create_entity(
                    EntityType.LOG,
                    {"type": "health_check", "timestamp": datetime.now().isoformat()}
                )
                self.data_layer.delete_entity(test_entity.id)
            
            # Check integration
            integration_stats = self.integration_manager.get_integration_stats()
            
            self.status.last_health_check = datetime.now()
            self.logger.info("✅ Health check completed")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.status.add_error(f"Health check failed: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        config = self.config_manager.get_config() if self.config_manager else None
        
        return {
            "system": {
                "version": "2.0.0",
                "started_at": self.status.started_at.isoformat(),
                "uptime_seconds": (datetime.now() - self.status.started_at).total_seconds(),
                "is_initialized": self.is_initialized,
                "is_running": self.is_running,
                "overall_status": self.status.get_overall_status()
            },
            "components": self.status.components_status,
            "configuration": {
                "version": config.version if config else "unknown",
                "config_file": str(self.config_manager.config_file) if self.config_manager else None
            },
            "data": self.data_layer.get_statistics() if self.data_layer else {},
            "integration": self.integration_manager.get_integration_stats() if self.integration_manager else {},
            "errors": self.status.errors[-10:] if self.status.errors else []  # Last 10 errors
        }
    
    async def start(self) -> bool:
        """Start the unified system."""
        try:
            if not self.is_initialized:
                success = await self.initialize()
                if not success:
                    return False
            
            self.is_running = True
            self.logger.info("🎉 Unified Web Scraper System is running!")
            
            # Publish system started event
            if self.integration_manager:
                await self.integration_manager.publish_event(
                    event_type=IntegrationEvent.CONFIG_UPDATED,
                    source_component=ComponentType.API,
                    data={"system_status": "started", "timestamp": datetime.now().isoformat()}
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.status.add_error(f"Start failed: {e}")
            return False


# Global unified system instance
_unified_system: Optional[SimpleUnifiedSystem] = None


def get_unified_system() -> SimpleUnifiedSystem:
    """Get the global unified system instance."""
    global _unified_system
    if _unified_system is None:
        _unified_system = SimpleUnifiedSystem()
    return _unified_system


if __name__ == "__main__":
    # Initialize and start the unified system
    async def main():
        system = get_unified_system()
        success = await system.start()
        
        if success:
            print("\n🎉 Unified Web Scraper System initialized successfully!")
            print("\nSystem Information:")
            info = system.get_system_info()
            print(f"  Status: {info['system']['overall_status']}")
            print(f"  Components: {len(info['components'])} initialized")
            print(f"  Configuration: {info['configuration']['version']}")
            print("\n✅ Core system ready for use!")
            print("\nNote: CLI and Web components can be started separately.")
        else:
            print("\n❌ Failed to initialize unified system")
            sys.exit(1)
    
    asyncio.run(main())
