"""
Unified Integration System
Provides seamless integration between CLI and Web interfaces.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum
import json
import uuid

from config.unified_config import get_unified_config_manager, ComponentType
from auth.unified_auth import get_unified_auth_manager, User, Session
from data.unified_data_layer import get_unified_data_layer, EntityType, DataEntity
from models.task import Task, TaskStatus, TaskType
from models.message import Message


class IntegrationEvent(str, Enum):
    """Integration event types."""
    CLI_COMMAND_EXECUTED = "cli_command_executed"
    WEB_ACTION_TRIGGERED = "web_action_triggered"
    JOB_CREATED = "job_created"
    JOB_UPDATED = "job_updated"
    JOB_COMPLETED = "job_completed"
    AGENT_STATUS_CHANGED = "agent_status_changed"
    CONFIG_UPDATED = "config_updated"
    USER_AUTHENTICATED = "user_authenticated"
    SESSION_CREATED = "session_created"


class IntegrationMessage(BaseModel):
    """Integration message between components."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: IntegrationEvent
    source_component: ComponentType
    target_component: Optional[ComponentType] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntegrationHandler:
    """Base integration handler."""
    
    def __init__(self, event_type: IntegrationEvent):
        self.event_type = event_type
        self.logger = logging.getLogger(f"integration.{event_type.value}")
    
    async def handle(self, message: IntegrationMessage) -> Optional[Dict[str, Any]]:
        """Handle an integration message."""
        raise NotImplementedError


class CLIWebIntegrationHandler(IntegrationHandler):
    """Handler for CLI-Web integration."""
    
    def __init__(self):
        super().__init__(IntegrationEvent.CLI_COMMAND_EXECUTED)
        self.data_layer = get_unified_data_layer()
    
    async def handle(self, message: IntegrationMessage) -> Optional[Dict[str, Any]]:
        """Handle CLI command execution for web interface."""
        try:
            command_data = message.data
            
            # Store CLI command execution in data layer
            entity = self.data_layer.create_entity(
                entity_type=EntityType.LOG,
                data={
                    "type": "cli_command",
                    "command": command_data.get("command"),
                    "args": command_data.get("args", []),
                    "result": command_data.get("result"),
                    "status": command_data.get("status", "completed"),
                    "execution_time": command_data.get("execution_time"),
                    "user_id": message.user_id
                },
                metadata={
                    "source": "cli",
                    "session_id": message.session_id
                }
            )
            
            # Notify web interface if needed
            if command_data.get("notify_web", False):
                await self._notify_web_interface(message, entity)
            
            return {"entity_id": entity.id, "status": "processed"}
            
        except Exception as e:
            self.logger.error(f"Failed to handle CLI command: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _notify_web_interface(self, message: IntegrationMessage, entity: DataEntity) -> None:
        """Notify web interface of CLI command execution."""
        # This would integrate with WebSocket notifications
        notification = {
            "type": "cli_command_executed",
            "command": entity.data.get("command"),
            "status": entity.data.get("status"),
            "timestamp": entity.created_at.isoformat(),
            "user_id": message.user_id
        }
        
        # Store notification for web interface
        self.data_layer.create_entity(
            entity_type=EntityType.LOG,
            data={
                "type": "web_notification",
                "notification": notification
            }
        )


class JobIntegrationHandler(IntegrationHandler):
    """Handler for job lifecycle integration."""
    
    def __init__(self):
        super().__init__(IntegrationEvent.JOB_CREATED)
        self.data_layer = get_unified_data_layer()
    
    async def handle(self, message: IntegrationMessage) -> Optional[Dict[str, Any]]:
        """Handle job lifecycle events."""
        try:
            job_data = message.data
            
            # Create or update job entity
            if message.event_type == IntegrationEvent.JOB_CREATED:
                entity = self.data_layer.create_entity(
                    entity_type=EntityType.JOB,
                    data=job_data,
                    entity_id=job_data.get("id"),
                    metadata={
                        "created_by": message.user_id,
                        "source_component": message.source_component.value
                    }
                )
            else:
                # Update existing job
                job_id = job_data.get("id")
                entity = self.data_layer.update_entity(
                    entity_id=job_id,
                    data=job_data,
                    metadata={
                        "updated_by": message.user_id,
                        "last_update_source": message.source_component.value
                    }
                )
            
            # Notify other components
            await self._notify_components(message, entity)
            
            return {"entity_id": entity.id if entity else None, "status": "processed"}
            
        except Exception as e:
            self.logger.error(f"Failed to handle job event: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _notify_components(self, message: IntegrationMessage, entity: Optional[DataEntity]) -> None:
        """Notify other components of job changes."""
        if not entity:
            return
        
        # Create notification for real-time updates
        notification = {
            "type": f"job_{message.event_type.value}",
            "job_id": entity.id,
            "job_name": entity.data.get("name"),
            "status": entity.data.get("status"),
            "progress": entity.data.get("progress", 0),
            "timestamp": entity.updated_at.isoformat()
        }
        
        # Store for web interface
        self.data_layer.create_entity(
            entity_type=EntityType.LOG,
            data={
                "type": "job_notification",
                "notification": notification
            }
        )


class ConfigIntegrationHandler(IntegrationHandler):
    """Handler for configuration changes."""
    
    def __init__(self):
        super().__init__(IntegrationEvent.CONFIG_UPDATED)
        self.config_manager = get_unified_config_manager()
        self.data_layer = get_unified_data_layer()
    
    async def handle(self, message: IntegrationMessage) -> Optional[Dict[str, Any]]:
        """Handle configuration updates."""
        try:
            config_data = message.data
            
            # Log configuration change
            entity = self.data_layer.create_entity(
                entity_type=EntityType.CONFIG,
                data={
                    "component": config_data.get("component"),
                    "changes": config_data.get("changes", {}),
                    "previous_values": config_data.get("previous_values", {}),
                    "new_values": config_data.get("new_values", {}),
                    "changed_by": message.user_id
                },
                metadata={
                    "source": message.source_component.value,
                    "session_id": message.session_id
                }
            )
            
            # Reload configuration if needed
            if config_data.get("reload_required", False):
                self.config_manager.reload_configuration()
            
            # Notify components of config change
            await self._notify_config_change(message, entity)
            
            return {"entity_id": entity.id, "status": "processed"}
            
        except Exception as e:
            self.logger.error(f"Failed to handle config update: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _notify_config_change(self, message: IntegrationMessage, entity: DataEntity) -> None:
        """Notify components of configuration changes."""
        notification = {
            "type": "config_updated",
            "component": entity.data.get("component"),
            "changes": entity.data.get("changes", {}),
            "timestamp": entity.created_at.isoformat(),
            "requires_restart": entity.data.get("requires_restart", False)
        }
        
        # Store notification
        self.data_layer.create_entity(
            entity_type=EntityType.LOG,
            data={
                "type": "config_notification",
                "notification": notification
            }
        )


class UnifiedIntegrationManager:
    """Unified integration manager for all components."""
    
    def __init__(self):
        """Initialize the unified integration manager."""
        self.logger = logging.getLogger("unified_integration")
        
        # Component managers
        self.config_manager = get_unified_config_manager()
        self.auth_manager = get_unified_auth_manager()
        self.data_layer = get_unified_data_layer()
        
        # Event handlers
        self._handlers: Dict[IntegrationEvent, List[IntegrationHandler]] = {}
        self._subscribers: Dict[IntegrationEvent, List[Callable]] = {}
        
        # Message queue for async processing
        self._message_queue: List[IntegrationMessage] = []
        self._processing = False
        
        # Initialize default handlers
        self._register_default_handlers()
        
        self.logger.info("Unified integration manager initialized")
    
    def _register_default_handlers(self) -> None:
        """Register default integration handlers."""
        # CLI-Web integration
        self.register_handler(IntegrationEvent.CLI_COMMAND_EXECUTED, CLIWebIntegrationHandler())
        
        # Job lifecycle integration
        job_handler = JobIntegrationHandler()
        self.register_handler(IntegrationEvent.JOB_CREATED, job_handler)
        self.register_handler(IntegrationEvent.JOB_UPDATED, job_handler)
        self.register_handler(IntegrationEvent.JOB_COMPLETED, job_handler)
        
        # Configuration integration
        self.register_handler(IntegrationEvent.CONFIG_UPDATED, ConfigIntegrationHandler())
    
    def register_handler(self, event_type: IntegrationEvent, handler: IntegrationHandler) -> None:
        """Register an integration handler."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        
        self.logger.debug(f"Registered handler for {event_type.value}")
    
    def subscribe(self, event_type: IntegrationEvent, callback: Callable) -> None:
        """Subscribe to integration events."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        
        self.logger.debug(f"Subscribed to {event_type.value}")
    
    async def publish_event(self, event_type: IntegrationEvent, source_component: ComponentType,
                           data: Dict[str, Any], user_id: Optional[str] = None,
                           session_id: Optional[str] = None, target_component: Optional[ComponentType] = None) -> str:
        """Publish an integration event."""
        message = IntegrationMessage(
            event_type=event_type,
            source_component=source_component,
            target_component=target_component,
            user_id=user_id,
            session_id=session_id,
            data=data
        )
        
        # Add to queue for processing
        self._message_queue.append(message)
        
        # Process queue if not already processing
        if not self._processing:
            await self._process_message_queue()
        
        self.logger.debug(f"Published event: {event_type.value} from {source_component.value}")
        return message.id

    async def _process_message_queue(self) -> None:
        """Process the message queue."""
        self._processing = True

        try:
            while self._message_queue:
                message = self._message_queue.pop(0)
                await self._process_message(message)

        finally:
            self._processing = False

    async def _process_message(self, message: IntegrationMessage) -> None:
        """Process a single integration message."""
        try:
            # Handle with registered handlers
            handlers = self._handlers.get(message.event_type, [])
            for handler in handlers:
                try:
                    result = await handler.handle(message)
                    if result:
                        self.logger.debug(f"Handler result: {result}")
                except Exception as e:
                    self.logger.error(f"Handler error: {e}")

            # Notify subscribers
            subscribers = self._subscribers.get(message.event_type, [])
            for callback in subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    self.logger.error(f"Subscriber error: {e}")

            # Store message for audit trail
            self.data_layer.create_entity(
                entity_type=EntityType.LOG,
                data={
                    "type": "integration_event",
                    "event_type": message.event_type.value,
                    "source_component": message.source_component.value,
                    "target_component": message.target_component.value if message.target_component else None,
                    "data": message.data,
                    "user_id": message.user_id,
                    "session_id": message.session_id
                },
                metadata=message.metadata
            )

        except Exception as e:
            self.logger.error(f"Failed to process message: {e}")

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            "handlers_registered": {
                event.value: len(handlers)
                for event, handlers in self._handlers.items()
            },
            "subscribers_count": {
                event.value: len(subscribers)
                for event, subscribers in self._subscribers.items()
            },
            "queue_size": len(self._message_queue),
            "processing": self._processing
        }

    async def trigger_cli_command(self, command: str, args: List[str], user_id: Optional[str] = None,
                                 session_id: Optional[str] = None) -> str:
        """Trigger a CLI command from web interface."""
        return await self.publish_event(
            event_type=IntegrationEvent.CLI_COMMAND_EXECUTED,
            source_component=ComponentType.WEB,
            target_component=ComponentType.CLI,
            data={
                "command": command,
                "args": args,
                "triggered_from": "web",
                "notify_web": True
            },
            user_id=user_id,
            session_id=session_id
        )

    async def trigger_web_action(self, action: str, data: Dict[str, Any], user_id: Optional[str] = None,
                                session_id: Optional[str] = None) -> str:
        """Trigger a web action from CLI."""
        return await self.publish_event(
            event_type=IntegrationEvent.WEB_ACTION_TRIGGERED,
            source_component=ComponentType.CLI,
            target_component=ComponentType.WEB,
            data={
                "action": action,
                "data": data,
                "triggered_from": "cli"
            },
            user_id=user_id,
            session_id=session_id
        )

    async def sync_job_status(self, job_id: str, status: str, progress: int = 0,
                             user_id: Optional[str] = None) -> str:
        """Sync job status across components."""
        return await self.publish_event(
            event_type=IntegrationEvent.JOB_UPDATED,
            source_component=ComponentType.API,
            data={
                "id": job_id,
                "status": status,
                "progress": progress,
                "updated_at": datetime.now().isoformat()
            },
            user_id=user_id
        )

    async def sync_config_change(self, component: str, changes: Dict[str, Any],
                                user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Sync configuration changes across components."""
        return await self.publish_event(
            event_type=IntegrationEvent.CONFIG_UPDATED,
            source_component=ComponentType.API,
            data={
                "component": component,
                "changes": changes,
                "reload_required": True
            },
            user_id=user_id,
            session_id=session_id
        )


# Global unified integration manager instance
_unified_integration_manager: Optional[UnifiedIntegrationManager] = None


def get_unified_integration_manager() -> UnifiedIntegrationManager:
    """Get the global unified integration manager instance."""
    global _unified_integration_manager
    if _unified_integration_manager is None:
        _unified_integration_manager = UnifiedIntegrationManager()
    return _unified_integration_manager


# Convenience functions for common integration tasks
async def notify_cli_command_executed(command: str, result: Any, user_id: Optional[str] = None,
                                     session_id: Optional[str] = None) -> str:
    """Notify that a CLI command was executed."""
    manager = get_unified_integration_manager()
    return await manager.publish_event(
        event_type=IntegrationEvent.CLI_COMMAND_EXECUTED,
        source_component=ComponentType.CLI,
        data={
            "command": command,
            "result": result,
            "status": "completed",
            "execution_time": datetime.now().isoformat()
        },
        user_id=user_id,
        session_id=session_id
    )


async def notify_job_created(job_data: Dict[str, Any], user_id: Optional[str] = None) -> str:
    """Notify that a job was created."""
    manager = get_unified_integration_manager()
    return await manager.publish_event(
        event_type=IntegrationEvent.JOB_CREATED,
        source_component=ComponentType.API,
        data=job_data,
        user_id=user_id
    )


async def notify_agent_status_changed(agent_id: str, status: str, details: Dict[str, Any]) -> str:
    """Notify that an agent status changed."""
    manager = get_unified_integration_manager()
    return await manager.publish_event(
        event_type=IntegrationEvent.AGENT_STATUS_CHANGED,
        source_component=ComponentType.AGENT,
        data={
            "agent_id": agent_id,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
    )
