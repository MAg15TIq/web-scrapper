"""
Event-Driven Architecture System for Phase 2 Enhancement

This module provides webhook integration, schedule-based scraping,
event queuing with priority handling, and reactive scraping.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import croniter
import httpx


# Configure logging
logger = logging.getLogger("event_system")


class EventType(Enum):
    """Types of events in the system."""
    SCRAPING_REQUESTED = "scraping_requested"
    SCRAPING_COMPLETED = "scraping_completed"
    SCRAPING_FAILED = "scraping_failed"
    CHANGE_DETECTED = "change_detected"
    SCHEDULE_TRIGGERED = "schedule_triggered"
    WEBHOOK_RECEIVED = "webhook_received"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    SYSTEM_ALERT = "system_alert"
    USER_ACTION = "user_action"
    CUSTOM_EVENT = "custom_event"


class EventPriority(Enum):
    """Event priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class EventStatus(Enum):
    """Event processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Event:
    """Represents an event in the system."""
    event_id: str
    event_type: EventType
    priority: EventPriority
    status: EventStatus
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3
    scheduled_for: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class EventHandler:
    """Configuration for an event handler."""
    handler_id: str
    event_types: List[EventType]
    handler_function: Callable
    priority_filter: Optional[List[EventPriority]] = None
    source_filter: Optional[List[str]] = None
    enabled: bool = True
    async_handler: bool = True


@dataclass
class WebhookConfig:
    """Configuration for webhook endpoints."""
    webhook_id: str
    url: str
    secret: Optional[str]
    events: List[EventType]
    headers: Dict[str, str]
    timeout: int = 30
    retry_attempts: int = 3
    enabled: bool = True


@dataclass
class ScheduleConfig:
    """Configuration for scheduled events."""
    schedule_id: str
    name: str
    cron_expression: str
    event_type: EventType
    event_data: Dict[str, Any]
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


class EventDrivenSystem:
    """
    Enhanced Event-Driven Architecture System for Phase 2.
    
    Features:
    - Webhook integration for external triggers
    - Schedule-based scraping with cron-like functionality
    - Event queuing system with priority handling
    - Reactive scraping based on external data sources
    - Event handlers with filtering and routing
    """
    
    def __init__(self):
        self.logger = logging.getLogger("event_system")
        
        # Event storage and queues
        self.event_queues: Dict[EventPriority, deque] = {
            priority: deque() for priority in EventPriority
        }
        self.event_history: deque = deque(maxlen=10000)
        self.active_events: Dict[str, Event] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, EventHandler] = {}
        self.event_type_handlers: Dict[EventType, List[str]] = defaultdict(list)
        
        # Webhooks and schedules
        self.webhook_configs: Dict[str, WebhookConfig] = {}
        self.schedule_configs: Dict[str, ScheduleConfig] = {}
        
        # HTTP client for webhook calls
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
        
        # Performance metrics
        self.metrics = {
            "total_events": 0,
            "processed_events": 0,
            "failed_events": 0,
            "webhook_calls": 0,
            "scheduled_events": 0,
            "average_processing_time": 0.0
        }
        
        self.logger.info("Event-driven system initialized")
    
    async def start(self):
        """Start the event-driven system."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.background_tasks.add(asyncio.create_task(self._event_processor()))
        self.background_tasks.add(asyncio.create_task(self._schedule_processor()))
        self.background_tasks.add(asyncio.create_task(self._metrics_collector()))
        self.background_tasks.add(asyncio.create_task(self._cleanup_old_events()))
        
        self.logger.info("Event-driven system started")
    
    async def stop(self):
        """Stop the event-driven system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        # Close HTTP client
        await self.http_client.aclose()
        
        self.logger.info("Event-driven system stopped")
    
    async def emit_event(self, event_type: EventType, data: Dict[str, Any],
                        priority: EventPriority = EventPriority.NORMAL,
                        source: str = "system",
                        metadata: Optional[Dict[str, Any]] = None,
                        scheduled_for: Optional[datetime] = None) -> str:
        """Emit a new event."""
        try:
            event = Event(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                priority=priority,
                status=EventStatus.PENDING,
                timestamp=datetime.now(),
                source=source,
                data=data,
                metadata=metadata or {},
                scheduled_for=scheduled_for
            )
            
            # Add to appropriate queue
            if scheduled_for and scheduled_for > datetime.now():
                # Scheduled event - will be processed by schedule processor
                self.active_events[event.event_id] = event
            else:
                # Immediate event - add to priority queue
                self.event_queues[priority].append(event)
                self.active_events[event.event_id] = event
            
            self.metrics["total_events"] += 1
            
            self.logger.debug(f"Event emitted: {event.event_id} ({event_type.value})")
            return event.event_id
            
        except Exception as e:
            self.logger.error(f"Error emitting event: {e}")
            raise
    
    def register_event_handler(self, handler_config: EventHandler) -> str:
        """Register an event handler."""
        try:
            handler_id = handler_config.handler_id
            self.event_handlers[handler_id] = handler_config
            
            # Index by event types for faster lookup
            for event_type in handler_config.event_types:
                self.event_type_handlers[event_type].append(handler_id)
            
            self.logger.info(f"Event handler registered: {handler_id}")
            return handler_id
            
        except Exception as e:
            self.logger.error(f"Error registering event handler: {e}")
            raise
    
    def unregister_event_handler(self, handler_id: str) -> bool:
        """Unregister an event handler."""
        try:
            if handler_id not in self.event_handlers:
                return False
            
            handler = self.event_handlers[handler_id]
            
            # Remove from event type index
            for event_type in handler.event_types:
                if handler_id in self.event_type_handlers[event_type]:
                    self.event_type_handlers[event_type].remove(handler_id)
            
            # Remove handler
            del self.event_handlers[handler_id]
            
            self.logger.info(f"Event handler unregistered: {handler_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unregistering event handler: {e}")
            return False
    
    async def add_webhook_config(self, config: WebhookConfig) -> str:
        """Add a webhook configuration."""
        try:
            self.webhook_configs[config.webhook_id] = config
            
            self.logger.info(f"Webhook configured: {config.webhook_id} -> {config.url}")
            return config.webhook_id
            
        except Exception as e:
            self.logger.error(f"Error adding webhook config: {e}")
            raise
    
    async def add_schedule_config(self, config: ScheduleConfig) -> str:
        """Add a schedule configuration."""
        try:
            # Calculate next run time
            if config.enabled:
                cron = croniter.croniter(config.cron_expression, datetime.now())
                config.next_run = cron.get_next(datetime)
            
            self.schedule_configs[config.schedule_id] = config
            
            self.logger.info(f"Schedule configured: {config.schedule_id} ({config.cron_expression})")
            return config.schedule_id
            
        except Exception as e:
            self.logger.error(f"Error adding schedule config: {e}")
            raise
    
    async def process_webhook(self, webhook_data: Dict[str, Any], 
                             source: str = "webhook") -> str:
        """Process incoming webhook data."""
        try:
            # Emit webhook received event
            event_id = await self.emit_event(
                event_type=EventType.WEBHOOK_RECEIVED,
                data=webhook_data,
                priority=EventPriority.HIGH,
                source=source,
                metadata={"webhook_timestamp": datetime.now().isoformat()}
            )
            
            self.logger.info(f"Webhook processed: {event_id}")
            return event_id
            
        except Exception as e:
            self.logger.error(f"Error processing webhook: {e}")
            raise
    
    async def get_event_status(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an event."""
        try:
            if event_id in self.active_events:
                event = self.active_events[event_id]
                return asdict(event)
            
            # Check event history
            for event_data in self.event_history:
                if event_data.get('event_id') == event_id:
                    return event_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting event status: {e}")
            return None
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics and status."""
        try:
            # Calculate queue sizes
            queue_sizes = {
                priority.value: len(queue) 
                for priority, queue in self.event_queues.items()
            }
            
            # Count active events by status
            status_counts = defaultdict(int)
            for event in self.active_events.values():
                status_counts[event.status.value] += 1
            
            return {
                "is_running": self.is_running,
                "metrics": self.metrics,
                "queue_sizes": queue_sizes,
                "active_events": len(self.active_events),
                "event_status_counts": dict(status_counts),
                "registered_handlers": len(self.event_handlers),
                "webhook_configs": len(self.webhook_configs),
                "schedule_configs": len(self.schedule_configs),
                "background_tasks": len(self.background_tasks)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e)}

    # ===== BACKGROUND TASKS =====

    async def _event_processor(self):
        """Background task to process events from queues."""
        while self.is_running:
            try:
                # Process events by priority
                event_processed = False

                for priority in [EventPriority.CRITICAL, EventPriority.HIGH,
                               EventPriority.NORMAL, EventPriority.LOW]:
                    queue = self.event_queues[priority]

                    if queue:
                        event = queue.popleft()
                        await self._process_event(event)
                        event_processed = True
                        break

                # If no events to process, wait a bit
                if not event_processed:
                    await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in event processor: {e}")
                await asyncio.sleep(1)

    async def _schedule_processor(self):
        """Background task to process scheduled events."""
        while self.is_running:
            try:
                current_time = datetime.now()

                # Check scheduled events
                for config in list(self.schedule_configs.values()):
                    if not config.enabled:
                        continue

                    if config.next_run and current_time >= config.next_run:
                        # Trigger scheduled event
                        await self._trigger_scheduled_event(config)

                # Check for delayed events
                delayed_events = [
                    event for event in self.active_events.values()
                    if (event.scheduled_for and
                        event.status == EventStatus.PENDING and
                        current_time >= event.scheduled_for)
                ]

                for event in delayed_events:
                    # Move to processing queue
                    self.event_queues[event.priority].append(event)
                    event.scheduled_for = None

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Error in schedule processor: {e}")
                await asyncio.sleep(60)

    async def _metrics_collector(self):
        """Background task to collect and update metrics."""
        while self.is_running:
            try:
                # Update metrics
                total_queue_size = sum(len(queue) for queue in self.event_queues.values())

                # Log metrics periodically
                if self.metrics["total_events"] > 0:
                    success_rate = (self.metrics["processed_events"] /
                                  self.metrics["total_events"] * 100)

                    self.logger.info(
                        f"Event System Metrics - Total: {self.metrics['total_events']}, "
                        f"Processed: {self.metrics['processed_events']}, "
                        f"Success Rate: {success_rate:.1f}%, "
                        f"Queue Size: {total_queue_size}"
                    )

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(300)

    async def _cleanup_old_events(self):
        """Background task to clean up old events."""
        while self.is_running:
            try:
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(hours=24)

                # Clean up completed/failed events older than 24 hours
                old_event_ids = [
                    event_id for event_id, event in self.active_events.items()
                    if (event.status in [EventStatus.COMPLETED, EventStatus.FAILED, EventStatus.CANCELLED] and
                        event.timestamp < cutoff_time)
                ]

                for event_id in old_event_ids:
                    event = self.active_events[event_id]

                    # Move to history
                    self.event_history.append(asdict(event))

                    # Remove from active events
                    del self.active_events[event_id]

                if old_event_ids:
                    self.logger.info(f"Cleaned up {len(old_event_ids)} old events")

                await asyncio.sleep(3600)  # Clean up every hour

            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(3600)

    async def _process_event(self, event: Event):
        """Process a single event."""
        try:
            start_time = time.time()
            event.status = EventStatus.PROCESSING

            # Find handlers for this event type
            handler_ids = self.event_type_handlers.get(event.event_type, [])

            if not handler_ids:
                self.logger.warning(f"No handlers found for event type: {event.event_type.value}")
                event.status = EventStatus.COMPLETED
                return

            # Process with each handler
            success_count = 0
            for handler_id in handler_ids:
                if handler_id not in self.event_handlers:
                    continue

                handler = self.event_handlers[handler_id]

                if not handler.enabled:
                    continue

                # Apply filters
                if not self._event_matches_handler(event, handler):
                    continue

                try:
                    # Call handler function
                    if handler.async_handler:
                        await handler.handler_function(event)
                    else:
                        handler.handler_function(event)

                    success_count += 1

                except Exception as e:
                    self.logger.error(f"Error in handler {handler_id}: {e}")

            # Update event status
            if success_count > 0:
                event.status = EventStatus.COMPLETED
                self.metrics["processed_events"] += 1
            else:
                event.status = EventStatus.FAILED
                self.metrics["failed_events"] += 1
                event.error_message = "No handlers processed the event successfully"

            event.processed_at = datetime.now()

            # Update processing time metric
            processing_time = time.time() - start_time
            current_avg = self.metrics["average_processing_time"]
            processed_count = self.metrics["processed_events"]

            if processed_count > 0:
                self.metrics["average_processing_time"] = (
                    (current_avg * (processed_count - 1) + processing_time) / processed_count
                )

            # Send webhooks if configured
            await self._send_event_webhooks(event)

        except Exception as e:
            self.logger.error(f"Error processing event {event.event_id}: {e}")
            event.status = EventStatus.FAILED
            event.error_message = str(e)
            self.metrics["failed_events"] += 1

    def _event_matches_handler(self, event: Event, handler: EventHandler) -> bool:
        """Check if an event matches a handler's filters."""
        try:
            # Check priority filter
            if (handler.priority_filter and
                event.priority not in handler.priority_filter):
                return False

            # Check source filter
            if (handler.source_filter and
                event.source not in handler.source_filter):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking event filters: {e}")
            return False

    async def _trigger_scheduled_event(self, config: ScheduleConfig):
        """Trigger a scheduled event."""
        try:
            # Emit the scheduled event
            event_id = await self.emit_event(
                event_type=config.event_type,
                data=config.event_data,
                priority=EventPriority.NORMAL,
                source=f"schedule:{config.schedule_id}",
                metadata={
                    "schedule_id": config.schedule_id,
                    "schedule_name": config.name,
                    "cron_expression": config.cron_expression
                }
            )

            # Update schedule config
            config.last_run = datetime.now()

            # Calculate next run time
            cron = croniter.croniter(config.cron_expression, config.last_run)
            config.next_run = cron.get_next(datetime)

            self.metrics["scheduled_events"] += 1

            self.logger.info(
                f"Scheduled event triggered: {config.name} ({event_id}), "
                f"next run: {config.next_run}"
            )

        except Exception as e:
            self.logger.error(f"Error triggering scheduled event {config.schedule_id}: {e}")

    async def _send_event_webhooks(self, event: Event):
        """Send webhooks for an event."""
        try:
            # Find webhooks that should receive this event
            relevant_webhooks = [
                config for config in self.webhook_configs.values()
                if (config.enabled and event.event_type in config.events)
            ]

            for webhook_config in relevant_webhooks:
                await self._send_webhook(webhook_config, event)

        except Exception as e:
            self.logger.error(f"Error sending event webhooks: {e}")

    async def _send_webhook(self, webhook_config: WebhookConfig, event: Event):
        """Send a webhook for an event."""
        try:
            # Prepare webhook payload
            payload = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "data": event.data,
                "metadata": event.metadata
            }

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                **webhook_config.headers
            }

            # Add signature if secret is configured
            if webhook_config.secret:
                import hmac
                import hashlib

                payload_str = json.dumps(payload, sort_keys=True)
                signature = hmac.new(
                    webhook_config.secret.encode(),
                    payload_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                headers["X-Webhook-Signature"] = f"sha256={signature}"

            # Send webhook with retries
            for attempt in range(webhook_config.retry_attempts):
                try:
                    response = await self.http_client.post(
                        webhook_config.url,
                        json=payload,
                        headers=headers,
                        timeout=webhook_config.timeout
                    )

                    if response.status_code < 400:
                        self.metrics["webhook_calls"] += 1
                        self.logger.debug(
                            f"Webhook sent successfully: {webhook_config.webhook_id} "
                            f"for event {event.event_id}"
                        )
                        break
                    else:
                        self.logger.warning(
                            f"Webhook failed with status {response.status_code}: "
                            f"{webhook_config.webhook_id}"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Webhook attempt {attempt + 1} failed for "
                        f"{webhook_config.webhook_id}: {e}"
                    )

                    if attempt < webhook_config.retry_attempts - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff

        except Exception as e:
            self.logger.error(f"Error sending webhook: {e}")


# Global event system instance
event_system = EventDrivenSystem()
