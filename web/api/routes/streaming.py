"""
Streaming API Routes for Phase 2 Enhancement

This module provides API endpoints for real-time streaming capabilities,
change detection, and event-driven features.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from web.streaming.real_time_engine import streaming_engine, StreamConfig, StreamType
from web.monitoring.change_detection import change_detection_engine, MonitoringTarget, MonitoringFrequency
from web.events.event_system import event_system, EventType, EventPriority, WebhookConfig, ScheduleConfig
from web.api.dependencies import get_config
from config.web_config import WebConfig


# Configure logging
logger = logging.getLogger("streaming_api")

# Create router
router = APIRouter()


# ===== PYDANTIC MODELS =====

class StreamCreateRequest(BaseModel):
    """Request model for creating a stream."""
    stream_type: str = Field(..., description="Type of stream (websocket, sse, json_stream, csv_stream)")
    buffer_size: int = Field(1000, description="Buffer size for the stream")
    batch_size: int = Field(10, description="Batch size for processing")
    flush_interval: float = Field(1.0, description="Flush interval in seconds")
    compression: bool = Field(False, description="Enable compression")
    format_options: Optional[Dict[str, Any]] = Field(None, description="Format-specific options")
    filters: Optional[Dict[str, Any]] = Field(None, description="Data filters")


class MonitoringTargetRequest(BaseModel):
    """Request model for adding monitoring targets."""
    url: str = Field(..., description="URL to monitor")
    name: Optional[str] = Field(None, description="Name for the monitoring target")
    frequency: str = Field("medium", description="Monitoring frequency")
    custom_interval: Optional[int] = Field(None, description="Custom interval in seconds")
    selectors: List[str] = Field([], description="CSS selectors to monitor")
    ignore_patterns: List[str] = Field([], description="Patterns to ignore")
    threshold: float = Field(0.1, description="Change threshold (0.0 - 1.0)")
    enabled: bool = Field(True, description="Enable monitoring")


class WebhookRequest(BaseModel):
    """Request model for webhook configuration."""
    url: str = Field(..., description="Webhook URL")
    secret: Optional[str] = Field(None, description="Webhook secret for signature")
    events: List[str] = Field(..., description="Event types to send")
    headers: Dict[str, str] = Field({}, description="Additional headers")
    timeout: int = Field(30, description="Timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")
    enabled: bool = Field(True, description="Enable webhook")


class ScheduleRequest(BaseModel):
    """Request model for schedule configuration."""
    name: str = Field(..., description="Schedule name")
    cron_expression: str = Field(..., description="Cron expression")
    event_type: str = Field(..., description="Event type to trigger")
    event_data: Dict[str, Any] = Field({}, description="Event data")
    enabled: bool = Field(True, description="Enable schedule")


class EventEmitRequest(BaseModel):
    """Request model for emitting events."""
    event_type: str = Field(..., description="Event type")
    data: Dict[str, Any] = Field(..., description="Event data")
    priority: str = Field("normal", description="Event priority")
    source: str = Field("api", description="Event source")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Event metadata")
    scheduled_for: Optional[datetime] = Field(None, description="Schedule event for later")


# ===== STREAMING ENDPOINTS =====

@router.post("/streams", response_model=Dict[str, str])
async def create_stream(request: StreamCreateRequest):
    """Create a new data stream."""
    try:
        # Validate stream type
        try:
            stream_type = StreamType(request.stream_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid stream type: {request.stream_type}")
        
        # Create stream configuration
        config = StreamConfig(
            stream_id="",  # Will be generated
            stream_type=stream_type,
            buffer_size=request.buffer_size,
            batch_size=request.batch_size,
            flush_interval=request.flush_interval,
            compression=request.compression,
            format_options=request.format_options or {},
            filters=request.filters or {}
        )
        
        # Create stream
        stream_id = await streaming_engine.create_stream(config)
        
        return {"stream_id": stream_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Error creating stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/streams/{stream_id}/ws")
async def websocket_stream_endpoint(websocket: WebSocket, stream_id: str, client_id: str = Query(...)):
    """WebSocket endpoint for real-time streaming."""
    try:
        # Add WebSocket connection to stream
        connected = await streaming_engine.add_websocket_connection(stream_id, websocket, client_id)
        
        if not connected:
            return
        
        try:
            # Keep connection alive and handle messages
            while True:
                try:
                    # Wait for messages (ping/pong, subscription updates, etc.)
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    
                    # Handle different message types
                    if data.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    break
        
        finally:
            # Clean up connection
            await streaming_engine.remove_websocket_connection(stream_id, client_id)
    
    except Exception as e:
        logger.error(f"Error in WebSocket stream endpoint: {e}")


@router.get("/streams/{stream_id}/sse")
async def sse_stream_endpoint(stream_id: str):
    """Server-Sent Events endpoint for streaming."""
    try:
        return await streaming_engine.create_sse_response(stream_id)
        
    except Exception as e:
        logger.error(f"Error creating SSE response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams/{stream_id}/json")
async def json_stream_endpoint(stream_id: str):
    """JSON streaming endpoint."""
    try:
        async def json_generator():
            async for data in streaming_engine.get_stream_generator(stream_id, "json"):
                yield data
        
        return StreamingResponse(
            json_generator(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        logger.error(f"Error creating JSON stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams/{stream_id}/csv")
async def csv_stream_endpoint(stream_id: str):
    """CSV streaming endpoint."""
    try:
        async def csv_generator():
            async for data in streaming_engine.get_stream_generator(stream_id, "csv"):
                yield data
        
        return StreamingResponse(
            csv_generator(),
            media_type="text/csv",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        logger.error(f"Error creating CSV stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/streams/{stream_id}/publish")
async def publish_to_stream(stream_id: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
    """Publish data to a stream."""
    try:
        success = await streaming_engine.publish_to_stream(stream_id, data, metadata)
        
        if success:
            return {"status": "published", "stream_id": stream_id}
        else:
            raise HTTPException(status_code=404, detail="Stream not found")
        
    except Exception as e:
        logger.error(f"Error publishing to stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams/{stream_id}/metrics")
async def get_stream_metrics(stream_id: str):
    """Get metrics for a specific stream."""
    try:
        metrics = await streaming_engine.get_stream_metrics(stream_id)
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting stream metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams/metrics")
async def get_all_stream_metrics():
    """Get metrics for all streams."""
    try:
        metrics = await streaming_engine.get_stream_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting all stream metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== CHANGE DETECTION ENDPOINTS =====

@router.post("/monitoring/targets", response_model=Dict[str, str])
async def add_monitoring_target(request: MonitoringTargetRequest):
    """Add a new monitoring target."""
    try:
        # Validate frequency
        try:
            frequency = MonitoringFrequency(request.frequency)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid frequency: {request.frequency}")
        
        # Create target configuration
        target_config = {
            "url": request.url,
            "name": request.name or request.url,
            "frequency": request.frequency,
            "custom_interval": request.custom_interval,
            "selectors": request.selectors,
            "ignore_patterns": request.ignore_patterns,
            "threshold": request.threshold,
            "enabled": request.enabled
        }
        
        # Add monitoring target
        target_id = await change_detection_engine.add_monitoring_target(target_config)
        
        return {"target_id": target_id, "status": "added"}
        
    except Exception as e:
        logger.error(f"Error adding monitoring target: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/monitoring/targets/{target_id}")
async def remove_monitoring_target(target_id: str):
    """Remove a monitoring target."""
    try:
        success = await change_detection_engine.remove_monitoring_target(target_id)
        
        if success:
            return {"status": "removed", "target_id": target_id}
        else:
            raise HTTPException(status_code=404, detail="Monitoring target not found")
        
    except Exception as e:
        logger.error(f"Error removing monitoring target: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/targets/{target_id}/check")
async def check_target_changes(target_id: str):
    """Manually check for changes on a monitoring target."""
    try:
        changes = await change_detection_engine.check_for_changes(target_id)
        
        return {
            "target_id": target_id,
            "changes_detected": len(changes),
            "changes": [
                {
                    "change_id": change.change_id,
                    "change_type": change.change_type.value,
                    "description": change.description,
                    "confidence": change.confidence,
                    "timestamp": change.timestamp.isoformat()
                }
                for change in changes
            ]
        }
        
    except Exception as e:
        logger.error(f"Error checking target changes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/changes")
async def get_change_events(
    target_id: Optional[str] = Query(None),
    since: Optional[datetime] = Query(None),
    change_types: Optional[List[str]] = Query(None)
):
    """Get change events with optional filtering."""
    try:
        # Convert change types if provided
        change_type_enums = None
        if change_types:
            try:
                from web.monitoring.change_detection import ChangeType
                change_type_enums = [ChangeType(ct) for ct in change_types]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid change type: {e}")
        
        events = await change_detection_engine.get_change_events(
            target_id=target_id,
            since=since,
            change_types=change_type_enums
        )
        
        return {
            "events": [
                {
                    "change_id": event.change_id,
                    "url": event.url,
                    "change_type": event.change_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "description": event.description,
                    "confidence": event.confidence,
                    "old_value": event.old_value,
                    "new_value": event.new_value,
                    "xpath": event.xpath,
                    "metadata": event.metadata
                }
                for event in events
            ],
            "total_events": len(events)
        }
        
    except Exception as e:
        logger.error(f"Error getting change events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/status")
async def get_monitoring_status():
    """Get current monitoring status and metrics."""
    try:
        status = await change_detection_engine.get_monitoring_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== EVENT SYSTEM ENDPOINTS =====

@router.post("/events/emit", response_model=Dict[str, str])
async def emit_event(request: EventEmitRequest):
    """Emit a new event."""
    try:
        # Validate event type
        try:
            event_type = EventType(request.event_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {request.event_type}")

        # Validate priority
        try:
            priority = EventPriority(request.priority)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {request.priority}")

        # Emit event
        event_id = await event_system.emit_event(
            event_type=event_type,
            data=request.data,
            priority=priority,
            source=request.source,
            metadata=request.metadata,
            scheduled_for=request.scheduled_for
        )

        return {"event_id": event_id, "status": "emitted"}

    except Exception as e:
        logger.error(f"Error emitting event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/webhooks", response_model=Dict[str, str])
async def add_webhook_config(request: WebhookRequest):
    """Add a webhook configuration."""
    try:
        # Validate event types
        try:
            event_types = [EventType(et) for et in request.events]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {e}")

        # Create webhook configuration
        webhook_config = WebhookConfig(
            webhook_id=f"webhook_{int(datetime.now().timestamp())}",
            url=request.url,
            secret=request.secret,
            events=event_types,
            headers=request.headers,
            timeout=request.timeout,
            retry_attempts=request.retry_attempts,
            enabled=request.enabled
        )

        # Add webhook
        webhook_id = await event_system.add_webhook_config(webhook_config)

        return {"webhook_id": webhook_id, "status": "added"}

    except Exception as e:
        logger.error(f"Error adding webhook config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/schedules", response_model=Dict[str, str])
async def add_schedule_config(request: ScheduleRequest):
    """Add a schedule configuration."""
    try:
        # Validate event type
        try:
            event_type = EventType(request.event_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {request.event_type}")

        # Validate cron expression
        try:
            import croniter
            croniter.croniter(request.cron_expression)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid cron expression: {e}")

        # Create schedule configuration
        schedule_config = ScheduleConfig(
            schedule_id=f"schedule_{int(datetime.now().timestamp())}",
            name=request.name,
            cron_expression=request.cron_expression,
            event_type=event_type,
            event_data=request.event_data,
            enabled=request.enabled
        )

        # Add schedule
        schedule_id = await event_system.add_schedule_config(schedule_config)

        return {"schedule_id": schedule_id, "status": "added"}

    except Exception as e:
        logger.error(f"Error adding schedule config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/webhook")
async def process_webhook(webhook_data: Dict[str, Any], source: str = "external_webhook"):
    """Process incoming webhook data."""
    try:
        event_id = await event_system.process_webhook(webhook_data, source)

        return {"event_id": event_id, "status": "processed"}

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/{event_id}/status")
async def get_event_status(event_id: str):
    """Get the status of an event."""
    try:
        status = await event_system.get_event_status(event_id)

        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Event not found")

    except Exception as e:
        logger.error(f"Error getting event status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/metrics")
async def get_event_system_metrics():
    """Get event system metrics and status."""
    try:
        metrics = await event_system.get_system_metrics()
        return metrics

    except Exception as e:
        logger.error(f"Error getting event system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== COMBINED REAL-TIME ENDPOINTS =====

@router.get("/realtime/status")
async def get_realtime_status():
    """Get overall real-time system status."""
    try:
        # Get status from all systems
        streaming_metrics = await streaming_engine.get_stream_metrics()
        monitoring_status = await change_detection_engine.get_monitoring_status()
        event_metrics = await event_system.get_system_metrics()

        return {
            "timestamp": datetime.now().isoformat(),
            "streaming": streaming_metrics,
            "monitoring": monitoring_status,
            "events": event_metrics,
            "overall_status": "operational"
        }

    except Exception as e:
        logger.error(f"Error getting real-time status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/realtime/start")
async def start_realtime_systems():
    """Start all real-time systems."""
    try:
        # Start all systems
        await streaming_engine.start()
        await change_detection_engine.start()
        await event_system.start()

        return {"status": "started", "systems": ["streaming", "monitoring", "events"]}

    except Exception as e:
        logger.error(f"Error starting real-time systems: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/realtime/stop")
async def stop_realtime_systems():
    """Stop all real-time systems."""
    try:
        # Stop all systems
        await streaming_engine.stop()
        await change_detection_engine.stop()
        await event_system.stop()

        return {"status": "stopped", "systems": ["streaming", "monitoring", "events"]}

    except Exception as e:
        logger.error(f"Error stopping real-time systems: {e}")
        raise HTTPException(status_code=500, detail=str(e))
