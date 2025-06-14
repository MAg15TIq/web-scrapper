"""
Real-time Streaming Engine for Phase 2 Enhancement

This module provides live data streaming capabilities with WebSocket support,
Server-Sent Events (SSE), and streaming JSON/CSV output for large datasets.
"""

import asyncio
import json
import logging
import time
import csv
import io
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse


# Configure logging
logger = logging.getLogger("real_time_engine")


class StreamType(Enum):
    """Types of data streams."""
    WEBSOCKET = "websocket"
    SSE = "sse"
    JSON_STREAM = "json_stream"
    CSV_STREAM = "csv_stream"
    LIVE_UPDATES = "live_updates"


class StreamStatus(Enum):
    """Stream status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StreamMetrics:
    """Metrics for stream performance monitoring."""
    stream_id: str
    stream_type: StreamType
    start_time: datetime
    messages_sent: int
    bytes_sent: int
    active_connections: int
    error_count: int
    last_activity: datetime


@dataclass
class StreamConfig:
    """Configuration for data streams."""
    stream_id: str
    stream_type: StreamType
    buffer_size: int = 1000
    batch_size: int = 10
    flush_interval: float = 1.0
    compression: bool = False
    format_options: Dict[str, Any] = None
    filters: Dict[str, Any] = None


class RealTimeStreamingEngine:
    """
    Enhanced Real-time Streaming Engine for Phase 2.
    
    Features:
    - WebSocket-based real-time data delivery
    - Server-Sent Events (SSE) for live updates
    - Streaming JSON/CSV output for large datasets
    - Real-time progress tracking with granular metrics
    - Intelligent buffering and batching
    """
    
    def __init__(self):
        self.logger = logging.getLogger("real_time_streaming")
        
        # Active streams and connections
        self.active_streams: Dict[str, StreamConfig] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.sse_connections: Dict[str, Any] = {}
        
        # Stream buffers and queues
        self.stream_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.stream_subscribers: Dict[str, Set[str]] = defaultdict(set)
        
        # Metrics and monitoring
        self.stream_metrics: Dict[str, StreamMetrics] = {}
        self.global_metrics = {
            "total_streams": 0,
            "active_connections": 0,
            "messages_processed": 0,
            "bytes_transferred": 0,
            "uptime": time.time()
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
        
        self.logger.info("Real-time streaming engine initialized")
    
    async def start(self):
        """Start the streaming engine."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.background_tasks.add(asyncio.create_task(self._metrics_collector()))
        self.background_tasks.add(asyncio.create_task(self._buffer_flusher()))
        self.background_tasks.add(asyncio.create_task(self._connection_monitor()))
        
        self.logger.info("Real-time streaming engine started")
    
    async def stop(self):
        """Stop the streaming engine."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        # Close all connections
        await self._close_all_connections()
        
        self.logger.info("Real-time streaming engine stopped")
    
    async def create_stream(self, config: StreamConfig) -> str:
        """Create a new data stream."""
        try:
            stream_id = config.stream_id or str(uuid.uuid4())
            config.stream_id = stream_id
            
            # Store stream configuration
            self.active_streams[stream_id] = config
            
            # Initialize stream metrics
            self.stream_metrics[stream_id] = StreamMetrics(
                stream_id=stream_id,
                stream_type=config.stream_type,
                start_time=datetime.now(),
                messages_sent=0,
                bytes_sent=0,
                active_connections=0,
                error_count=0,
                last_activity=datetime.now()
            )
            
            # Initialize stream buffer
            self.stream_buffers[stream_id] = deque(maxlen=config.buffer_size)
            
            self.global_metrics["total_streams"] += 1
            
            self.logger.info(f"Stream created: {stream_id} ({config.stream_type.value})")
            return stream_id
            
        except Exception as e:
            self.logger.error(f"Error creating stream: {e}")
            raise
    
    async def add_websocket_connection(self, stream_id: str, websocket: WebSocket, client_id: str) -> bool:
        """Add a WebSocket connection to a stream."""
        try:
            if stream_id not in self.active_streams:
                await websocket.close(code=1000, reason="Stream not found")
                return False
            
            # Accept connection
            await websocket.accept()
            
            # Store connection
            connection_key = f"{stream_id}:{client_id}"
            self.websocket_connections[connection_key] = websocket
            
            # Add to subscribers
            self.stream_subscribers[stream_id].add(client_id)
            
            # Update metrics
            if stream_id in self.stream_metrics:
                self.stream_metrics[stream_id].active_connections += 1
            
            self.global_metrics["active_connections"] += 1
            
            # Send welcome message
            await self._send_websocket_message(websocket, {
                "type": "stream_connected",
                "stream_id": stream_id,
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info(f"WebSocket connected to stream {stream_id}: {client_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding WebSocket connection: {e}")
            return False
    
    async def remove_websocket_connection(self, stream_id: str, client_id: str):
        """Remove a WebSocket connection from a stream."""
        try:
            connection_key = f"{stream_id}:{client_id}"
            
            if connection_key in self.websocket_connections:
                websocket = self.websocket_connections[connection_key]
                
                try:
                    await websocket.close()
                except:
                    pass  # Connection might already be closed
                
                del self.websocket_connections[connection_key]
                
                # Remove from subscribers
                self.stream_subscribers[stream_id].discard(client_id)
                
                # Update metrics
                if stream_id in self.stream_metrics:
                    self.stream_metrics[stream_id].active_connections = max(0, 
                        self.stream_metrics[stream_id].active_connections - 1)
                
                self.global_metrics["active_connections"] = max(0, 
                    self.global_metrics["active_connections"] - 1)
                
                self.logger.info(f"WebSocket disconnected from stream {stream_id}: {client_id}")
            
        except Exception as e:
            self.logger.error(f"Error removing WebSocket connection: {e}")
    
    async def publish_to_stream(self, stream_id: str, data: Dict[str, Any], 
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Publish data to a stream."""
        try:
            if stream_id not in self.active_streams:
                self.logger.warning(f"Stream not found: {stream_id}")
                return False
            
            # Prepare message
            message = {
                "stream_id": stream_id,
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "metadata": metadata or {}
            }
            
            # Add to buffer
            self.stream_buffers[stream_id].append(message)
            
            # Update metrics
            if stream_id in self.stream_metrics:
                metrics = self.stream_metrics[stream_id]
                metrics.messages_sent += 1
                metrics.last_activity = datetime.now()
                
                # Estimate message size
                message_size = len(json.dumps(message).encode('utf-8'))
                metrics.bytes_sent += message_size
                self.global_metrics["bytes_transferred"] += message_size
            
            self.global_metrics["messages_processed"] += 1
            
            # Immediate delivery for real-time streams
            config = self.active_streams[stream_id]
            if config.stream_type in [StreamType.WEBSOCKET, StreamType.LIVE_UPDATES]:
                await self._flush_stream_buffer(stream_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error publishing to stream {stream_id}: {e}")
            if stream_id in self.stream_metrics:
                self.stream_metrics[stream_id].error_count += 1
            return False
    
    async def get_stream_generator(self, stream_id: str, format_type: str = "json") -> AsyncGenerator[str, None]:
        """Get an async generator for streaming data."""
        if stream_id not in self.active_streams:
            raise ValueError(f"Stream not found: {stream_id}")
        
        config = self.active_streams[stream_id]
        buffer = self.stream_buffers[stream_id]
        
        # Send existing buffered data
        while buffer:
            message = buffer.popleft()
            
            if format_type == "json":
                yield json.dumps(message) + "\n"
            elif format_type == "csv":
                yield await self._format_as_csv(message)
            else:
                yield str(message) + "\n"
        
        # Continue streaming new data
        last_check = time.time()
        while stream_id in self.active_streams:
            current_time = time.time()
            
            # Check for new messages
            if buffer:
                batch = []
                batch_size = min(config.batch_size, len(buffer))
                
                for _ in range(batch_size):
                    if buffer:
                        batch.append(buffer.popleft())
                
                for message in batch:
                    if format_type == "json":
                        yield json.dumps(message) + "\n"
                    elif format_type == "csv":
                        yield await self._format_as_csv(message)
                    else:
                        yield str(message) + "\n"
            
            # Wait for flush interval
            await asyncio.sleep(config.flush_interval)
            
            # Check if stream is still active
            if current_time - last_check > 60:  # Check every minute
                if stream_id not in self.active_streams:
                    break
                last_check = current_time
    
    async def create_sse_response(self, stream_id: str) -> StreamingResponse:
        """Create a Server-Sent Events response for a stream."""
        async def event_generator():
            try:
                async for data in self.get_stream_generator(stream_id, "json"):
                    # Format as SSE
                    yield f"data: {data.strip()}\n\n"
                    
            except Exception as e:
                self.logger.error(f"Error in SSE generator: {e}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    
    async def get_stream_metrics(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for a specific stream or all streams."""
        if stream_id:
            if stream_id in self.stream_metrics:
                metrics = self.stream_metrics[stream_id]
                return {
                    "stream_id": metrics.stream_id,
                    "stream_type": metrics.stream_type.value,
                    "start_time": metrics.start_time.isoformat(),
                    "messages_sent": metrics.messages_sent,
                    "bytes_sent": metrics.bytes_sent,
                    "active_connections": metrics.active_connections,
                    "error_count": metrics.error_count,
                    "last_activity": metrics.last_activity.isoformat(),
                    "uptime_seconds": (datetime.now() - metrics.start_time).total_seconds()
                }
            else:
                return {"error": "Stream not found"}
        else:
            # Return global metrics and all stream metrics
            all_metrics = {
                "global": {
                    **self.global_metrics,
                    "uptime_seconds": time.time() - self.global_metrics["uptime"]
                },
                "streams": {}
            }
            
            for sid, metrics in self.stream_metrics.items():
                all_metrics["streams"][sid] = {
                    "stream_type": metrics.stream_type.value,
                    "messages_sent": metrics.messages_sent,
                    "bytes_sent": metrics.bytes_sent,
                    "active_connections": metrics.active_connections,
                    "error_count": metrics.error_count,
                    "uptime_seconds": (datetime.now() - metrics.start_time).total_seconds()
                }
            
            return all_metrics

    # ===== BACKGROUND TASKS =====

    async def _metrics_collector(self):
        """Background task to collect and update metrics."""
        while self.is_running:
            try:
                # Update global metrics
                self.global_metrics["active_connections"] = len(self.websocket_connections)

                # Clean up inactive streams
                current_time = datetime.now()
                inactive_streams = []

                for stream_id, metrics in self.stream_metrics.items():
                    # Mark streams as inactive if no activity for 1 hour
                    if (current_time - metrics.last_activity).total_seconds() > 3600:
                        if metrics.active_connections == 0:
                            inactive_streams.append(stream_id)

                # Remove inactive streams
                for stream_id in inactive_streams:
                    await self._cleanup_stream(stream_id)

                await asyncio.sleep(30)  # Collect metrics every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(60)

    async def _buffer_flusher(self):
        """Background task to flush stream buffers periodically."""
        while self.is_running:
            try:
                for stream_id in list(self.active_streams.keys()):
                    config = self.active_streams.get(stream_id)
                    if config and config.stream_type not in [StreamType.WEBSOCKET, StreamType.LIVE_UPDATES]:
                        await self._flush_stream_buffer(stream_id)

                await asyncio.sleep(1)  # Flush every second

            except Exception as e:
                self.logger.error(f"Error in buffer flusher: {e}")
                await asyncio.sleep(5)

    async def _connection_monitor(self):
        """Background task to monitor connection health."""
        while self.is_running:
            try:
                # Check WebSocket connections
                dead_connections = []

                for connection_key, websocket in self.websocket_connections.items():
                    try:
                        # Send ping to check if connection is alive
                        await websocket.ping()
                    except:
                        dead_connections.append(connection_key)

                # Remove dead connections
                for connection_key in dead_connections:
                    stream_id, client_id = connection_key.split(':', 1)
                    await self.remove_websocket_connection(stream_id, client_id)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in connection monitor: {e}")
                await asyncio.sleep(60)

    async def _flush_stream_buffer(self, stream_id: str):
        """Flush buffered messages for a stream."""
        try:
            if stream_id not in self.stream_buffers:
                return

            buffer = self.stream_buffers[stream_id]
            if not buffer:
                return

            config = self.active_streams.get(stream_id)
            if not config:
                return

            # Get messages to send
            messages_to_send = []
            batch_size = min(config.batch_size, len(buffer))

            for _ in range(batch_size):
                if buffer:
                    messages_to_send.append(buffer.popleft())

            if not messages_to_send:
                return

            # Send to WebSocket connections
            if config.stream_type in [StreamType.WEBSOCKET, StreamType.LIVE_UPDATES]:
                await self._send_to_websocket_subscribers(stream_id, messages_to_send)

        except Exception as e:
            self.logger.error(f"Error flushing stream buffer {stream_id}: {e}")

    async def _send_to_websocket_subscribers(self, stream_id: str, messages: List[Dict[str, Any]]):
        """Send messages to all WebSocket subscribers of a stream."""
        if stream_id not in self.stream_subscribers:
            return

        subscribers = list(self.stream_subscribers[stream_id])

        for client_id in subscribers:
            connection_key = f"{stream_id}:{client_id}"

            if connection_key in self.websocket_connections:
                websocket = self.websocket_connections[connection_key]

                try:
                    for message in messages:
                        await self._send_websocket_message(websocket, message)

                except WebSocketDisconnect:
                    await self.remove_websocket_connection(stream_id, client_id)
                except Exception as e:
                    self.logger.error(f"Error sending to WebSocket {client_id}: {e}")
                    await self.remove_websocket_connection(stream_id, client_id)

    async def _send_websocket_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send a message to a WebSocket connection."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Error sending WebSocket message: {e}")
            raise

    async def _format_as_csv(self, message: Dict[str, Any]) -> str:
        """Format a message as CSV."""
        try:
            data = message.get('data', {})
            if not isinstance(data, dict):
                return ""

            # Create CSV string
            output = io.StringIO()
            if data:
                writer = csv.DictWriter(output, fieldnames=data.keys())
                writer.writerow(data)
                return output.getvalue()

            return ""

        except Exception as e:
            self.logger.error(f"Error formatting as CSV: {e}")
            return ""

    async def _cleanup_stream(self, stream_id: str):
        """Clean up a stream and its resources."""
        try:
            # Remove from active streams
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]

            # Remove metrics
            if stream_id in self.stream_metrics:
                del self.stream_metrics[stream_id]

            # Remove buffer
            if stream_id in self.stream_buffers:
                del self.stream_buffers[stream_id]

            # Remove subscribers and close connections
            if stream_id in self.stream_subscribers:
                subscribers = list(self.stream_subscribers[stream_id])
                for client_id in subscribers:
                    await self.remove_websocket_connection(stream_id, client_id)
                del self.stream_subscribers[stream_id]

            self.logger.info(f"Stream cleaned up: {stream_id}")

        except Exception as e:
            self.logger.error(f"Error cleaning up stream {stream_id}: {e}")

    async def _close_all_connections(self):
        """Close all active connections."""
        try:
            # Close all WebSocket connections
            for connection_key, websocket in list(self.websocket_connections.items()):
                try:
                    await websocket.close()
                except:
                    pass

            self.websocket_connections.clear()
            self.stream_subscribers.clear()

            self.logger.info("All connections closed")

        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")


# Global streaming engine instance
streaming_engine = RealTimeStreamingEngine()
