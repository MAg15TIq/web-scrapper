"""
WebSocket API Routes
Handles real-time communication via WebSockets for live updates.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel, ValidationError

from web.api.dependencies import get_agent_monitor, get_job_manager, get_config
from web.dashboard.agent_monitor import AgentMonitor
from web.scheduler.job_manager import JobManager
from config.web_config import WebConfig


# Configure logging
logger = logging.getLogger("websocket_api")

# Create router
router = APIRouter()


@dataclass
class WebSocketConnection:
    """WebSocket connection information."""
    websocket: WebSocket
    client_id: str
    connected_at: datetime
    subscriptions: Set[str]
    user_id: Optional[str] = None
    last_ping: Optional[datetime] = None


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str
    payload: Dict[str, Any]
    timestamp: Optional[datetime] = None
    client_id: Optional[str] = None


class SubscriptionRequest(BaseModel):
    """Subscription request model."""
    channels: List[str]
    filters: Optional[Dict[str, Any]] = None


# Global connection manager
class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.channel_subscriptions: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger("connection_manager")
    
    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """Accept a new WebSocket connection."""
        try:
            await websocket.accept()
            
            connection = WebSocketConnection(
                websocket=websocket,
                client_id=client_id,
                connected_at=datetime.now(),
                subscriptions=set()
            )
            
            self.active_connections[client_id] = connection
            self.logger.info(f"Client connected: {client_id}")
            
            # Send welcome message
            await self.send_personal_message({
                "type": "connection_established",
                "payload": {
                    "client_id": client_id,
                    "server_time": datetime.now().isoformat(),
                    "available_channels": [
                        "agent_status",
                        "job_updates",
                        "system_metrics",
                        "alerts",
                        "logs"
                    ]
                }
            }, client_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting client {client_id}: {e}")
            return False
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            connection = self.active_connections[client_id]
            
            # Remove from all channel subscriptions
            for channel in connection.subscriptions:
                if channel in self.channel_subscriptions:
                    self.channel_subscriptions[channel].discard(client_id)
                    if not self.channel_subscriptions[channel]:
                        del self.channel_subscriptions[channel]
            
            del self.active_connections[client_id]
            self.logger.info(f"Client disconnected: {client_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                connection = self.active_connections[client_id]
                message["timestamp"] = datetime.now().isoformat()
                message["client_id"] = client_id
                
                await connection.websocket.send_text(json.dumps(message))
                
            except Exception as e:
                self.logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_channel(self, channel: str, message: Dict[str, Any]):
        """Broadcast a message to all clients subscribed to a channel."""
        if channel in self.channel_subscriptions:
            message["timestamp"] = datetime.now().isoformat()
            message["channel"] = channel
            
            disconnected_clients = []
            
            for client_id in self.channel_subscriptions[channel]:
                try:
                    if client_id in self.active_connections:
                        connection = self.active_connections[client_id]
                        await connection.websocket.send_text(json.dumps(message))
                except Exception as e:
                    self.logger.error(f"Error broadcasting to {client_id}: {e}")
                    disconnected_clients.append(client_id)
            
            # Clean up disconnected clients
            for client_id in disconnected_clients:
                self.disconnect(client_id)
    
    def subscribe_to_channel(self, client_id: str, channel: str):
        """Subscribe a client to a channel."""
        if client_id in self.active_connections:
            connection = self.active_connections[client_id]
            connection.subscriptions.add(channel)
            
            if channel not in self.channel_subscriptions:
                self.channel_subscriptions[channel] = set()
            
            self.channel_subscriptions[channel].add(client_id)
            self.logger.debug(f"Client {client_id} subscribed to {channel}")
    
    def unsubscribe_from_channel(self, client_id: str, channel: str):
        """Unsubscribe a client from a channel."""
        if client_id in self.active_connections:
            connection = self.active_connections[client_id]
            connection.subscriptions.discard(channel)
            
            if channel in self.channel_subscriptions:
                self.channel_subscriptions[channel].discard(client_id)
                if not self.channel_subscriptions[channel]:
                    del self.channel_subscriptions[channel]
            
            self.logger.debug(f"Client {client_id} unsubscribed from {channel}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self.active_connections),
            "active_channels": len(self.channel_subscriptions),
            "connections_by_channel": {
                channel: len(clients) 
                for channel, clients in self.channel_subscriptions.items()
            },
            "oldest_connection": min(
                (conn.connected_at for conn in self.active_connections.values()),
                default=None
            )
        }


# Global connection manager instance
connection_manager = ConnectionManager()


@router.websocket("/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    config: WebConfig = Depends(get_config)
):
    """
    Main WebSocket endpoint for real-time communication.
    
    Handles client connections and message routing for real-time updates
    including agent status, job progress, and system metrics.
    """
    if not config.enable_websockets:
        await websocket.close(code=1000, reason="WebSocket support is disabled")
        return
    
    # Connect client
    connected = await connection_manager.connect(websocket, client_id)
    if not connected:
        return
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                message = WebSocketMessage(**message_data)
                
                # Handle different message types
                await handle_websocket_message(message, client_id)
                
            except ValidationError as e:
                await connection_manager.send_personal_message({
                    "type": "error",
                    "payload": {
                        "message": "Invalid message format",
                        "details": str(e)
                    }
                }, client_id)
            
            except json.JSONDecodeError:
                await connection_manager.send_personal_message({
                    "type": "error",
                    "payload": {
                        "message": "Invalid JSON format"
                    }
                }, client_id)
    
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        connection_manager.disconnect(client_id)


async def handle_websocket_message(message: WebSocketMessage, client_id: str):
    """Handle incoming WebSocket messages."""
    try:
        if message.type == "subscribe":
            # Handle subscription requests
            channels = message.payload.get("channels", [])
            for channel in channels:
                if channel in ["agent_status", "job_updates", "system_metrics", "alerts", "logs"]:
                    connection_manager.subscribe_to_channel(client_id, channel)
            
            await connection_manager.send_personal_message({
                "type": "subscription_confirmed",
                "payload": {
                    "subscribed_channels": channels
                }
            }, client_id)
        
        elif message.type == "unsubscribe":
            # Handle unsubscription requests
            channels = message.payload.get("channels", [])
            for channel in channels:
                connection_manager.unsubscribe_from_channel(client_id, channel)
            
            await connection_manager.send_personal_message({
                "type": "unsubscription_confirmed",
                "payload": {
                    "unsubscribed_channels": channels
                }
            }, client_id)
        
        elif message.type == "ping":
            # Handle ping messages
            await connection_manager.send_personal_message({
                "type": "pong",
                "payload": {
                    "timestamp": datetime.now().isoformat()
                }
            }, client_id)
        
        elif message.type == "get_status":
            # Handle status requests
            await send_current_status(client_id)
        
        else:
            await connection_manager.send_personal_message({
                "type": "error",
                "payload": {
                    "message": f"Unknown message type: {message.type}"
                }
            }, client_id)
    
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")
        await connection_manager.send_personal_message({
            "type": "error",
            "payload": {
                "message": "Internal server error"
            }
        }, client_id)


async def send_current_status(client_id: str):
    """Send current system status to a client."""
    try:
        # Get current status from various sources
        status_data = {
            "agents": {
                "total": 5,
                "active": 4,
                "idle": 1
            },
            "jobs": {
                "pending": 3,
                "running": 2,
                "completed": 15
            },
            "system": {
                "cpu_usage": 45.2,
                "memory_usage": 62.8,
                "uptime": 3600
            }
        }
        
        await connection_manager.send_personal_message({
            "type": "status_update",
            "payload": status_data
        }, client_id)
        
    except Exception as e:
        logger.error(f"Error sending status to {client_id}: {e}")


# Background task to broadcast updates
async def broadcast_updates():
    """Background task to broadcast periodic updates."""
    while True:
        try:
            # Broadcast agent status updates
            await connection_manager.broadcast_to_channel("agent_status", {
                "type": "agent_status_update",
                "payload": {
                    "agents": [
                        {
                            "id": "agent_001",
                            "type": "scraper",
                            "status": "active",
                            "tasks": 3
                        }
                    ]
                }
            })
            
            # Broadcast system metrics
            await connection_manager.broadcast_to_channel("system_metrics", {
                "type": "metrics_update",
                "payload": {
                    "cpu_usage": 45.2,
                    "memory_usage": 62.8,
                    "active_connections": len(connection_manager.active_connections)
                }
            })
            
            # Wait before next update
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in broadcast updates: {e}")
            await asyncio.sleep(10)


# Function to handle WebSocket connections (called from main.py)
async def handle_websocket_connection(websocket: WebSocket, client_id: str):
    """Handle WebSocket connection (called from main.py)."""
    await websocket_endpoint(websocket, client_id)


# Function to get connection manager (for external use)
def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager."""
    return connection_manager
