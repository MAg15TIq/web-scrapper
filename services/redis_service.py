"""
Redis service for distributed agent communication and caching.
Provides message broker functionality and shared state management.
"""
import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    import redis.asyncio as redis
    import redis.exceptions
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Install redis-py for distributed communication.")

from config.langchain_config import get_config
from models.langchain_models import AgentMessage, TaskRequest, TaskResponse


@dataclass
class MessageSubscription:
    """Subscription configuration for message handling."""
    channel: str
    handler: Callable[[Dict[str, Any]], None]
    agent_id: str
    message_types: List[str] = None


class RedisService:
    """
    Redis service for distributed agent communication and state management.
    Provides pub/sub messaging, caching, and distributed locks.
    """
    
    def __init__(self):
        """Initialize Redis service."""
        self.logger = logging.getLogger("redis.service")
        self.config = get_config().redis
        
        # Redis connections
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        
        # Subscription management
        self.subscriptions: Dict[str, MessageSubscription] = {}
        self.running = False
        
        # Message statistics
        self.message_stats = {
            "sent": 0,
            "received": 0,
            "errors": 0,
            "last_activity": None
        }
        
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available - using mock implementation")
            self._use_mock = True
            self._mock_storage = {}
            self._mock_channels = {}
        else:
            self._use_mock = False
    
    async def connect(self) -> bool:
        """Connect to Redis server."""
        if self._use_mock:
            self.logger.info("Using mock Redis implementation")
            return True
        
        try:
            # Create Redis connections
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                ssl=self.config.redis_ssl,
                decode_responses=True,
                max_connections=self.config.redis_max_connections,
                retry_on_timeout=self.config.redis_retry_on_timeout,
                socket_timeout=self.config.redis_socket_timeout
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Create separate connection for pub/sub
            self.pubsub_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                ssl=self.config.redis_ssl,
                decode_responses=True
            )
            
            self.pubsub = self.pubsub_client.pubsub()
            
            self.logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            # Fall back to mock implementation
            self._use_mock = True
            self._mock_storage = {}
            self._mock_channels = {}
            return False
    
    async def disconnect(self):
        """Disconnect from Redis server."""
        if self._use_mock:
            return
        
        try:
            if self.pubsub:
                await self.pubsub.close()
            if self.redis_client:
                await self.redis_client.close()
            if self.pubsub_client:
                await self.pubsub_client.close()
            
            self.logger.info("Disconnected from Redis")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from Redis: {e}")
    
    async def publish_message(
        self,
        channel: str,
        message: Union[AgentMessage, Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Publish a message to a Redis channel.
        
        Args:
            channel: Channel name
            message: Message to publish
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        try:
            # Convert message to dict if needed
            if isinstance(message, AgentMessage):
                message_data = message.dict()
            else:
                message_data = message
            
            # Add metadata
            message_data["_timestamp"] = datetime.now().isoformat()
            message_data["_ttl"] = ttl
            
            message_json = json.dumps(message_data)
            
            if self._use_mock:
                # Mock implementation
                if channel not in self._mock_channels:
                    self._mock_channels[channel] = []
                self._mock_channels[channel].append(message_data)
                self.logger.debug(f"Mock published to {channel}: {message_data.get('message_type', 'unknown')}")
            else:
                # Real Redis implementation
                await self.redis_client.publish(channel, message_json)
                self.logger.debug(f"Published to {channel}: {message_data.get('message_type', 'unknown')}")
            
            self.message_stats["sent"] += 1
            self.message_stats["last_activity"] = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish message to {channel}: {e}")
            self.message_stats["errors"] += 1
            return False
    
    async def subscribe_to_channel(
        self,
        channel: str,
        handler: Callable[[Dict[str, Any]], None],
        agent_id: str,
        message_types: Optional[List[str]] = None
    ) -> bool:
        """
        Subscribe to a Redis channel.
        
        Args:
            channel: Channel name
            handler: Message handler function
            agent_id: ID of the subscribing agent
            message_types: Optional filter for message types
            
        Returns:
            Success status
        """
        try:
            subscription = MessageSubscription(
                channel=channel,
                handler=handler,
                agent_id=agent_id,
                message_types=message_types
            )
            
            self.subscriptions[channel] = subscription
            
            if not self._use_mock:
                await self.pubsub.subscribe(channel)
            
            self.logger.info(f"Agent {agent_id} subscribed to channel {channel}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {channel}: {e}")
            return False
    
    async def unsubscribe_from_channel(self, channel: str) -> bool:
        """Unsubscribe from a Redis channel."""
        try:
            if channel in self.subscriptions:
                del self.subscriptions[channel]
            
            if not self._use_mock and self.pubsub:
                await self.pubsub.unsubscribe(channel)
            
            self.logger.info(f"Unsubscribed from channel {channel}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from {channel}: {e}")
            return False
    
    async def start_message_listener(self):
        """Start the message listener loop."""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting Redis message listener")
        
        if self._use_mock:
            # Mock message listener
            await self._mock_message_listener()
        else:
            # Real Redis message listener
            await self._redis_message_listener()
    
    async def stop_message_listener(self):
        """Stop the message listener loop."""
        self.running = False
        self.logger.info("Stopping Redis message listener")
    
    async def _redis_message_listener(self):
        """Real Redis message listener implementation."""
        try:
            while self.running:
                message = await self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    await self._handle_message(message['channel'], message['data'])
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
        except Exception as e:
            self.logger.error(f"Error in Redis message listener: {e}")
    
    async def _mock_message_listener(self):
        """Mock message listener for testing without Redis."""
        while self.running:
            # Process mock messages
            for channel, messages in self._mock_channels.items():
                if messages and channel in self.subscriptions:
                    message_data = messages.pop(0)
                    await self._handle_message(channel, json.dumps(message_data))
            
            await asyncio.sleep(0.1)  # Check for messages every 100ms
    
    async def _handle_message(self, channel: str, message_data: str):
        """Handle incoming message."""
        try:
            message = json.loads(message_data)
            
            # Check TTL
            if "_ttl" in message and message["_ttl"]:
                message_time = datetime.fromisoformat(message["_timestamp"])
                if datetime.now() - message_time > timedelta(seconds=message["_ttl"]):
                    self.logger.debug(f"Message expired, ignoring: {message.get('id', 'unknown')}")
                    return
            
            # Find subscription
            if channel in self.subscriptions:
                subscription = self.subscriptions[channel]
                
                # Filter by message type if specified
                if subscription.message_types:
                    message_type = message.get("message_type")
                    if message_type not in subscription.message_types:
                        return
                
                # Call handler
                try:
                    await subscription.handler(message)
                    self.message_stats["received"] += 1
                    self.message_stats["last_activity"] = datetime.now()
                except Exception as e:
                    self.logger.error(f"Error in message handler for {channel}: {e}")
                    self.message_stats["errors"] += 1
            
        except Exception as e:
            self.logger.error(f"Error handling message from {channel}: {e}")
            self.message_stats["errors"] += 1
    
    async def set_cache(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set a value in Redis cache."""
        try:
            value_json = json.dumps(value)
            
            if self._use_mock:
                self._mock_storage[key] = {
                    "value": value,
                    "expires": datetime.now() + timedelta(seconds=ttl) if ttl else None
                }
            else:
                if ttl:
                    await self.redis_client.setex(key, ttl, value_json)
                else:
                    await self.redis_client.set(key, value_json)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    async def get_cache(self, key: str) -> Optional[Any]:
        """Get a value from Redis cache."""
        try:
            if self._use_mock:
                if key in self._mock_storage:
                    item = self._mock_storage[key]
                    if item["expires"] and datetime.now() > item["expires"]:
                        del self._mock_storage[key]
                        return None
                    return item["value"]
                return None
            else:
                value_json = await self.redis_client.get(key)
                if value_json:
                    return json.loads(value_json)
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """Delete a value from Redis cache."""
        try:
            if self._use_mock:
                if key in self._mock_storage:
                    del self._mock_storage[key]
            else:
                await self.redis_client.delete(key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    async def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10,
        blocking_timeout: int = 5
    ) -> Optional[str]:
        """Acquire a distributed lock."""
        if self._use_mock:
            # Simple mock lock implementation
            lock_key = f"lock:{lock_name}"
            if lock_key not in self._mock_storage:
                lock_id = f"lock_{int(time.time())}"
                self._mock_storage[lock_key] = {
                    "value": lock_id,
                    "expires": datetime.now() + timedelta(seconds=timeout)
                }
                return lock_id
            return None
        
        try:
            lock = self.redis_client.lock(
                lock_name,
                timeout=timeout,
                blocking_timeout=blocking_timeout
            )
            
            if await lock.acquire():
                return str(lock.token)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to acquire lock {lock_name}: {e}")
            return None
    
    async def release_lock(self, lock_name: str, lock_id: str) -> bool:
        """Release a distributed lock."""
        if self._use_mock:
            lock_key = f"lock:{lock_name}"
            if lock_key in self._mock_storage:
                if self._mock_storage[lock_key]["value"] == lock_id:
                    del self._mock_storage[lock_key]
                    return True
            return False
        
        try:
            lock = self.redis_client.lock(lock_name)
            lock.token = lock_id.encode()
            await lock.release()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to release lock {lock_name}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis service statistics."""
        return {
            **self.message_stats,
            "connected": not self._use_mock,
            "mock_mode": self._use_mock,
            "subscriptions": len(self.subscriptions),
            "cache_keys": len(self._mock_storage) if self._use_mock else "unknown"
        }


# Global Redis service instance
_redis_service: Optional[RedisService] = None


async def get_redis_service() -> RedisService:
    """Get the global Redis service instance."""
    global _redis_service
    
    if _redis_service is None:
        _redis_service = RedisService()
        await _redis_service.connect()
    
    return _redis_service


async def cleanup_redis_service():
    """Cleanup the global Redis service."""
    global _redis_service
    
    if _redis_service:
        await _redis_service.stop_message_listener()
        await _redis_service.disconnect()
        _redis_service = None
