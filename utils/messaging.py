"""
Messaging utility for inter-agent communication.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
import json

from models.message import Message


class MessageBroker:
    """
    Message broker for handling communication between agents.
    """
    def __init__(self):
        """Initialize a new message broker."""
        self.subscribers: Dict[str, List[Callable[[Message], Awaitable[None]]]] = {}
        self.logger = logging.getLogger("message_broker")
    
    def subscribe(self, agent_id: str, callback: Callable[[Message], Awaitable[None]]) -> None:
        """
        Subscribe an agent to receive messages.
        
        Args:
            agent_id: ID of the agent to subscribe.
            callback: Async function to call when a message is received.
        """
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        
        self.subscribers[agent_id].append(callback)
        self.logger.debug(f"Agent {agent_id} subscribed to messages")
    
    def unsubscribe(self, agent_id: str, callback: Optional[Callable[[Message], Awaitable[None]]] = None) -> None:
        """
        Unsubscribe an agent from receiving messages.
        
        Args:
            agent_id: ID of the agent to unsubscribe.
            callback: Specific callback to unsubscribe. If None, all callbacks for the agent are removed.
        """
        if agent_id not in self.subscribers:
            return
        
        if callback is None:
            # Remove all callbacks for this agent
            self.subscribers.pop(agent_id)
            self.logger.debug(f"Agent {agent_id} unsubscribed from all messages")
        else:
            # Remove specific callback
            self.subscribers[agent_id] = [cb for cb in self.subscribers[agent_id] if cb != callback]
            if not self.subscribers[agent_id]:
                self.subscribers.pop(agent_id)
            self.logger.debug(f"Agent {agent_id} unsubscribed from specific message callback")
    
    async def publish(self, message: Message) -> None:
        """
        Publish a message to its recipient.
        
        Args:
            message: The message to publish.
        """
        recipient_id = message.recipient_id
        
        if not recipient_id:
            self.logger.warning(f"Cannot publish message without recipient_id: {message}")
            return
        
        if recipient_id not in self.subscribers:
            self.logger.warning(f"No subscribers for recipient: {recipient_id}")
            return
        
        self.logger.debug(f"Publishing message from {message.sender_id} to {recipient_id}: {message.type}")
        
        # Call all callbacks for this recipient
        tasks = []
        for callback in self.subscribers[recipient_id]:
            tasks.append(asyncio.create_task(callback(message)))
        
        # Wait for all callbacks to complete
        if tasks:
            await asyncio.gather(*tasks)


class LocalMessageBroker(MessageBroker):
    """
    Local implementation of the message broker for in-process communication.
    """
    _instance = None
    
    def __new__(cls):
        """Implement the Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(LocalMessageBroker, cls).__new__(cls)
            cls._instance.subscribers = {}
            cls._instance.logger = logging.getLogger("local_message_broker")
        return cls._instance


class RedisMessageBroker(MessageBroker):
    """
    Redis-based implementation of the message broker for distributed communication.
    """
    def __init__(self, redis_url: str = "redis://localhost:6379/0", channel_prefix: str = "agent:"):
        """
        Initialize a new Redis message broker.
        
        Args:
            redis_url: URL of the Redis server.
            channel_prefix: Prefix for Redis channels.
        """
        super().__init__()
        self.redis_url = redis_url
        self.channel_prefix = channel_prefix
        self.redis_client = None
        self.pubsub = None
        self.running = False
        self.logger = logging.getLogger("redis_message_broker")
    
    async def connect(self) -> None:
        """Connect to the Redis server."""
        try:
            import redis.asyncio as redis
            
            self.redis_client = redis.from_url(self.redis_url)
            self.pubsub = self.redis_client.pubsub()
            self.logger.info(f"Connected to Redis at {self.redis_url}")
        except ImportError:
            self.logger.error("Redis package not installed. Please install with: pip install redis")
            raise
        except Exception as e:
            self.logger.error(f"Error connecting to Redis: {str(e)}")
            raise
    
    async def start(self) -> None:
        """Start listening for messages."""
        if not self.redis_client:
            await self.connect()
        
        self.running = True
        asyncio.create_task(self._listen_for_messages())
        self.logger.info("Redis message broker started")
    
    async def stop(self) -> None:
        """Stop listening for messages."""
        self.running = False
        
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        self.logger.info("Redis message broker stopped")
    
    async def _listen_for_messages(self) -> None:
        """Listen for messages from Redis."""
        while self.running:
            try:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True)
                if message:
                    channel = message["channel"].decode("utf-8")
                    data = message["data"].decode("utf-8")
                    
                    # Extract agent ID from channel
                    if channel.startswith(self.channel_prefix):
                        agent_id = channel[len(self.channel_prefix):]
                        
                        # Parse message
                        try:
                            message_obj = Message.from_json(data)
                            
                            # Call callbacks for this agent
                            if agent_id in self.subscribers:
                                for callback in self.subscribers[agent_id]:
                                    await callback(message_obj)
                        except Exception as e:
                            self.logger.error(f"Error processing message: {str(e)}")
                
                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Error in message listener: {str(e)}")
                await asyncio.sleep(1.0)  # Sleep longer on error
    
    def subscribe(self, agent_id: str, callback: Callable[[Message], Awaitable[None]]) -> None:
        """
        Subscribe an agent to receive messages.
        
        Args:
            agent_id: ID of the agent to subscribe.
            callback: Async function to call when a message is received.
        """
        super().subscribe(agent_id, callback)
        
        # Subscribe to Redis channel
        if self.pubsub:
            asyncio.create_task(self.pubsub.subscribe(f"{self.channel_prefix}{agent_id}"))
    
    def unsubscribe(self, agent_id: str, callback: Optional[Callable[[Message], Awaitable[None]]] = None) -> None:
        """
        Unsubscribe an agent from receiving messages.
        
        Args:
            agent_id: ID of the agent to unsubscribe.
            callback: Specific callback to unsubscribe. If None, all callbacks for the agent are removed.
        """
        if agent_id not in self.subscribers:
            return
        
        if callback is None or len(self.subscribers[agent_id]) == 1:
            # Unsubscribe from Redis channel if removing all callbacks
            if self.pubsub:
                asyncio.create_task(self.pubsub.unsubscribe(f"{self.channel_prefix}{agent_id}"))
        
        super().unsubscribe(agent_id, callback)
    
    async def publish(self, message: Message) -> None:
        """
        Publish a message to its recipient.
        
        Args:
            message: The message to publish.
        """
        recipient_id = message.recipient_id
        
        if not recipient_id:
            self.logger.warning(f"Cannot publish message without recipient_id: {message}")
            return
        
        if not self.redis_client:
            self.logger.warning("Redis client not connected")
            return
        
        self.logger.debug(f"Publishing message from {message.sender_id} to {recipient_id}: {message.type}")
        
        # Publish message to Redis
        channel = f"{self.channel_prefix}{recipient_id}"
        await self.redis_client.publish(channel, message.to_json())
