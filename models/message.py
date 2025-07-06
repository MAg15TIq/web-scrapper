"""
Message protocol for inter-agent communication.
"""
import uuid
import time
import json
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of messages that can be exchanged between agents."""
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    CONTROL = "control"
    STATUS = "status"


class Priority(int, Enum):
    """Priority levels for messages."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class Message(BaseModel):
    """
    Base message class for inter-agent communication.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    sender_id: str
    recipient_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    priority: Priority = Priority.NORMAL
    payload: Dict[str, Any] = {}
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class TaskMessage(Message):
    """
    Message containing a task to be executed by an agent.
    """
    type: MessageType = MessageType.TASK
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    parameters: Dict[str, Any] = {}
    dependencies: List[str] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "sender_id": "coordinator",
                "recipient_id": "scraper-1",
                "task_type": "fetch_url",
                "parameters": {
                    "url": "https://example.com",
                    "headers": {"User-Agent": "Mozilla/5.0"},
                    "timeout": 30
                }
            }
        }


class ResultMessage(Message):
    """
    Message containing the result of a task.
    """
    type: MessageType = MessageType.RESULT
    task_id: str
    result: Dict[str, Any] = {}
    execution_time: float = 0.0
    
    class Config:
        json_schema_extra = {
            "example": {
                "sender_id": "scraper-1",
                "recipient_id": "coordinator",
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "result": {
                    "status_code": 200,
                    "content": "<!DOCTYPE html><html>...</html>",
                    "headers": {"Content-Type": "text/html"}
                },
                "execution_time": 1.25
            }
        }


class ErrorMessage(Message):
    """
    Message indicating an error occurred during task execution.
    """
    type: MessageType = MessageType.ERROR
    task_id: Optional[str] = None
    error_type: str
    error_message: str
    traceback: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "sender_id": "scraper-1",
                "recipient_id": "coordinator",
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "error_type": "ConnectionError",
                "error_message": "Failed to connect to example.com",
                "traceback": "Traceback (most recent call last):..."
            }
        }


class ControlMessage(Message):
    """
    Message for controlling agent behavior.
    """
    type: MessageType = MessageType.CONTROL
    command: str
    arguments: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "sender_id": "coordinator",
                "recipient_id": "scraper-1",
                "command": "pause",
                "arguments": {"duration": 60}
            }
        }


class StatusMessage(Message):
    """
    Message for reporting agent status.
    """
    type: MessageType = MessageType.STATUS
    status: str
    details: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "sender_id": "scraper-1",
                "recipient_id": "coordinator",
                "status": "idle",
                "details": {
                    "tasks_completed": 15,
                    "memory_usage": "45MB",
                    "uptime": 3600
                }
            }
        }
