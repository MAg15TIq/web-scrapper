"""
Agent Communication Layer for Enhanced CLI
Handles communication between CLI and the multi-agent system.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

try:
    from agents.coordinator import CoordinatorAgent
    from agents.scraper import ScraperAgent
    from agents.parser import ParserAgent
    from agents.storage import StorageAgent
    from agents.javascript import JavaScriptAgent
    from agents.authentication import AuthenticationAgent
    from agents.anti_detection import AntiDetectionAgent
    from agents.data_transformation import DataTransformationAgent
    from agents.error_recovery import ErrorRecoveryAgent
    from agents.data_extractor import DataExtractorAgent
    from models.task import Task, TaskType, TaskStatus
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False


class MessageType(Enum):
    """Message types for agent communication."""
    COMMAND = "command"
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"
    TASK_SUBMIT = "task_submit"
    TASK_UPDATE = "task_update"
    TASK_COMPLETE = "task_complete"
    ERROR = "error"
    NOTIFICATION = "notification"


@dataclass
class AgentMessage:
    """Message structure for agent communication."""
    id: str
    type: MessageType
    sender: str
    recipient: str
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None


@dataclass
class AgentStatus:
    """Agent status information."""
    agent_id: str
    agent_type: str
    status: str
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    last_activity: datetime
    capabilities: List[str]
    configuration: Dict[str, Any]


class AgentCommunicationLayer:
    """Communication layer between CLI and agents."""
    
    def __init__(self):
        """Initialize the agent communication layer."""
        self.logger = logging.getLogger("agent_communication")
        
        # Agent registry
        self.agents: Dict[str, Any] = {}
        self.agent_statuses: Dict[str, AgentStatus] = {}
        
        # Message handling
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.message_queue: List[AgentMessage] = []
        self.response_callbacks: Dict[str, Callable] = {}
        
        # Communication state
        self.coordinator: Optional[Any] = None
        self.is_initialized = False
        
        # Initialize message handlers
        self._setup_message_handlers()
        
        self.logger.info(f"Agent communication layer initialized (Agents available: {AGENTS_AVAILABLE})")
    
    async def initialize_agents(self) -> bool:
        """Initialize the agent system."""
        if not AGENTS_AVAILABLE:
            self.logger.warning("Agent system not available")
            return False
        
        try:
            # Create coordinator
            self.coordinator = CoordinatorAgent()
            
            # Create and register agents
            agents_to_create = [
                ('scraper', ScraperAgent),
                ('parser', ParserAgent),
                ('storage', StorageAgent),
                ('javascript', JavaScriptAgent),
                ('authentication', AuthenticationAgent),
                ('anti_detection', AntiDetectionAgent),
                ('data_transformation', DataTransformationAgent),
                ('error_recovery', ErrorRecoveryAgent),
                ('data_extractor', DataExtractorAgent)
            ]
            
            for agent_name, agent_class in agents_to_create:
                try:
                    agent = agent_class(coordinator_id=self.coordinator.agent_id)
                    self.coordinator.register_agent(agent)
                    self.agents[agent_name] = agent
                    
                    # Create status entry
                    self.agent_statuses[agent.agent_id] = AgentStatus(
                        agent_id=agent.agent_id,
                        agent_type=agent_name,
                        status="idle",
                        active_tasks=0,
                        completed_tasks=0,
                        failed_tasks=0,
                        last_activity=datetime.now(),
                        capabilities=getattr(agent, 'capabilities', []),
                        configuration={}
                    )
                    
                    self.logger.debug(f"Initialized agent: {agent_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error creating agent {agent_name}: {e}")
            
            self.is_initialized = True
            self.logger.info(f"Agent system initialized with {len(self.agents)} agents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing agent system: {e}")
            return False
    
    def _setup_message_handlers(self):
        """Setup message handlers."""
        self.message_handlers = {
            MessageType.STATUS_REQUEST: self._handle_status_request,
            MessageType.TASK_SUBMIT: self._handle_task_submit,
            MessageType.COMMAND: self._handle_command,
            MessageType.ERROR: self._handle_error
        }
    
    async def send_message(self, message: AgentMessage) -> Optional[str]:
        """Send a message to an agent."""
        try:
            self.logger.debug(f"Sending message: {message.type} to {message.recipient}")
            
            # Add to message queue
            self.message_queue.append(message)
            
            # Handle message based on type
            handler = self.message_handlers.get(message.type)
            if handler:
                response = await handler(message)
                return response
            else:
                self.logger.warning(f"No handler for message type: {message.type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return None
    
    async def submit_task(self, task_type: str, parameters: Dict[str, Any], 
                         target_agent: Optional[str] = None) -> Optional[str]:
        """Submit a task to the agent system."""
        if not self.is_initialized:
            await self.initialize_agents()
        
        if not self.coordinator:
            self.logger.error("Coordinator not available")
            return None
        
        try:
            # Create task
            if AGENTS_AVAILABLE:
                # Map string task types to TaskType enum
                task_type_mapping = {
                    'scrape': TaskType.FETCH_URL,
                    'parse': TaskType.PARSE_CONTENT,
                    'store': TaskType.STORE_DATA,
                    'render_js': TaskType.RENDER_JS,
                    'extract': TaskType.EXTRACT_DATA,
                    'transform': TaskType.CLEAN_DATA,
                    'analyze': TaskType.ANALYZE_TEXT
                }
                
                task_enum = task_type_mapping.get(task_type, TaskType.FETCH_URL)
                task = Task(type=task_enum, parameters=parameters)
                
                # Submit task
                task_id = await self.coordinator.submit_task(task)
                
                self.logger.info(f"Task submitted: {task_id}")
                return task_id
            else:
                # Mock task submission
                task_id = str(uuid.uuid4())
                self.logger.info(f"Mock task submitted: {task_id}")
                return task_id
                
        except Exception as e:
            self.logger.error(f"Error submitting task: {e}")
            return None
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task."""
        if not self.coordinator:
            return None
        
        try:
            if AGENTS_AVAILABLE:
                status = self.coordinator.get_task_status(task_id)
                return status
            else:
                # Mock status
                return {
                    'task_id': task_id,
                    'status': 'completed',
                    'progress': 100,
                    'result': {'mock': 'data'}
                }
                
        except Exception as e:
            self.logger.error(f"Error getting task status: {e}")
            return None
    
    async def get_agent_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of agents."""
        try:
            if agent_id:
                # Get specific agent status
                if agent_id in self.agent_statuses:
                    return asdict(self.agent_statuses[agent_id])
                else:
                    return {}
            else:
                # Get all agent statuses
                return {aid: asdict(status) for aid, status in self.agent_statuses.items()}
                
        except Exception as e:
            self.logger.error(f"Error getting agent status: {e}")
            return {}
    
    async def configure_agent(self, agent_type: str, configuration: Dict[str, Any]) -> bool:
        """Configure an agent."""
        try:
            agent = self.agents.get(agent_type)
            if not agent:
                self.logger.error(f"Agent not found: {agent_type}")
                return False
            
            # Update agent configuration
            if hasattr(agent, 'configure'):
                await agent.configure(configuration)
            
            # Update status
            if agent.agent_id in self.agent_statuses:
                self.agent_statuses[agent.agent_id].configuration.update(configuration)
                self.agent_statuses[agent.agent_id].last_activity = datetime.now()
            
            self.logger.info(f"Agent configured: {agent_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring agent: {e}")
            return False
    
    async def _handle_status_request(self, message: AgentMessage) -> str:
        """Handle status request message."""
        try:
            agent_id = message.payload.get('agent_id')
            status = await self.get_agent_status(agent_id)
            
            response_message = AgentMessage(
                id=str(uuid.uuid4()),
                type=MessageType.STATUS_RESPONSE,
                sender="cli",
                recipient=message.sender,
                payload=status,
                timestamp=datetime.now(),
                correlation_id=message.id
            )
            
            return response_message.id
            
        except Exception as e:
            self.logger.error(f"Error handling status request: {e}")
            return ""
    
    async def _handle_task_submit(self, message: AgentMessage) -> str:
        """Handle task submission message."""
        try:
            task_type = message.payload.get('task_type')
            parameters = message.payload.get('parameters', {})
            target_agent = message.payload.get('target_agent')
            
            task_id = await self.submit_task(task_type, parameters, target_agent)
            
            return task_id or ""
            
        except Exception as e:
            self.logger.error(f"Error handling task submit: {e}")
            return ""
    
    async def _handle_command(self, message: AgentMessage) -> str:
        """Handle command message."""
        try:
            command = message.payload.get('command')
            args = message.payload.get('args', {})
            
            # Process command
            if command == 'start_agents':
                success = await self.initialize_agents()
                return "success" if success else "failed"
            elif command == 'stop_agents':
                return await self._stop_agents()
            elif command == 'restart_agent':
                agent_type = args.get('agent_type')
                return await self._restart_agent(agent_type)
            else:
                self.logger.warning(f"Unknown command: {command}")
                return "unknown_command"
                
        except Exception as e:
            self.logger.error(f"Error handling command: {e}")
            return "error"
    
    async def _handle_error(self, message: AgentMessage) -> str:
        """Handle error message."""
        try:
            error_info = message.payload
            self.logger.error(f"Agent error: {error_info}")
            
            # Update agent status if applicable
            agent_id = error_info.get('agent_id')
            if agent_id and agent_id in self.agent_statuses:
                self.agent_statuses[agent_id].failed_tasks += 1
                self.agent_statuses[agent_id].last_activity = datetime.now()
            
            return "acknowledged"
            
        except Exception as e:
            self.logger.error(f"Error handling error message: {e}")
            return "error"
    
    async def _stop_agents(self) -> str:
        """Stop all agents."""
        try:
            # Stop coordinator and agents
            if self.coordinator:
                # In a real implementation, you would properly shut down agents
                pass
            
            self.agents.clear()
            self.agent_statuses.clear()
            self.is_initialized = False
            
            self.logger.info("All agents stopped")
            return "success"
            
        except Exception as e:
            self.logger.error(f"Error stopping agents: {e}")
            return "error"
    
    async def _restart_agent(self, agent_type: str) -> str:
        """Restart a specific agent."""
        try:
            if agent_type not in self.agents:
                return "agent_not_found"
            
            # In a real implementation, you would restart the agent
            # For now, just update the status
            agent = self.agents[agent_type]
            if agent.agent_id in self.agent_statuses:
                self.agent_statuses[agent.agent_id].status = "restarted"
                self.agent_statuses[agent.agent_id].last_activity = datetime.now()
            
            self.logger.info(f"Agent restarted: {agent_type}")
            return "success"
            
        except Exception as e:
            self.logger.error(f"Error restarting agent: {e}")
            return "error"
    
    def get_message_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get message history."""
        return [asdict(msg) for msg in self.message_queue[-limit:]]
    
    def clear_message_history(self):
        """Clear message history."""
        self.message_queue.clear()
        self.logger.info("Message history cleared")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        total_agents = len(self.agents)
        active_agents = len([s for s in self.agent_statuses.values() if s.status in ['active', 'running']])
        total_tasks = sum(s.completed_tasks + s.failed_tasks for s in self.agent_statuses.values())
        
        return {
            'total_agents': total_agents,
            'active_agents': active_agents,
            'total_tasks': total_tasks,
            'messages_processed': len(self.message_queue),
            'system_initialized': self.is_initialized,
            'coordinator_available': self.coordinator is not None
        }
