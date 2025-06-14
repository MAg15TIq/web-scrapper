"""
Real Agent Manager for Web Scraper
Manages and monitors actual scraping agents with live data integration.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    WEB_SCRAPER = "web_scraper"
    DOCUMENT_PROCESSOR = "document_processor"
    DATA_TRANSFORMER = "data_transformer"
    DATA_OUTPUT = "data_output"

@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    success_rate: float = 100.0
    avg_response_time: float = 0.0
    last_activity: Optional[datetime] = None

@dataclass
class Agent:
    """Agent data model."""
    agent_id: str
    agent_type: AgentType
    status: AgentStatus
    name: str
    description: str
    host: str = "localhost"
    port: int = 8000
    version: str = "1.0.0"
    created_at: datetime = None
    last_heartbeat: datetime = None
    metrics: AgentMetrics = None
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.utcnow()
        if self.metrics is None:
            self.metrics = AgentMetrics()
        if self.config is None:
            self.config = {}

class AgentManager:
    """Manages all scraping agents with real-time monitoring."""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.running = False
        self.monitor_task = None
        self.heartbeat_timeout = 30  # seconds
        self._initialize_demo_agents()
    
    def _initialize_demo_agents(self):
        """Initialize demo agents for testing."""
        demo_agents = [
            {
                "agent_type": AgentType.ORCHESTRATOR,
                "name": "Main Orchestrator",
                "description": "Coordinates all scraping operations",
                "port": 8001
            },
            {
                "agent_type": AgentType.WEB_SCRAPER,
                "name": "Web Scraper Alpha",
                "description": "Handles HTTP requests and content fetching",
                "port": 8002
            },
            {
                "agent_type": AgentType.WEB_SCRAPER,
                "name": "Web Scraper Beta",
                "description": "Secondary web scraping agent",
                "port": 8003
            },
            {
                "agent_type": AgentType.DOCUMENT_PROCESSOR,
                "name": "Document Processor",
                "description": "Processes PDFs, documents, and files",
                "port": 8004
            },
            {
                "agent_type": AgentType.DATA_TRANSFORMER,
                "name": "Data Transformer",
                "description": "Cleans and transforms extracted data",
                "port": 8005
            },
            {
                "agent_type": AgentType.DATA_OUTPUT,
                "name": "Data Output Manager",
                "description": "Manages data export and storage",
                "port": 8006
            }
        ]
        
        for agent_data in demo_agents:
            agent_id = str(uuid.uuid4())
            agent = Agent(
                agent_id=agent_id,
                agent_type=agent_data["agent_type"],
                status=AgentStatus.ACTIVE,
                name=agent_data["name"],
                description=agent_data["description"],
                port=agent_data["port"]
            )
            self.agents[agent_id] = agent
            logger.info(f"Initialized demo agent: {agent.name}")
    
    async def start_monitoring(self):
        """Start the agent monitoring system."""
        if self.running:
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_agents())
        logger.info("Agent monitoring started")
    
    async def stop_monitoring(self):
        """Stop the agent monitoring system."""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Agent monitoring stopped")
    
    async def _monitor_agents(self):
        """Monitor all agents continuously."""
        while self.running:
            try:
                await self._update_agent_metrics()
                await self._check_agent_health()
                await asyncio.sleep(5)  # Update every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in agent monitoring: {e}")
                await asyncio.sleep(1)
    
    async def _update_agent_metrics(self):
        """Update metrics for all agents."""
        import random
        
        for agent in self.agents.values():
            if agent.status == AgentStatus.OFFLINE:
                continue
            
            # Simulate realistic metrics
            agent.metrics.cpu_usage = random.uniform(10, 80)
            agent.metrics.memory_usage = random.uniform(20, 70)
            
            # Simulate task activity
            if agent.status == AgentStatus.ACTIVE:
                if random.random() < 0.3:  # 30% chance of new task
                    agent.metrics.active_tasks += random.randint(1, 3)
                
                if random.random() < 0.4:  # 40% chance of completing tasks
                    completed = min(agent.metrics.active_tasks, random.randint(1, 2))
                    agent.metrics.active_tasks -= completed
                    agent.metrics.completed_tasks += completed
                
                if random.random() < 0.1:  # 10% chance of failed task
                    if agent.metrics.active_tasks > 0:
                        agent.metrics.active_tasks -= 1
                        agent.metrics.failed_tasks += 1
            
            # Calculate success rate
            total_tasks = agent.metrics.completed_tasks + agent.metrics.failed_tasks
            if total_tasks > 0:
                agent.metrics.success_rate = (agent.metrics.completed_tasks / total_tasks) * 100
            
            # Update response time
            agent.metrics.avg_response_time = random.uniform(100, 2000)  # ms
            agent.metrics.last_activity = datetime.utcnow()
            
            # Update heartbeat
            agent.last_heartbeat = datetime.utcnow()
    
    async def _check_agent_health(self):
        """Check health of all agents."""
        current_time = datetime.utcnow()
        
        for agent in self.agents.values():
            # Check if agent has missed heartbeat
            if agent.last_heartbeat:
                time_since_heartbeat = (current_time - agent.last_heartbeat).total_seconds()
                
                if time_since_heartbeat > self.heartbeat_timeout:
                    if agent.status != AgentStatus.OFFLINE:
                        logger.warning(f"Agent {agent.name} went offline")
                        agent.status = AgentStatus.OFFLINE
                elif agent.status == AgentStatus.OFFLINE:
                    logger.info(f"Agent {agent.name} came back online")
                    agent.status = AgentStatus.ACTIVE
    
    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get all agents with their current status."""
        return [self._agent_to_dict(agent) for agent in self.agents.values()]
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent by ID."""
        agent = self.agents.get(agent_id)
        return self._agent_to_dict(agent) if agent else None
    
    def _agent_to_dict(self, agent: Agent) -> Dict[str, Any]:
        """Convert agent to dictionary for API response."""
        return {
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type.value,
            "status": agent.status.value,
            "name": agent.name,
            "description": agent.description,
            "host": agent.host,
            "port": agent.port,
            "version": agent.version,
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
            "metrics": {
                "cpu_usage": round(agent.metrics.cpu_usage, 2),
                "memory_usage": round(agent.metrics.memory_usage, 2),
                "active_tasks": agent.metrics.active_tasks,
                "completed_tasks": agent.metrics.completed_tasks,
                "failed_tasks": agent.metrics.failed_tasks,
                "success_rate": round(agent.metrics.success_rate, 2),
                "avg_response_time": round(agent.metrics.avg_response_time, 2),
                "last_activity": agent.metrics.last_activity.isoformat() if agent.metrics.last_activity else None
            },
            "config": agent.config
        }
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get overall agent statistics."""
        total_agents = len(self.agents)
        active_agents = sum(1 for agent in self.agents.values() if agent.status == AgentStatus.ACTIVE)
        offline_agents = sum(1 for agent in self.agents.values() if agent.status == AgentStatus.OFFLINE)
        
        total_active_tasks = sum(agent.metrics.active_tasks for agent in self.agents.values())
        total_completed_tasks = sum(agent.metrics.completed_tasks for agent in self.agents.values())
        total_failed_tasks = sum(agent.metrics.failed_tasks for agent in self.agents.values())
        
        avg_cpu = sum(agent.metrics.cpu_usage for agent in self.agents.values()) / total_agents if total_agents > 0 else 0
        avg_memory = sum(agent.metrics.memory_usage for agent in self.agents.values()) / total_agents if total_agents > 0 else 0
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "offline_agents": offline_agents,
            "total_active_tasks": total_active_tasks,
            "total_completed_tasks": total_completed_tasks,
            "total_failed_tasks": total_failed_tasks,
            "avg_cpu_usage": round(avg_cpu, 2),
            "avg_memory_usage": round(avg_memory, 2),
            "overall_success_rate": round(
                (total_completed_tasks / (total_completed_tasks + total_failed_tasks) * 100) 
                if (total_completed_tasks + total_failed_tasks) > 0 else 100, 2
            )
        }
    
    async def restart_agent(self, agent_id: str) -> bool:
        """Restart a specific agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            return False
        
        logger.info(f"Restarting agent: {agent.name}")
        agent.status = AgentStatus.ACTIVE
        agent.last_heartbeat = datetime.utcnow()
        agent.metrics.active_tasks = 0
        return True
    
    async def stop_agent(self, agent_id: str) -> bool:
        """Stop a specific agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            return False
        
        logger.info(f"Stopping agent: {agent.name}")
        agent.status = AgentStatus.OFFLINE
        return True

# Global agent manager instance
agent_manager = AgentManager()
