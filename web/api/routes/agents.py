"""
Agents API Routes
Handles agent management, configuration, and status endpoints.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from pydantic import BaseModel, Field

from web.api.dependencies import (
    get_agent_manager, get_current_active_user, get_pagination_params,
    validate_agent_type, get_request_context
)
from cli.agent_communication import AgentCommunicationLayer


# Configure logging
logger = logging.getLogger("agents_api")

# Create router
router = APIRouter()


# Pydantic models for API
class AgentStatusResponse(BaseModel):
    """Agent status response model."""
    agent_id: str
    agent_type: str
    status: str
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    last_activity: datetime
    capabilities: List[str]
    configuration: Dict[str, Any]


class AgentConfigurationRequest(BaseModel):
    """Agent configuration request model."""
    configuration: Dict[str, Any] = Field(..., description="Agent configuration parameters")
    restart_agent: bool = Field(default=False, description="Whether to restart the agent after configuration")


class AgentConfigurationResponse(BaseModel):
    """Agent configuration response model."""
    success: bool
    message: str
    agent_id: str
    configuration: Dict[str, Any]


class TaskSubmissionRequest(BaseModel):
    """Task submission request model."""
    task_type: str = Field(..., description="Type of task to submit")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    priority: str = Field(default="normal", description="Task priority")
    timeout: Optional[int] = Field(default=None, description="Task timeout in seconds")


class TaskSubmissionResponse(BaseModel):
    """Task submission response model."""
    task_id: str
    status: str
    message: str
    estimated_completion: Optional[datetime] = None


class AgentListResponse(BaseModel):
    """Agent list response model."""
    agents: List[AgentStatusResponse]
    total_count: int
    page: int
    size: int


class SystemMetricsResponse(BaseModel):
    """System metrics response model."""
    total_agents: int
    active_agents: int
    total_tasks: int
    messages_processed: int
    system_initialized: bool
    coordinator_available: bool
    timestamp: datetime


@router.get("/", response_model=AgentListResponse)
async def list_agents(
    pagination: Dict[str, int] = Depends(get_pagination_params),
    status_filter: Optional[str] = Query(None, description="Filter agents by status"),
    agent_manager: AgentCommunicationLayer = Depends(get_agent_manager),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    List all available agents with their current status.
    
    Returns a paginated list of agents with their status information,
    including active tasks, completion statistics, and configuration.
    """
    try:
        logger.info(f"Listing agents for user: {current_user.get('username')}")
        
        # Get agent statuses
        agent_statuses = await agent_manager.get_agent_status()
        
        # Convert to response format
        agents = []
        for agent_id, status_data in agent_statuses.items():
            # Filter by status if specified
            if status_filter and status_data.get('status') != status_filter:
                continue
            
            agent_response = AgentStatusResponse(
                agent_id=agent_id,
                agent_type=status_data.get('agent_type', 'unknown'),
                status=status_data.get('status', 'unknown'),
                active_tasks=status_data.get('active_tasks', 0),
                completed_tasks=status_data.get('completed_tasks', 0),
                failed_tasks=status_data.get('failed_tasks', 0),
                last_activity=status_data.get('last_activity', datetime.now()),
                capabilities=status_data.get('capabilities', []),
                configuration=status_data.get('configuration', {})
            )
            agents.append(agent_response)
        
        # Apply pagination
        total_count = len(agents)
        start_idx = pagination['offset']
        end_idx = start_idx + pagination['size']
        paginated_agents = agents[start_idx:end_idx]
        
        return AgentListResponse(
            agents=paginated_agents,
            total_count=total_count,
            page=pagination['page'],
            size=pagination['size']
        )
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agent list"
        )


@router.get("/{agent_id}", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_id: str = Path(..., description="Agent ID"),
    agent_manager: AgentCommunicationLayer = Depends(get_agent_manager),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get detailed status information for a specific agent.
    
    Returns comprehensive status information including current tasks,
    performance metrics, and configuration details.
    """
    try:
        logger.info(f"Getting status for agent: {agent_id}")
        
        # Get specific agent status
        status_data = await agent_manager.get_agent_status(agent_id)
        
        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {agent_id}"
            )
        
        return AgentStatusResponse(
            agent_id=agent_id,
            agent_type=status_data.get('agent_type', 'unknown'),
            status=status_data.get('status', 'unknown'),
            active_tasks=status_data.get('active_tasks', 0),
            completed_tasks=status_data.get('completed_tasks', 0),
            failed_tasks=status_data.get('failed_tasks', 0),
            last_activity=status_data.get('last_activity', datetime.now()),
            capabilities=status_data.get('capabilities', []),
            configuration=status_data.get('configuration', {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agent status"
        )


@router.put("/{agent_id}/configure", response_model=AgentConfigurationResponse)
async def configure_agent(
    agent_id: str = Path(..., description="Agent ID"),
    config_request: AgentConfigurationRequest = ...,
    agent_manager: AgentCommunicationLayer = Depends(get_agent_manager),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Configure an agent with new settings.
    
    Updates the agent's configuration with the provided parameters.
    Optionally restarts the agent to apply the new configuration.
    """
    try:
        logger.info(f"Configuring agent: {agent_id}")
        
        # Validate agent exists
        status_data = await agent_manager.get_agent_status(agent_id)
        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {agent_id}"
            )
        
        # Get agent type from status
        agent_type = status_data.get('agent_type')
        
        # Configure agent
        success = await agent_manager.configure_agent(agent_type, config_request.configuration)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to configure agent"
            )
        
        # Restart agent if requested
        if config_request.restart_agent:
            # In a real implementation, you would restart the agent here
            logger.info(f"Restart requested for agent: {agent_id}")
        
        return AgentConfigurationResponse(
            success=True,
            message="Agent configured successfully",
            agent_id=agent_id,
            configuration=config_request.configuration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to configure agent"
        )


@router.post("/{agent_id}/tasks", response_model=TaskSubmissionResponse)
async def submit_task_to_agent(
    agent_id: str = Path(..., description="Agent ID"),
    task_request: TaskSubmissionRequest = ...,
    agent_manager: AgentCommunicationLayer = Depends(get_agent_manager),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Submit a task to a specific agent.
    
    Creates and submits a new task to the specified agent with the
    provided parameters and priority settings.
    """
    try:
        logger.info(f"Submitting task to agent: {agent_id}")
        
        # Validate agent exists
        status_data = await agent_manager.get_agent_status(agent_id)
        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {agent_id}"
            )
        
        # Submit task
        task_id = await agent_manager.submit_task(
            task_type=task_request.task_type,
            parameters=task_request.parameters,
            target_agent=agent_id
        )
        
        if not task_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to submit task"
            )
        
        return TaskSubmissionResponse(
            task_id=task_id,
            status="submitted",
            message="Task submitted successfully",
            estimated_completion=None  # Could be calculated based on agent load
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit task"
        )


@router.get("/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    agent_manager: AgentCommunicationLayer = Depends(get_agent_manager),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get system-wide metrics for all agents.
    
    Returns comprehensive metrics including agent counts, task statistics,
    and system health information.
    """
    try:
        logger.info("Getting system metrics")
        
        # Get system metrics
        metrics = agent_manager.get_system_metrics()
        
        return SystemMetricsResponse(
            total_agents=metrics.get('total_agents', 0),
            active_agents=metrics.get('active_agents', 0),
            total_tasks=metrics.get('total_tasks', 0),
            messages_processed=metrics.get('messages_processed', 0),
            system_initialized=metrics.get('system_initialized', False),
            coordinator_available=metrics.get('coordinator_available', False),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system metrics"
        )


@router.post("/{agent_id}/restart")
async def restart_agent(
    agent_id: str = Path(..., description="Agent ID"),
    agent_manager: AgentCommunicationLayer = Depends(get_agent_manager),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Restart a specific agent.
    
    Safely restarts the specified agent, preserving its configuration
    and resuming any pending tasks.
    """
    try:
        logger.info(f"Restarting agent: {agent_id}")
        
        # Validate agent exists
        status_data = await agent_manager.get_agent_status(agent_id)
        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {agent_id}"
            )
        
        # Get agent type
        agent_type = status_data.get('agent_type')
        
        # Send restart command
        from cli.agent_communication import AgentMessage, MessageType
        import uuid
        
        restart_message = AgentMessage(
            id=str(uuid.uuid4()),
            type=MessageType.COMMAND,
            sender="api",
            recipient=agent_id,
            payload={
                "command": "restart_agent",
                "args": {"agent_type": agent_type}
            },
            timestamp=datetime.now()
        )
        
        response = await agent_manager.send_message(restart_message)
        
        if response != "success":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to restart agent"
            )
        
        return {
            "success": True,
            "message": f"Agent {agent_id} restarted successfully",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restarting agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to restart agent"
        )
