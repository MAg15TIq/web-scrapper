"""
Monitoring API Routes
Handles system monitoring, metrics, and health check endpoints.
"""

import logging
import psutil
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from web.api.dependencies import (
    get_agent_monitor, get_current_active_user, get_config
)
from web.dashboard.agent_monitor import AgentMonitor
from config.web_config import WebConfig


# Configure logging
logger = logging.getLogger("monitoring_api")

# Create router
router = APIRouter()


# Pydantic models
class SystemMetrics(BaseModel):
    """System metrics model."""
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    disk_usage: float = Field(..., description="Disk usage percentage")
    network_io: Dict[str, int] = Field(..., description="Network I/O statistics")
    process_count: int = Field(..., description="Number of running processes")
    uptime: float = Field(..., description="System uptime in seconds")
    timestamp: datetime = Field(..., description="Metrics timestamp")


class AgentMetrics(BaseModel):
    """Agent metrics model."""
    agent_id: str
    agent_type: str
    status: str
    cpu_usage: float
    memory_usage: float
    tasks_completed: int
    tasks_failed: int
    average_response_time: float
    last_activity: datetime


class ApplicationMetrics(BaseModel):
    """Application metrics model."""
    total_requests: int
    active_connections: int
    response_times: Dict[str, float]
    error_rates: Dict[str, float]
    cache_hit_rate: float
    database_connections: int
    queue_sizes: Dict[str, int]


class HealthStatus(BaseModel):
    """Health status model."""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    services: Dict[str, bool] = Field(..., description="Service health status")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Application uptime in seconds")


class AlertRule(BaseModel):
    """Alert rule model."""
    id: str
    name: str
    description: str
    metric: str
    threshold: float
    operator: str  # gt, lt, eq, gte, lte
    enabled: bool
    created_at: datetime


class Alert(BaseModel):
    """Alert model."""
    id: str
    rule_id: str
    message: str
    severity: str  # low, medium, high, critical
    triggered_at: datetime
    resolved_at: Optional[datetime]
    status: str  # active, resolved, acknowledged


@router.get("/health", response_model=HealthStatus)
async def get_health_status(
    config: WebConfig = Depends(get_config)
):
    """
    Get system health status.
    
    Returns overall system health including service status,
    version information, and uptime.
    """
    try:
        logger.debug("Getting health status")
        
        # Check service health
        services = {
            "database": True,  # Add actual database health check
            "redis": True,     # Add actual Redis health check
            "agents": True,    # Add actual agent system health check
            "scheduler": True  # Add actual scheduler health check
        }
        
        # Determine overall status
        overall_status = "healthy" if all(services.values()) else "unhealthy"
        
        # Calculate uptime (placeholder)
        uptime = 3600.0  # 1 hour placeholder
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.now(),
            services=services,
            version=config.version,
            uptime=uptime
        )
        
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve health status"
        )


@router.get("/metrics/system", response_model=SystemMetrics)
async def get_system_metrics(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get system-level metrics.
    
    Returns CPU, memory, disk usage, and other system metrics.
    """
    try:
        logger.debug("Getting system metrics")
        
        # Get CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # Get network I/O
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }
        
        # Get process count
        process_count = len(psutil.pids())
        
        # Get uptime
        boot_time = psutil.boot_time()
        uptime = datetime.now().timestamp() - boot_time
        
        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            process_count=process_count,
            uptime=uptime,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system metrics"
        )


@router.get("/metrics/agents", response_model=List[AgentMetrics])
async def get_agent_metrics(
    agent_monitor: AgentMonitor = Depends(get_agent_monitor),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get metrics for all agents.
    
    Returns performance metrics for each agent including
    CPU usage, memory usage, and task statistics.
    """
    try:
        logger.debug("Getting agent metrics")
        
        # Get agent metrics from monitor
        agent_metrics_data = await agent_monitor.get_agent_metrics()
        
        agent_metrics = []
        for agent_id, metrics in agent_metrics_data.items():
            agent_metric = AgentMetrics(
                agent_id=agent_id,
                agent_type=metrics.get("agent_type", "unknown"),
                status=metrics.get("status", "unknown"),
                cpu_usage=metrics.get("cpu_usage", 0.0),
                memory_usage=metrics.get("memory_usage", 0.0),
                tasks_completed=metrics.get("tasks_completed", 0),
                tasks_failed=metrics.get("tasks_failed", 0),
                average_response_time=metrics.get("average_response_time", 0.0),
                last_activity=metrics.get("last_activity", datetime.now())
            )
            agent_metrics.append(agent_metric)
        
        return agent_metrics
        
    except Exception as e:
        logger.error(f"Error getting agent metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agent metrics"
        )


@router.get("/metrics/application", response_model=ApplicationMetrics)
async def get_application_metrics(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get application-level metrics.
    
    Returns metrics specific to the web application including
    request counts, response times, and error rates.
    """
    try:
        logger.debug("Getting application metrics")
        
        # In a real implementation, these would come from actual metrics collection
        # For now, return mock data
        return ApplicationMetrics(
            total_requests=12500,
            active_connections=45,
            response_times={
                "avg": 125.5,
                "p50": 98.2,
                "p95": 245.8,
                "p99": 456.3
            },
            error_rates={
                "4xx": 2.1,
                "5xx": 0.3
            },
            cache_hit_rate=87.5,
            database_connections=8,
            queue_sizes={
                "job_queue": 23,
                "notification_queue": 5,
                "cleanup_queue": 2
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting application metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve application metrics"
        )


@router.get("/alerts", response_model=List[Alert])
async def get_alerts(
    status_filter: Optional[str] = Query(None, description="Filter alerts by status"),
    severity_filter: Optional[str] = Query(None, description="Filter alerts by severity"),
    limit: int = Query(default=50, description="Maximum number of alerts to return"),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get system alerts.
    
    Returns a list of system alerts with optional filtering
    by status and severity.
    """
    try:
        logger.debug("Getting alerts")
        
        # In a real implementation, this would query the alerts database
        # For now, return mock alerts
        mock_alerts = [
            Alert(
                id="alert_001",
                rule_id="rule_cpu_high",
                message="CPU usage exceeded 80% threshold",
                severity="high",
                triggered_at=datetime.now() - timedelta(minutes=15),
                resolved_at=None,
                status="active"
            ),
            Alert(
                id="alert_002",
                rule_id="rule_memory_high",
                message="Memory usage exceeded 85% threshold",
                severity="medium",
                triggered_at=datetime.now() - timedelta(hours=2),
                resolved_at=datetime.now() - timedelta(hours=1),
                status="resolved"
            )
        ]
        
        # Apply filters
        filtered_alerts = mock_alerts
        if status_filter:
            filtered_alerts = [a for a in filtered_alerts if a.status == status_filter]
        if severity_filter:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity_filter]
        
        # Apply limit
        return filtered_alerts[:limit]
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve alerts"
        )


@router.get("/logs")
async def get_logs(
    level: Optional[str] = Query(None, description="Filter logs by level"),
    component: Optional[str] = Query(None, description="Filter logs by component"),
    limit: int = Query(default=100, description="Maximum number of log entries"),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get system logs.
    
    Returns recent system logs with optional filtering
    by log level and component.
    """
    try:
        logger.debug("Getting logs")
        
        # In a real implementation, this would read from log files or log database
        # For now, return mock log entries
        mock_logs = [
            {
                "timestamp": datetime.now() - timedelta(minutes=1),
                "level": "INFO",
                "component": "agent_manager",
                "message": "Agent scraper_001 completed task successfully",
                "details": {"task_id": "task_123", "duration": 45.2}
            },
            {
                "timestamp": datetime.now() - timedelta(minutes=5),
                "level": "WARNING",
                "component": "job_scheduler",
                "message": "Job queue size approaching limit",
                "details": {"queue_size": 95, "limit": 100}
            },
            {
                "timestamp": datetime.now() - timedelta(minutes=10),
                "level": "ERROR",
                "component": "web_scraper",
                "message": "Failed to scrape URL due to timeout",
                "details": {"url": "https://example.com", "timeout": 30}
            }
        ]
        
        # Apply filters
        filtered_logs = mock_logs
        if level:
            filtered_logs = [log for log in filtered_logs if log["level"] == level.upper()]
        if component:
            filtered_logs = [log for log in filtered_logs if log["component"] == component]
        
        # Apply limit and return
        return {
            "logs": filtered_logs[:limit],
            "total_count": len(filtered_logs),
            "filters_applied": {
                "level": level,
                "component": component,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve logs"
        )


@router.get("/performance")
async def get_performance_metrics(
    timeframe: str = Query(default="1h", description="Timeframe for metrics (1h, 6h, 24h, 7d)"),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get performance metrics over time.
    
    Returns time-series performance data for the specified timeframe.
    """
    try:
        logger.debug(f"Getting performance metrics for timeframe: {timeframe}")
        
        # Parse timeframe
        timeframe_mapping = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(days=1),
            "7d": timedelta(days=7)
        }
        
        if timeframe not in timeframe_mapping:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid timeframe. Use: 1h, 6h, 24h, or 7d"
            )
        
        # In a real implementation, this would query time-series data
        # For now, return mock performance data
        end_time = datetime.now()
        start_time = end_time - timeframe_mapping[timeframe]
        
        # Generate mock time series data
        data_points = []
        current_time = start_time
        interval = timeframe_mapping[timeframe] / 20  # 20 data points
        
        while current_time <= end_time:
            data_points.append({
                "timestamp": current_time,
                "cpu_usage": 45.2 + (current_time.minute % 10) * 2,
                "memory_usage": 62.8 + (current_time.minute % 15) * 1.5,
                "response_time": 125.5 + (current_time.minute % 8) * 10,
                "throughput": 850 + (current_time.minute % 12) * 25
            })
            current_time += interval
        
        return {
            "timeframe": timeframe,
            "start_time": start_time,
            "end_time": end_time,
            "data_points": data_points,
            "summary": {
                "avg_cpu": sum(dp["cpu_usage"] for dp in data_points) / len(data_points),
                "avg_memory": sum(dp["memory_usage"] for dp in data_points) / len(data_points),
                "avg_response_time": sum(dp["response_time"] for dp in data_points) / len(data_points),
                "avg_throughput": sum(dp["throughput"] for dp in data_points) / len(data_points)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )
