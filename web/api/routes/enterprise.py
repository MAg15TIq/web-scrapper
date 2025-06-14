"""
Enterprise API endpoints for Phase 3: Enterprise & Scalability Features.

This module provides REST API endpoints for managing the enterprise-grade
distributed scraping system with cluster management, multi-tenancy,
horizontal scaling, and advanced monitoring.
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from services.enterprise_orchestrator import EnterpriseOrchestrator, EnterpriseConfig
from services.multi_tenant_manager import BillingModel, ResourceQuota
from services.horizontal_scaling_manager import LoadBalancingAlgorithm, ScalingDirection
from services.enhanced_health_monitor import FailoverStrategy
from services.advanced_queue_manager import QueueType, ResourceRequirement


# Pydantic models for API requests/responses

class TenantCreateRequest(BaseModel):
    name: str = Field(..., description="Tenant name")
    email: str = Field(..., description="Tenant email")
    billing_model: BillingModel = Field(BillingModel.FREEMIUM, description="Billing model")
    custom_quota: Optional[Dict[str, Any]] = Field(None, description="Custom resource quota")


class JobSubmissionRequest(BaseModel):
    task_data: Dict[str, Any] = Field(..., description="Task data")
    tenant_id: str = Field("default", description="Tenant ID")
    priority: int = Field(1, description="Job priority (1-10)")
    partition_id: str = Field("default", description="Queue partition ID")
    resource_requirements: Optional[Dict[str, Any]] = Field(None, description="Resource requirements")


class ScalingRequest(BaseModel):
    target_nodes: int = Field(..., description="Target number of nodes")
    force: bool = Field(False, description="Force scaling action")


class LoadBalancerConfigRequest(BaseModel):
    algorithm: LoadBalancingAlgorithm = Field(..., description="Load balancing algorithm")
    health_check_interval: Optional[int] = Field(None, description="Health check interval in seconds")
    sticky_sessions: Optional[bool] = Field(None, description="Enable sticky sessions")


class QueuePartitionRequest(BaseModel):
    partition_id: str = Field(..., description="Partition ID")
    name: str = Field(..., description="Partition name")
    queue_type: QueueType = Field(QueueType.PRIORITY, description="Queue type")
    max_size: int = Field(10000, description="Maximum queue size")
    priority_weight: float = Field(1.0, description="Priority weight")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for isolation")


# Initialize router
router = APIRouter(prefix="/enterprise", tags=["enterprise"])

# Global orchestrator instance (would be injected in production)
orchestrator: Optional[EnterpriseOrchestrator] = None


def get_orchestrator() -> EnterpriseOrchestrator:
    """Dependency to get the enterprise orchestrator."""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Enterprise orchestrator not initialized")
    return orchestrator


@router.post("/initialize")
async def initialize_enterprise_system(
    config: Dict[str, Any] = Body(..., description="Enterprise configuration")
):
    """Initialize the enterprise orchestrator system."""
    global orchestrator
    
    try:
        # Create enterprise config
        enterprise_config = EnterpriseConfig(
            cluster_id=config.get("cluster_id", "default-cluster"),
            cluster_name=config.get("cluster_name", "Web Scraper Cluster"),
            max_nodes=config.get("max_nodes", 50),
            auto_scaling=config.get("auto_scaling", True),
            multi_tenant_enabled=config.get("multi_tenant_enabled", True),
            scaling_enabled=config.get("scaling_enabled", True),
            health_monitoring_enabled=config.get("health_monitoring_enabled", True)
        )
        
        # Initialize orchestrator
        orchestrator = EnterpriseOrchestrator(enterprise_config)
        
        # Start the system
        success = await orchestrator.start()
        
        if success:
            return {
                "status": "success",
                "message": "Enterprise system initialized successfully",
                "cluster_id": enterprise_config.cluster_id,
                "startup_time": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start enterprise system")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@router.get("/status")
async def get_system_status(orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)):
    """Get comprehensive system status."""
    try:
        status = orchestrator.get_system_status()
        return JSONResponse(content=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@router.get("/metrics")
async def get_performance_metrics(orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)):
    """Get detailed performance metrics."""
    try:
        metrics = orchestrator.get_performance_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


# Cluster Management Endpoints

@router.get("/cluster/nodes")
async def get_cluster_nodes(orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)):
    """Get all nodes in the cluster."""
    try:
        nodes = orchestrator.cluster_manager.get_cluster_nodes()
        return {
            "nodes": {
                node_id: {
                    "node_id": node.node_id,
                    "hostname": node.hostname,
                    "ip_address": node.ip_address,
                    "port": node.port,
                    "role": node.role.value,
                    "status": node.status.value,
                    "capabilities": node.capabilities,
                    "resources": node.resources,
                    "last_heartbeat": node.last_heartbeat.isoformat(),
                    "joined_at": node.joined_at.isoformat()
                }
                for node_id, node in nodes.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cluster nodes: {str(e)}")


@router.get("/cluster/stats")
async def get_cluster_stats(orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)):
    """Get cluster statistics."""
    try:
        stats = orchestrator.cluster_manager.get_cluster_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cluster stats: {str(e)}")


# Multi-Tenant Management Endpoints

@router.post("/tenants")
async def create_tenant(
    request: TenantCreateRequest,
    orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)
):
    """Create a new tenant."""
    try:
        # Convert custom quota if provided
        custom_quota = None
        if request.custom_quota:
            custom_quota = ResourceQuota(**request.custom_quota)
        
        tenant_id = await orchestrator.create_tenant(
            name=request.name,
            email=request.email,
            billing_model=request.billing_model,
            custom_quota=custom_quota
        )
        
        if tenant_id:
            return {
                "status": "success",
                "tenant_id": tenant_id,
                "message": f"Tenant '{request.name}' created successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create tenant")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create tenant: {str(e)}")


@router.get("/tenants")
async def get_all_tenants(orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)):
    """Get all tenants."""
    try:
        if not orchestrator.config.multi_tenant_enabled:
            raise HTTPException(status_code=400, detail="Multi-tenancy is not enabled")
        
        tenants = orchestrator.multi_tenant_manager.get_all_tenants()
        return {
            "tenants": {
                tenant_id: {
                    "tenant_id": tenant.tenant_id,
                    "name": tenant.name,
                    "email": tenant.email,
                    "status": tenant.status.value,
                    "billing_model": tenant.billing_model.value,
                    "created_at": tenant.created_at.isoformat(),
                    "expires_at": tenant.expires_at.isoformat() if tenant.expires_at else None
                }
                for tenant_id, tenant in tenants.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tenants: {str(e)}")


@router.get("/tenants/{tenant_id}/usage")
async def get_tenant_usage(
    tenant_id: str,
    orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)
):
    """Get tenant usage metrics."""
    try:
        if not orchestrator.config.multi_tenant_enabled:
            raise HTTPException(status_code=400, detail="Multi-tenancy is not enabled")
        
        usage = orchestrator.multi_tenant_manager.get_tenant_usage(tenant_id)
        if not usage:
            raise HTTPException(status_code=404, detail="Tenant not found")
        
        return {
            "tenant_id": usage.tenant_id,
            "period_start": usage.period_start.isoformat(),
            "period_end": usage.period_end.isoformat(),
            "total_jobs": usage.total_jobs,
            "successful_jobs": usage.successful_jobs,
            "failed_jobs": usage.failed_jobs,
            "total_requests": usage.total_requests,
            "cpu_hours_used": usage.cpu_hours_used,
            "memory_gb_hours_used": usage.memory_gb_hours_used,
            "storage_gb_used": usage.storage_gb_used,
            "bandwidth_gb_used": usage.bandwidth_gb_used,
            "api_calls": usage.api_calls,
            "total_cost": usage.total_cost
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tenant usage: {str(e)}")


# Job Management Endpoints

@router.post("/jobs/submit")
async def submit_job(
    request: JobSubmissionRequest,
    orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)
):
    """Submit a job to the enterprise system."""
    try:
        # Convert resource requirements if provided
        resource_requirements = None
        if request.resource_requirements:
            resource_requirements = ResourceRequirement(**request.resource_requirements)
        
        job_id = await orchestrator.submit_job(
            task_data=request.task_data,
            tenant_id=request.tenant_id,
            priority=request.priority,
            resource_requirements=resource_requirements,
            partition_id=request.partition_id
        )
        
        if job_id:
            return {
                "status": "success",
                "job_id": job_id,
                "message": "Job submitted successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to submit job")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@router.get("/queues/stats")
async def get_queue_statistics(orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)):
    """Get queue statistics."""
    try:
        stats = orchestrator.queue_manager.get_queue_statistics()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue stats: {str(e)}")


@router.post("/queues/partitions")
async def create_queue_partition(
    request: QueuePartitionRequest,
    orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)
):
    """Create a new queue partition."""
    try:
        success = orchestrator.queue_manager.create_partition(
            partition_id=request.partition_id,
            name=request.name,
            queue_type=request.queue_type,
            max_size=request.max_size,
            priority_weight=request.priority_weight,
            tenant_id=request.tenant_id
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Queue partition '{request.partition_id}' created successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create queue partition")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create queue partition: {str(e)}")


# Scaling Management Endpoints

@router.post("/scaling/manual")
async def manual_scaling(
    request: ScalingRequest,
    orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)
):
    """Manually scale the cluster."""
    try:
        if not orchestrator.config.scaling_enabled:
            raise HTTPException(status_code=400, detail="Scaling is not enabled")
        
        success = await orchestrator.scale_cluster(request.target_nodes)
        
        if success:
            return {
                "status": "success",
                "message": f"Cluster scaling to {request.target_nodes} nodes initiated"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to initiate scaling")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scale cluster: {str(e)}")


@router.get("/scaling/recommendations")
async def get_scaling_recommendations(orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)):
    """Get scaling recommendations."""
    try:
        if not orchestrator.config.scaling_enabled:
            raise HTTPException(status_code=400, detail="Scaling is not enabled")
        
        recommendations = orchestrator.scaling_manager.get_scaling_recommendations()
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scaling recommendations: {str(e)}")


@router.put("/scaling/load-balancer")
async def update_load_balancer_config(
    request: LoadBalancerConfigRequest,
    orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)
):
    """Update load balancer configuration."""
    try:
        if not orchestrator.config.scaling_enabled:
            raise HTTPException(status_code=400, detail="Scaling is not enabled")
        
        config_updates = {"algorithm": request.algorithm.value}
        if request.health_check_interval is not None:
            config_updates["health_check_interval"] = request.health_check_interval
        if request.sticky_sessions is not None:
            config_updates["sticky_sessions"] = request.sticky_sessions
        
        orchestrator.scaling_manager.update_load_balancer_config(config_updates)
        
        return {
            "status": "success",
            "message": "Load balancer configuration updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update load balancer config: {str(e)}")


# Health Monitoring Endpoints

@router.get("/health/nodes")
async def get_node_health_scores(orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)):
    """Get health scores for all nodes."""
    try:
        if not orchestrator.config.health_monitoring_enabled:
            raise HTTPException(status_code=400, detail="Health monitoring is not enabled")
        
        health_scores = orchestrator.health_monitor.get_all_health_scores()
        return {
            "health_scores": {
                node_id: {
                    "node_id": score.node_id,
                    "overall_score": score.overall_score,
                    "availability_score": score.availability_score,
                    "performance_score": score.performance_score,
                    "resource_score": score.resource_score,
                    "health_trend": score.health_trend,
                    "last_updated": score.last_updated.isoformat(),
                    "uptime_percentage": score.uptime_percentage,
                    "average_response_time": score.average_response_time,
                    "error_rate": score.error_rate,
                    "cpu_usage": score.cpu_usage,
                    "memory_usage": score.memory_usage,
                    "disk_usage": score.disk_usage
                }
                for node_id, score in health_scores.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health scores: {str(e)}")


@router.get("/health/failovers")
async def get_failover_history(
    hours: int = Query(24, description="Hours of history to retrieve"),
    orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)
):
    """Get failover history."""
    try:
        if not orchestrator.config.health_monitoring_enabled:
            raise HTTPException(status_code=400, detail="Health monitoring is not enabled")
        
        failover_history = orchestrator.health_monitor.get_failover_history(hours)
        return {
            "failover_events": [
                {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "failed_node_id": event.failed_node_id,
                    "replacement_node_id": event.replacement_node_id,
                    "strategy": event.strategy.value,
                    "trigger_reason": event.trigger_reason,
                    "success": event.success,
                    "recovery_time": event.recovery_time,
                    "affected_tasks": event.affected_tasks
                }
                for event in failover_history
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get failover history: {str(e)}")


@router.post("/health/failover/{node_id}")
async def manual_failover(
    node_id: str,
    reason: str = Body("Manual failover", description="Reason for failover"),
    orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)
):
    """Manually trigger failover for a node."""
    try:
        if not orchestrator.config.health_monitoring_enabled:
            raise HTTPException(status_code=400, detail="Health monitoring is not enabled")
        
        success = await orchestrator.health_monitor.manual_failover(node_id, reason)
        
        if success:
            return {
                "status": "success",
                "message": f"Failover initiated for node {node_id}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to initiate failover")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger failover: {str(e)}")


@router.delete("/shutdown")
async def shutdown_enterprise_system(orchestrator: EnterpriseOrchestrator = Depends(get_orchestrator)):
    """Shutdown the enterprise system gracefully."""
    global orchestrator as global_orchestrator
    
    try:
        success = await orchestrator.stop()
        
        if success:
            global_orchestrator = None
            return {
                "status": "success",
                "message": "Enterprise system shutdown successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to shutdown system")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to shutdown system: {str(e)}")
