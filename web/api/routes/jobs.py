"""
Jobs API Routes
Handles job creation, monitoring, and management endpoints.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from pydantic import BaseModel, Field, validator

from web.api.dependencies import (
    get_job_manager, get_current_active_user, get_pagination_params,
    validate_job_status, get_request_context
)
from web.scheduler.job_manager import JobManager


# Configure logging
logger = logging.getLogger("jobs_api")

# Create router
router = APIRouter()


class JobPriority(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class JobType(str, Enum):
    """Job types."""
    SCRAPE = "scrape"
    ANALYZE = "analyze"
    TRANSFORM = "transform"
    EXPORT = "export"
    MONITOR = "monitor"
    SCHEDULED = "scheduled"


class JobStatus(str, Enum):
    """Job status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


# Pydantic models
class JobCreateRequest(BaseModel):
    """Job creation request model."""
    name: str = Field(..., description="Job name")
    description: Optional[str] = Field(None, description="Job description")
    job_type: JobType = Field(..., description="Type of job")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Job parameters")
    priority: JobPriority = Field(default=JobPriority.NORMAL, description="Job priority")
    timeout: Optional[int] = Field(default=None, description="Job timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retry attempts")
    schedule: Optional[str] = Field(None, description="Cron schedule for recurring jobs")
    tags: List[str] = Field(default_factory=list, description="Job tags")
    
    @validator('timeout')
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    @validator('retry_count')
    def validate_retry_count(cls, v):
        if v < 0 or v > 10:
            raise ValueError("Retry count must be between 0 and 10")
        return v


class JobResponse(BaseModel):
    """Job response model."""
    id: str
    name: str
    description: Optional[str]
    job_type: JobType
    status: JobStatus
    priority: JobPriority
    parameters: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    progress: float
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    retry_count: int
    current_retry: int
    tags: List[str]
    created_by: str


class JobListResponse(BaseModel):
    """Job list response model."""
    jobs: List[JobResponse]
    total_count: int
    page: int
    size: int
    filters_applied: Dict[str, Any]


class JobStatsResponse(BaseModel):
    """Job statistics response model."""
    total_jobs: int
    pending_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    cancelled_jobs: int
    average_completion_time: Optional[float]
    success_rate: float
    jobs_by_type: Dict[str, int]
    jobs_by_priority: Dict[str, int]


@router.post("/", response_model=JobResponse)
async def create_job(
    job_request: JobCreateRequest,
    job_manager: JobManager = Depends(get_job_manager),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Create a new job.
    
    Creates a new job with the specified parameters and adds it to the job queue.
    Returns the created job with its assigned ID and initial status.
    """
    try:
        logger.info(f"Creating job: {job_request.name} for user: {current_user.get('username')}")
        
        # Create job data
        job_data = {
            "name": job_request.name,
            "description": job_request.description,
            "job_type": job_request.job_type.value,
            "parameters": job_request.parameters,
            "priority": job_request.priority.value,
            "timeout": job_request.timeout,
            "retry_count": job_request.retry_count,
            "schedule": job_request.schedule,
            "tags": job_request.tags,
            "created_by": current_user.get("username", "unknown")
        }
        
        # Create job through job manager
        job = await job_manager.create_job(job_data)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create job"
            )
        
        return JobResponse(
            id=job["id"],
            name=job["name"],
            description=job.get("description"),
            job_type=JobType(job["job_type"]),
            status=JobStatus(job["status"]),
            priority=JobPriority(job["priority"]),
            parameters=job["parameters"],
            created_at=job["created_at"],
            started_at=job.get("started_at"),
            completed_at=job.get("completed_at"),
            progress=job.get("progress", 0.0),
            result=job.get("result"),
            error_message=job.get("error_message"),
            retry_count=job["retry_count"],
            current_retry=job.get("current_retry", 0),
            tags=job["tags"],
            created_by=job["created_by"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create job"
        )


@router.get("/", response_model=JobListResponse)
async def list_jobs(
    pagination: Dict[str, int] = Depends(get_pagination_params),
    status_filter: Optional[JobStatus] = Query(None, description="Filter by job status"),
    job_type_filter: Optional[JobType] = Query(None, description="Filter by job type"),
    priority_filter: Optional[JobPriority] = Query(None, description="Filter by priority"),
    created_by_filter: Optional[str] = Query(None, description="Filter by creator"),
    tag_filter: Optional[str] = Query(None, description="Filter by tag"),
    job_manager: JobManager = Depends(get_job_manager),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    List jobs with optional filtering and pagination.
    
    Returns a paginated list of jobs with optional filters for status,
    type, priority, creator, and tags.
    """
    try:
        logger.info(f"Listing jobs for user: {current_user.get('username')}")
        
        # Build filters
        filters = {}
        if status_filter:
            filters["status"] = status_filter.value
        if job_type_filter:
            filters["job_type"] = job_type_filter.value
        if priority_filter:
            filters["priority"] = priority_filter.value
        if created_by_filter:
            filters["created_by"] = created_by_filter
        if tag_filter:
            filters["tags"] = tag_filter
        
        # Get jobs from job manager
        jobs_data = await job_manager.list_jobs(
            offset=pagination["offset"],
            limit=pagination["limit"],
            filters=filters
        )
        
        # Convert to response format
        jobs = []
        for job_data in jobs_data.get("jobs", []):
            job_response = JobResponse(
                id=job_data["id"],
                name=job_data["name"],
                description=job_data.get("description"),
                job_type=JobType(job_data["job_type"]),
                status=JobStatus(job_data["status"]),
                priority=JobPriority(job_data["priority"]),
                parameters=job_data["parameters"],
                created_at=job_data["created_at"],
                started_at=job_data.get("started_at"),
                completed_at=job_data.get("completed_at"),
                progress=job_data.get("progress", 0.0),
                result=job_data.get("result"),
                error_message=job_data.get("error_message"),
                retry_count=job_data["retry_count"],
                current_retry=job_data.get("current_retry", 0),
                tags=job_data["tags"],
                created_by=job_data["created_by"]
            )
            jobs.append(job_response)
        
        return JobListResponse(
            jobs=jobs,
            total_count=jobs_data.get("total_count", 0),
            page=pagination["page"],
            size=pagination["size"],
            filters_applied=filters
        )
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job list"
        )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str = Path(..., description="Job ID"),
    job_manager: JobManager = Depends(get_job_manager),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get detailed information about a specific job.
    
    Returns comprehensive job information including status, progress,
    results, and execution history.
    """
    try:
        logger.info(f"Getting job: {job_id}")
        
        # Get job from job manager
        job_data = await job_manager.get_job(job_id)
        
        if not job_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}"
            )
        
        return JobResponse(
            id=job_data["id"],
            name=job_data["name"],
            description=job_data.get("description"),
            job_type=JobType(job_data["job_type"]),
            status=JobStatus(job_data["status"]),
            priority=JobPriority(job_data["priority"]),
            parameters=job_data["parameters"],
            created_at=job_data["created_at"],
            started_at=job_data.get("started_at"),
            completed_at=job_data.get("completed_at"),
            progress=job_data.get("progress", 0.0),
            result=job_data.get("result"),
            error_message=job_data.get("error_message"),
            retry_count=job_data["retry_count"],
            current_retry=job_data.get("current_retry", 0),
            tags=job_data["tags"],
            created_by=job_data["created_by"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job"
        )


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str = Path(..., description="Job ID"),
    job_manager: JobManager = Depends(get_job_manager),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Cancel a running or pending job.
    
    Cancels the specified job if it's in a cancellable state.
    Returns the updated job status.
    """
    try:
        logger.info(f"Cancelling job: {job_id}")
        
        # Cancel job through job manager
        success = await job_manager.cancel_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to cancel job or job is not cancellable"
            )
        
        return {
            "success": True,
            "message": f"Job {job_id} cancelled successfully",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel job"
        )


@router.get("/stats/summary", response_model=JobStatsResponse)
async def get_job_statistics(
    days: int = Query(default=7, description="Number of days to include in statistics"),
    job_manager: JobManager = Depends(get_job_manager),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get job statistics and metrics.
    
    Returns comprehensive statistics including job counts by status,
    success rates, and performance metrics.
    """
    try:
        logger.info(f"Getting job statistics for {days} days")
        
        # Get statistics from job manager
        stats = await job_manager.get_statistics(days=days)
        
        return JobStatsResponse(
            total_jobs=stats.get("total_jobs", 0),
            pending_jobs=stats.get("pending_jobs", 0),
            running_jobs=stats.get("running_jobs", 0),
            completed_jobs=stats.get("completed_jobs", 0),
            failed_jobs=stats.get("failed_jobs", 0),
            cancelled_jobs=stats.get("cancelled_jobs", 0),
            average_completion_time=stats.get("average_completion_time"),
            success_rate=stats.get("success_rate", 0.0),
            jobs_by_type=stats.get("jobs_by_type", {}),
            jobs_by_priority=stats.get("jobs_by_priority", {})
        )
        
    except Exception as e:
        logger.error(f"Error getting job statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job statistics"
        )
