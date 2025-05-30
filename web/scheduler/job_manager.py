"""
Job Manager
Handles job creation, scheduling, execution, and monitoring.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import json

from cli.agent_communication import AgentCommunicationLayer


# Configure logging
logger = logging.getLogger("job_manager")


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class JobPriority(Enum):
    """Job priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Job:
    """Job data structure."""
    id: str
    name: str
    description: Optional[str]
    job_type: str
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
    timeout: Optional[int]
    schedule: Optional[str]
    tags: List[str]
    created_by: str
    agent_id: Optional[str] = None
    task_id: Optional[str] = None


class JobQueue:
    """Priority-based job queue."""
    
    def __init__(self):
        self.queues = {
            JobPriority.URGENT: deque(),
            JobPriority.HIGH: deque(),
            JobPriority.NORMAL: deque(),
            JobPriority.LOW: deque()
        }
        self.job_lookup: Dict[str, Job] = {}
    
    def add_job(self, job: Job):
        """Add job to appropriate priority queue."""
        self.queues[job.priority].append(job)
        self.job_lookup[job.id] = job
    
    def get_next_job(self) -> Optional[Job]:
        """Get next job from highest priority queue."""
        for priority in [JobPriority.URGENT, JobPriority.HIGH, JobPriority.NORMAL, JobPriority.LOW]:
            if self.queues[priority]:
                job = self.queues[priority].popleft()
                return job
        return None
    
    def remove_job(self, job_id: str) -> Optional[Job]:
        """Remove job from queue."""
        if job_id in self.job_lookup:
            job = self.job_lookup[job_id]
            try:
                self.queues[job.priority].remove(job)
            except ValueError:
                pass  # Job might have already been removed
            del self.job_lookup[job_id]
            return job
        return None
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self.job_lookup.get(job_id)
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get sizes of all priority queues."""
        return {priority.value: len(queue) for priority, queue in self.queues.items()}
    
    def get_all_jobs(self) -> List[Job]:
        """Get all jobs in queue."""
        return list(self.job_lookup.values())


class JobManager:
    """Manages job lifecycle, scheduling, and execution."""
    
    def __init__(self):
        self.logger = logging.getLogger("job_manager")
        self.agent_comm = AgentCommunicationLayer()
        
        # Job storage
        self.pending_queue = JobQueue()
        self.running_jobs: Dict[str, Job] = {}
        self.completed_jobs: Dict[str, Job] = {}
        self.job_history: deque = deque(maxlen=10000)
        
        # Execution state
        self.is_running = False
        self.execution_task = None
        self.max_concurrent_jobs = 10
        self.job_timeout_default = 3600  # 1 hour
        
        # Statistics
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "cancelled_jobs": 0,
            "total_execution_time": 0.0
        }
        
        self.logger.info("Job manager initialized")
    
    async def start(self):
        """Start the job manager."""
        if self.is_running:
            self.logger.warning("Job manager is already running")
            return
        
        self.is_running = True
        self.execution_task = asyncio.create_task(self._execution_loop())
        
        # Initialize agent communication
        await self.agent_comm.initialize_agents()
        
        self.logger.info("Job manager started")
    
    async def stop(self):
        """Stop the job manager."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.execution_task:
            self.execution_task.cancel()
            try:
                await self.execution_task
            except asyncio.CancelledError:
                pass
        
        # Cancel running jobs
        for job in list(self.running_jobs.values()):
            await self._cancel_job_execution(job)
        
        self.logger.info("Job manager stopped")
    
    async def create_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new job."""
        try:
            # Generate job ID
            job_id = str(uuid.uuid4())
            
            # Create job object
            job = Job(
                id=job_id,
                name=job_data["name"],
                description=job_data.get("description"),
                job_type=job_data["job_type"],
                status=JobStatus.PENDING,
                priority=JobPriority(job_data.get("priority", "normal")),
                parameters=job_data.get("parameters", {}),
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                progress=0.0,
                result=None,
                error_message=None,
                retry_count=job_data.get("retry_count", 3),
                current_retry=0,
                timeout=job_data.get("timeout", self.job_timeout_default),
                schedule=job_data.get("schedule"),
                tags=job_data.get("tags", []),
                created_by=job_data["created_by"]
            )
            
            # Add to pending queue
            self.pending_queue.add_job(job)
            
            # Update statistics
            self.stats["total_jobs"] += 1
            
            # Add to history
            self.job_history.append({
                "action": "created",
                "job_id": job_id,
                "timestamp": datetime.now(),
                "details": {"name": job.name, "type": job.job_type}
            })
            
            self.logger.info(f"Job created: {job_id} - {job.name}")
            
            return asdict(job)
            
        except Exception as e:
            self.logger.error(f"Error creating job: {e}")
            return None
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID."""
        # Check pending queue
        job = self.pending_queue.get_job(job_id)
        if job:
            return asdict(job)
        
        # Check running jobs
        job = self.running_jobs.get(job_id)
        if job:
            return asdict(job)
        
        # Check completed jobs
        job = self.completed_jobs.get(job_id)
        if job:
            return asdict(job)
        
        return None
    
    async def list_jobs(self, offset: int = 0, limit: int = 50, 
                       filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """List jobs with pagination and filtering."""
        try:
            # Collect all jobs
            all_jobs = []
            
            # Add pending jobs
            all_jobs.extend(self.pending_queue.get_all_jobs())
            
            # Add running jobs
            all_jobs.extend(self.running_jobs.values())
            
            # Add completed jobs
            all_jobs.extend(self.completed_jobs.values())
            
            # Apply filters
            if filters:
                filtered_jobs = []
                for job in all_jobs:
                    if self._job_matches_filters(job, filters):
                        filtered_jobs.append(job)
                all_jobs = filtered_jobs
            
            # Sort by creation time (newest first)
            all_jobs.sort(key=lambda j: j.created_at, reverse=True)
            
            # Apply pagination
            total_count = len(all_jobs)
            paginated_jobs = all_jobs[offset:offset + limit]
            
            # Convert to dict format
            job_dicts = [asdict(job) for job in paginated_jobs]
            
            return {
                "jobs": job_dicts,
                "total_count": total_count,
                "offset": offset,
                "limit": limit
            }
            
        except Exception as e:
            self.logger.error(f"Error listing jobs: {e}")
            return {"jobs": [], "total_count": 0, "offset": offset, "limit": limit}
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        try:
            # Check if job is running
            if job_id in self.running_jobs:
                job = self.running_jobs[job_id]
                await self._cancel_job_execution(job)
                return True
            
            # Check if job is pending
            job = self.pending_queue.remove_job(job_id)
            if job:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                self.completed_jobs[job_id] = job
                
                # Update statistics
                self.stats["cancelled_jobs"] += 1
                
                # Add to history
                self.job_history.append({
                    "action": "cancelled",
                    "job_id": job_id,
                    "timestamp": datetime.now(),
                    "details": {"reason": "user_request"}
                })
                
                self.logger.info(f"Job cancelled: {job_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling job: {e}")
            return False
    
    async def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get job statistics."""
        try:
            # Calculate time-based statistics
            cutoff_time = datetime.now() - timedelta(days=days)
            
            recent_jobs = [
                job for job in list(self.completed_jobs.values())
                if job.created_at >= cutoff_time
            ]
            
            # Calculate success rate
            total_recent = len(recent_jobs)
            completed_recent = len([j for j in recent_jobs if j.status == JobStatus.COMPLETED])
            success_rate = (completed_recent / total_recent * 100) if total_recent > 0 else 0.0
            
            # Calculate average completion time
            completed_jobs = [j for j in recent_jobs if j.status == JobStatus.COMPLETED and j.started_at and j.completed_at]
            avg_completion_time = None
            if completed_jobs:
                total_time = sum((j.completed_at - j.started_at).total_seconds() for j in completed_jobs)
                avg_completion_time = total_time / len(completed_jobs)
            
            # Jobs by type
            jobs_by_type = defaultdict(int)
            for job in recent_jobs:
                jobs_by_type[job.job_type] += 1
            
            # Jobs by priority
            jobs_by_priority = defaultdict(int)
            for job in recent_jobs:
                jobs_by_priority[job.priority.value] += 1
            
            return {
                "total_jobs": self.stats["total_jobs"],
                "pending_jobs": len(self.pending_queue.get_all_jobs()),
                "running_jobs": len(self.running_jobs),
                "completed_jobs": self.stats["completed_jobs"],
                "failed_jobs": self.stats["failed_jobs"],
                "cancelled_jobs": self.stats["cancelled_jobs"],
                "success_rate": success_rate,
                "average_completion_time": avg_completion_time,
                "jobs_by_type": dict(jobs_by_type),
                "jobs_by_priority": dict(jobs_by_priority),
                "queue_sizes": self.pending_queue.get_queue_sizes()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def _execution_loop(self):
        """Main job execution loop."""
        while self.is_running:
            try:
                # Check if we can start more jobs
                if len(self.running_jobs) < self.max_concurrent_jobs:
                    # Get next job from queue
                    job = self.pending_queue.get_next_job()
                    if job:
                        await self._start_job_execution(job)
                
                # Check running jobs for completion/timeout
                await self._check_running_jobs()
                
                # Wait before next iteration
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(5)
    
    async def _start_job_execution(self, job: Job):
        """Start executing a job."""
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            self.running_jobs[job.id] = job
            
            # Submit task to agent system
            task_id = await self.agent_comm.submit_task(
                task_type=job.job_type,
                parameters=job.parameters
            )
            
            if task_id:
                job.task_id = task_id
                self.logger.info(f"Job started: {job.id} - Task: {task_id}")
            else:
                await self._complete_job(job, False, "Failed to submit task to agent system")
            
        except Exception as e:
            self.logger.error(f"Error starting job {job.id}: {e}")
            await self._complete_job(job, False, str(e))
    
    async def _check_running_jobs(self):
        """Check status of running jobs."""
        for job_id, job in list(self.running_jobs.items()):
            try:
                # Check for timeout
                if job.timeout and job.started_at:
                    elapsed = (datetime.now() - job.started_at).total_seconds()
                    if elapsed > job.timeout:
                        await self._complete_job(job, False, "Job timed out")
                        continue
                
                # Check task status if we have a task ID
                if job.task_id:
                    task_status = await self.agent_comm.get_task_status(job.task_id)
                    if task_status:
                        status = task_status.get("status")
                        progress = task_status.get("progress", 0)
                        
                        job.progress = progress
                        
                        if status == "completed":
                            result = task_status.get("result")
                            await self._complete_job(job, True, None, result)
                        elif status == "failed":
                            error = task_status.get("error", "Task failed")
                            await self._complete_job(job, False, error)
                
            except Exception as e:
                self.logger.error(f"Error checking job {job_id}: {e}")
    
    async def _complete_job(self, job: Job, success: bool, error_message: Optional[str] = None, 
                           result: Optional[Dict[str, Any]] = None):
        """Complete a job."""
        try:
            # Remove from running jobs
            if job.id in self.running_jobs:
                del self.running_jobs[job.id]
            
            # Update job status
            job.status = JobStatus.COMPLETED if success else JobStatus.FAILED
            job.completed_at = datetime.now()
            job.progress = 100.0 if success else job.progress
            job.error_message = error_message
            job.result = result
            
            # Move to completed jobs
            self.completed_jobs[job.id] = job
            
            # Update statistics
            if success:
                self.stats["completed_jobs"] += 1
            else:
                self.stats["failed_jobs"] += 1
            
            if job.started_at and job.completed_at:
                execution_time = (job.completed_at - job.started_at).total_seconds()
                self.stats["total_execution_time"] += execution_time
            
            # Add to history
            self.job_history.append({
                "action": "completed" if success else "failed",
                "job_id": job.id,
                "timestamp": datetime.now(),
                "details": {
                    "success": success,
                    "error": error_message,
                    "execution_time": execution_time if job.started_at else None
                }
            })
            
            self.logger.info(f"Job {'completed' if success else 'failed'}: {job.id}")
            
        except Exception as e:
            self.logger.error(f"Error completing job {job.id}: {e}")
    
    async def _cancel_job_execution(self, job: Job):
        """Cancel a running job."""
        try:
            # Remove from running jobs
            if job.id in self.running_jobs:
                del self.running_jobs[job.id]
            
            # Update job status
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            
            # Move to completed jobs
            self.completed_jobs[job.id] = job
            
            # Update statistics
            self.stats["cancelled_jobs"] += 1
            
            self.logger.info(f"Job execution cancelled: {job.id}")
            
        except Exception as e:
            self.logger.error(f"Error cancelling job execution {job.id}: {e}")
    
    def _job_matches_filters(self, job: Job, filters: Dict[str, Any]) -> bool:
        """Check if job matches the given filters."""
        for key, value in filters.items():
            if key == "status" and job.status.value != value:
                return False
            elif key == "job_type" and job.job_type != value:
                return False
            elif key == "priority" and job.priority.value != value:
                return False
            elif key == "created_by" and job.created_by != value:
                return False
            elif key == "tags" and value not in job.tags:
                return False
        
        return True
