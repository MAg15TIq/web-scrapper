"""
Advanced Queue Management System for Phase 3: Enterprise & Scalability Features.

This module provides enhanced job queue management with resource allocation optimization,
queue partitioning, sharding, and priority-based resource allocation.
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import heapq
import hashlib

from models.task import Task, TaskType, TaskStatus, TaskPriority
from services.redis_service import RedisService


class QueueType(Enum):
    """Queue type enumeration."""
    PRIORITY = "priority"
    FIFO = "fifo"
    LIFO = "lifo"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"


class ResourceType(Enum):
    """Resource type enumeration."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    GPU = "gpu"


@dataclass
class ResourceRequirement:
    """Resource requirement for a task."""
    cpu_cores: float = 1.0
    memory_mb: int = 512
    network_mbps: float = 10.0
    storage_mb: int = 100
    gpu_count: int = 0
    estimated_duration: int = 300  # seconds
    max_retries: int = 3


@dataclass
class QueuePartition:
    """Queue partition configuration."""
    partition_id: str
    name: str
    queue_type: QueueType
    max_size: int = 10000
    priority_weight: float = 1.0
    resource_limits: Dict[str, Any] = None
    tenant_id: Optional[str] = None
    tags: List[str] = None


@dataclass
class QueueMetrics:
    """Queue performance metrics."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    average_wait_time: float = 0.0
    average_execution_time: float = 0.0
    throughput_per_minute: float = 0.0
    resource_utilization: Dict[str, float] = None


class AdvancedQueueManager:
    """
    Advanced queue management system with resource allocation optimization,
    partitioning, sharding, and intelligent task distribution.
    """
    
    def __init__(self, redis_service: RedisService):
        self.redis_service = redis_service
        self.logger = logging.getLogger("advanced_queue_manager")
        
        # Queue partitions
        self.partitions: Dict[str, QueuePartition] = {}
        self.partition_queues: Dict[str, deque] = {}
        self.partition_metrics: Dict[str, QueueMetrics] = {}
        
        # Resource management
        self.available_resources: Dict[str, float] = {
            "cpu_cores": 8.0,
            "memory_mb": 16384,
            "network_mbps": 1000.0,
            "storage_mb": 100000,
            "gpu_count": 0
        }
        self.allocated_resources: Dict[str, float] = defaultdict(float)
        self.resource_reservations: Dict[str, Dict[str, float]] = {}
        
        # Task management
        self.pending_tasks: Dict[str, Task] = {}
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.task_requirements: Dict[str, ResourceRequirement] = {}
        
        # Scheduling
        self.scheduler_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.scheduling_interval = 1.0  # seconds
        
        # Load balancing
        self.load_balancing_strategy = "least_loaded"
        self.node_loads: Dict[str, float] = {}
        
        # Metrics
        self.metrics_history: List[Dict[str, Any]] = []
        self.metrics_retention_hours = 24
        
        self.logger.info("Advanced queue manager initialized")
    
    async def start(self):
        """Start the queue manager."""
        if not self.is_running:
            self.is_running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            self.logger.info("Advanced queue manager started")
    
    async def stop(self):
        """Stop the queue manager."""
        if self.is_running:
            self.is_running = False
            if self.scheduler_task:
                self.scheduler_task.cancel()
                try:
                    await self.scheduler_task
                except asyncio.CancelledError:
                    pass
            self.logger.info("Advanced queue manager stopped")
    
    def create_partition(
        self,
        partition_id: str,
        name: str,
        queue_type: QueueType = QueueType.PRIORITY,
        max_size: int = 10000,
        priority_weight: float = 1.0,
        resource_limits: Dict[str, Any] = None,
        tenant_id: Optional[str] = None,
        tags: List[str] = None
    ) -> bool:
        """Create a new queue partition."""
        try:
            if partition_id in self.partitions:
                self.logger.warning(f"Partition {partition_id} already exists")
                return False
            
            partition = QueuePartition(
                partition_id=partition_id,
                name=name,
                queue_type=queue_type,
                max_size=max_size,
                priority_weight=priority_weight,
                resource_limits=resource_limits or {},
                tenant_id=tenant_id,
                tags=tags or []
            )
            
            self.partitions[partition_id] = partition
            self.partition_queues[partition_id] = deque()
            self.partition_metrics[partition_id] = QueueMetrics(
                resource_utilization=defaultdict(float)
            )
            
            self.logger.info(f"Created partition: {partition_id} ({name})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create partition {partition_id}: {e}")
            return False
    
    def remove_partition(self, partition_id: str) -> bool:
        """Remove a queue partition."""
        try:
            if partition_id not in self.partitions:
                self.logger.warning(f"Partition {partition_id} does not exist")
                return False
            
            # Move pending tasks to default partition
            if partition_id in self.partition_queues:
                pending_tasks = list(self.partition_queues[partition_id])
                if pending_tasks and "default" in self.partition_queues:
                    self.partition_queues["default"].extend(pending_tasks)
            
            # Clean up
            self.partitions.pop(partition_id, None)
            self.partition_queues.pop(partition_id, None)
            self.partition_metrics.pop(partition_id, None)
            
            self.logger.info(f"Removed partition: {partition_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove partition {partition_id}: {e}")
            return False
    
    async def submit_task(
        self,
        task: Task,
        resource_requirements: ResourceRequirement = None,
        partition_id: str = "default",
        priority_boost: float = 0.0
    ) -> bool:
        """Submit a task to a specific partition."""
        try:
            # Ensure default partition exists
            if "default" not in self.partitions:
                self.create_partition("default", "Default Queue")
            
            # Use specified partition or default
            target_partition = partition_id if partition_id in self.partitions else "default"
            
            # Set resource requirements
            if resource_requirements:
                self.task_requirements[task.id] = resource_requirements
            else:
                self.task_requirements[task.id] = ResourceRequirement()
            
            # Check partition capacity
            partition = self.partitions[target_partition]
            if len(self.partition_queues[target_partition]) >= partition.max_size:
                self.logger.warning(f"Partition {target_partition} is at capacity")
                return False
            
            # Check resource availability
            if not await self._check_resource_availability(task.id):
                self.logger.warning(f"Insufficient resources for task {task.id}")
                # Still queue the task, but mark it as resource-constrained
                task.metadata = task.metadata or {}
                task.metadata["resource_constrained"] = True
            
            # Add to partition queue
            task.metadata = task.metadata or {}
            task.metadata["partition_id"] = target_partition
            task.metadata["priority_boost"] = priority_boost
            task.metadata["submitted_at"] = datetime.now().isoformat()
            
            # Add to appropriate queue based on partition type
            await self._add_to_partition_queue(task, target_partition)
            
            # Update metrics
            self.partition_metrics[target_partition].total_tasks += 1
            self.partition_metrics[target_partition].pending_tasks += 1
            
            self.logger.debug(f"Task {task.id} submitted to partition {target_partition}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit task {task.id}: {e}")
            return False
    
    async def _add_to_partition_queue(self, task: Task, partition_id: str):
        """Add task to partition queue based on queue type."""
        partition = self.partitions[partition_id]
        queue = self.partition_queues[partition_id]
        
        if partition.queue_type == QueueType.PRIORITY:
            # Priority queue - use heapq for efficient priority handling
            priority_score = self._calculate_priority_score(task, partition)
            heapq.heappush(queue, (priority_score, time.time(), task))
        
        elif partition.queue_type == QueueType.FIFO:
            # First In, First Out
            queue.append(task)
        
        elif partition.queue_type == QueueType.LIFO:
            # Last In, First Out
            queue.appendleft(task)
        
        elif partition.queue_type == QueueType.WEIGHTED:
            # Weighted queue based on partition weight
            weight_score = partition.priority_weight * task.priority
            heapq.heappush(queue, (weight_score, time.time(), task))
        
        else:
            # Default to FIFO
            queue.append(task)
        
        # Store in pending tasks
        self.pending_tasks[task.id] = task
    
    def _calculate_priority_score(self, task: Task, partition: QueuePartition) -> float:
        """Calculate priority score for task placement."""
        base_priority = task.priority
        partition_weight = partition.priority_weight
        priority_boost = task.metadata.get("priority_boost", 0.0)
        
        # Consider resource requirements
        requirements = self.task_requirements.get(task.id, ResourceRequirement())
        resource_factor = 1.0 / max(requirements.estimated_duration, 1)
        
        # Calculate final score (lower is higher priority)
        score = -(base_priority * partition_weight + priority_boost) * resource_factor
        return score

    async def _check_resource_availability(self, task_id: str) -> bool:
        """Check if resources are available for a task."""
        requirements = self.task_requirements.get(task_id, ResourceRequirement())

        # Check each resource type
        resource_checks = {
            "cpu_cores": requirements.cpu_cores,
            "memory_mb": requirements.memory_mb,
            "network_mbps": requirements.network_mbps,
            "storage_mb": requirements.storage_mb,
            "gpu_count": requirements.gpu_count
        }

        for resource_type, required_amount in resource_checks.items():
            available = self.available_resources.get(resource_type, 0)
            allocated = self.allocated_resources.get(resource_type, 0)

            if available - allocated < required_amount:
                return False

        return True

    async def _reserve_resources(self, task_id: str) -> bool:
        """Reserve resources for a task."""
        requirements = self.task_requirements.get(task_id, ResourceRequirement())

        # Check availability first
        if not await self._check_resource_availability(task_id):
            return False

        # Reserve resources
        reservations = {
            "cpu_cores": requirements.cpu_cores,
            "memory_mb": requirements.memory_mb,
            "network_mbps": requirements.network_mbps,
            "storage_mb": requirements.storage_mb,
            "gpu_count": requirements.gpu_count
        }

        for resource_type, amount in reservations.items():
            self.allocated_resources[resource_type] += amount

        self.resource_reservations[task_id] = reservations
        return True

    async def _release_resources(self, task_id: str):
        """Release resources reserved for a task."""
        if task_id in self.resource_reservations:
            reservations = self.resource_reservations[task_id]

            for resource_type, amount in reservations.items():
                self.allocated_resources[resource_type] -= amount
                # Ensure we don't go negative
                self.allocated_resources[resource_type] = max(0, self.allocated_resources[resource_type])

            del self.resource_reservations[task_id]

    async def _scheduler_loop(self):
        """Main scheduler loop for task assignment."""
        while self.is_running:
            try:
                await self._schedule_tasks()
                await self._update_metrics()
                await asyncio.sleep(self.scheduling_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5)

    async def _schedule_tasks(self):
        """Schedule tasks from partitions to available resources."""
        # Get next tasks from each partition
        schedulable_tasks = []

        for partition_id, partition in self.partitions.items():
            queue = self.partition_queues[partition_id]

            if not queue:
                continue

            # Get next task based on queue type
            next_task = await self._get_next_task_from_partition(partition_id)
            if next_task:
                schedulable_tasks.append((next_task, partition_id, partition.priority_weight))

        # Sort by priority weight
        schedulable_tasks.sort(key=lambda x: x[2], reverse=True)

        # Try to schedule tasks
        for task, partition_id, weight in schedulable_tasks:
            if await self._try_schedule_task(task, partition_id):
                break  # Schedule one task per iteration to maintain fairness

    async def _get_next_task_from_partition(self, partition_id: str) -> Optional[Task]:
        """Get the next task from a partition queue."""
        partition = self.partitions[partition_id]
        queue = self.partition_queues[partition_id]

        if not queue:
            return None

        try:
            if partition.queue_type == QueueType.PRIORITY or partition.queue_type == QueueType.WEIGHTED:
                # Priority queue - get highest priority task
                if queue:
                    _, _, task = heapq.heappop(queue)
                    return task

            elif partition.queue_type == QueueType.FIFO:
                # First In, First Out
                return queue.popleft()

            elif partition.queue_type == QueueType.LIFO:
                # Last In, First Out
                return queue.pop()

            elif partition.queue_type == QueueType.ROUND_ROBIN:
                # Round robin - simple FIFO for now
                return queue.popleft()

            else:
                # Default to FIFO
                return queue.popleft()

        except (IndexError, KeyError):
            return None

    async def _try_schedule_task(self, task: Task, partition_id: str) -> bool:
        """Try to schedule a task if resources are available."""
        try:
            # Check resource availability
            if not await self._check_resource_availability(task.id):
                # Put task back in queue if resources not available
                await self._add_to_partition_queue(task, partition_id)
                return False

            # Reserve resources
            if not await self._reserve_resources(task.id):
                await self._add_to_partition_queue(task, partition_id)
                return False

            # Move task to running state
            self.pending_tasks.pop(task.id, None)
            self.running_tasks[task.id] = task
            task.status = TaskStatus.RUNNING
            task.metadata = task.metadata or {}
            task.metadata["started_at"] = datetime.now().isoformat()

            # Update metrics
            metrics = self.partition_metrics[partition_id]
            metrics.pending_tasks -= 1
            metrics.running_tasks += 1

            # Calculate wait time
            submitted_at = task.metadata.get("submitted_at")
            if submitted_at:
                wait_time = (datetime.now() - datetime.fromisoformat(submitted_at)).total_seconds()
                metrics.average_wait_time = (metrics.average_wait_time + wait_time) / 2

            self.logger.debug(f"Scheduled task {task.id} from partition {partition_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to schedule task {task.id}: {e}")
            return False

    async def complete_task(self, task_id: str, success: bool = True) -> bool:
        """Mark a task as completed and release its resources."""
        try:
            if task_id not in self.running_tasks:
                self.logger.warning(f"Task {task_id} not found in running tasks")
                return False

            task = self.running_tasks.pop(task_id)
            self.completed_tasks[task_id] = task

            # Update task status
            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            task.metadata = task.metadata or {}
            task.metadata["completed_at"] = datetime.now().isoformat()

            # Calculate execution time
            started_at = task.metadata.get("started_at")
            if started_at:
                execution_time = (datetime.now() - datetime.fromisoformat(started_at)).total_seconds()
                task.metadata["execution_time"] = execution_time

            # Release resources
            await self._release_resources(task_id)

            # Update metrics
            partition_id = task.metadata.get("partition_id", "default")
            if partition_id in self.partition_metrics:
                metrics = self.partition_metrics[partition_id]
                metrics.running_tasks -= 1

                if success:
                    metrics.completed_tasks += 1
                else:
                    metrics.failed_tasks += 1

                # Update average execution time
                if started_at:
                    execution_time = (datetime.now() - datetime.fromisoformat(started_at)).total_seconds()
                    metrics.average_execution_time = (metrics.average_execution_time + execution_time) / 2

            self.logger.debug(f"Completed task {task_id} (success: {success})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to complete task {task_id}: {e}")
            return False

    async def _update_metrics(self):
        """Update queue metrics and performance statistics."""
        try:
            current_time = datetime.now()

            for partition_id, metrics in self.partition_metrics.items():
                # Calculate throughput
                completed_in_last_minute = 0
                cutoff_time = current_time - timedelta(minutes=1)

                for task in self.completed_tasks.values():
                    completed_at = task.metadata.get("completed_at")
                    if completed_at:
                        completed_time = datetime.fromisoformat(completed_at)
                        if completed_time > cutoff_time:
                            task_partition = task.metadata.get("partition_id", "default")
                            if task_partition == partition_id:
                                completed_in_last_minute += 1

                metrics.throughput_per_minute = completed_in_last_minute

                # Update resource utilization
                partition = self.partitions[partition_id]
                if partition.resource_limits:
                    for resource_type, limit in partition.resource_limits.items():
                        allocated = self.allocated_resources.get(resource_type, 0)
                        utilization = (allocated / limit) * 100 if limit > 0 else 0
                        metrics.resource_utilization[resource_type] = utilization

            # Store metrics history
            metrics_snapshot = {
                "timestamp": current_time.isoformat(),
                "partitions": {
                    partition_id: asdict(metrics)
                    for partition_id, metrics in self.partition_metrics.items()
                },
                "global_resources": {
                    "available": self.available_resources.copy(),
                    "allocated": dict(self.allocated_resources),
                    "utilization": {
                        resource_type: (allocated / self.available_resources.get(resource_type, 1)) * 100
                        for resource_type, allocated in self.allocated_resources.items()
                    }
                }
            }

            self.metrics_history.append(metrics_snapshot)

            # Cleanup old metrics
            cutoff_time = current_time - timedelta(hours=self.metrics_retention_hours)
            self.metrics_history = [
                m for m in self.metrics_history
                if datetime.fromisoformat(m["timestamp"]) > cutoff_time
            ]

        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")

    # Public API methods

    def get_partition_info(self, partition_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific partition."""
        if partition_id not in self.partitions:
            return None

        partition = self.partitions[partition_id]
        metrics = self.partition_metrics[partition_id]
        queue_size = len(self.partition_queues[partition_id])

        return {
            "partition": asdict(partition),
            "metrics": asdict(metrics),
            "queue_size": queue_size,
            "queue_type": partition.queue_type.value
        }

    def get_all_partitions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all partitions."""
        return {
            partition_id: self.get_partition_info(partition_id)
            for partition_id in self.partitions.keys()
        }

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource allocation status."""
        utilization = {}
        for resource_type, available in self.available_resources.items():
            allocated = self.allocated_resources.get(resource_type, 0)
            utilization[resource_type] = {
                "available": available,
                "allocated": allocated,
                "free": available - allocated,
                "utilization_percent": (allocated / available) * 100 if available > 0 else 0
            }

        return {
            "resources": utilization,
            "active_reservations": len(self.resource_reservations),
            "total_tasks": {
                "pending": len(self.pending_tasks),
                "running": len(self.running_tasks),
                "completed": len(self.completed_tasks)
            }
        }

    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        total_pending = sum(len(queue) for queue in self.partition_queues.values())
        total_running = len(self.running_tasks)
        total_completed = len(self.completed_tasks)

        partition_stats = {}
        for partition_id, metrics in self.partition_metrics.items():
            partition_stats[partition_id] = {
                "total_tasks": metrics.total_tasks,
                "completed_tasks": metrics.completed_tasks,
                "failed_tasks": metrics.failed_tasks,
                "pending_tasks": len(self.partition_queues[partition_id]),
                "running_tasks": metrics.running_tasks,
                "success_rate": (metrics.completed_tasks / max(metrics.total_tasks, 1)) * 100,
                "average_wait_time": metrics.average_wait_time,
                "average_execution_time": metrics.average_execution_time,
                "throughput_per_minute": metrics.throughput_per_minute
            }

        return {
            "global": {
                "total_pending": total_pending,
                "total_running": total_running,
                "total_completed": total_completed,
                "total_partitions": len(self.partitions)
            },
            "partitions": partition_stats,
            "resource_utilization": {
                resource_type: (allocated / self.available_resources.get(resource_type, 1)) * 100
                for resource_type, allocated in self.allocated_resources.items()
            }
        }

    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics history for the specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            m for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]

    def update_available_resources(self, resources: Dict[str, float]):
        """Update available resources."""
        self.available_resources.update(resources)
        self.logger.info(f"Updated available resources: {resources}")

    def set_load_balancing_strategy(self, strategy: str):
        """Set load balancing strategy."""
        valid_strategies = ["least_loaded", "round_robin", "weighted", "random"]
        if strategy in valid_strategies:
            self.load_balancing_strategy = strategy
            self.logger.info(f"Load balancing strategy set to: {strategy}")
        else:
            self.logger.warning(f"Invalid load balancing strategy: {strategy}")

    async def rebalance_queues(self):
        """Rebalance tasks across partitions based on load."""
        try:
            # Calculate load for each partition
            partition_loads = {}
            for partition_id in self.partitions.keys():
                queue_size = len(self.partition_queues[partition_id])
                running_tasks = self.partition_metrics[partition_id].running_tasks
                partition_loads[partition_id] = queue_size + running_tasks

            # Find overloaded and underloaded partitions
            avg_load = sum(partition_loads.values()) / len(partition_loads) if partition_loads else 0
            overloaded = {pid: load for pid, load in partition_loads.items() if load > avg_load * 1.5}
            underloaded = {pid: load for pid, load in partition_loads.items() if load < avg_load * 0.5}

            # Move tasks from overloaded to underloaded partitions
            for overloaded_partition in overloaded.keys():
                if not underloaded:
                    break

                underloaded_partition = min(underloaded.keys(), key=lambda x: underloaded[x])

                # Move one task
                source_queue = self.partition_queues[overloaded_partition]
                target_queue = self.partition_queues[underloaded_partition]

                if source_queue:
                    if self.partitions[overloaded_partition].queue_type in [QueueType.PRIORITY, QueueType.WEIGHTED]:
                        # For priority queues, get the lowest priority task
                        tasks = []
                        while source_queue:
                            tasks.append(heapq.heappop(source_queue))

                        if tasks:
                            # Move the lowest priority task
                            moved_task = tasks.pop()  # Last item has lowest priority
                            target_queue.append(moved_task[2])  # Extract task from tuple

                            # Put remaining tasks back
                            for task_tuple in tasks:
                                heapq.heappush(source_queue, task_tuple)
                    else:
                        # For FIFO/LIFO queues, move from the end
                        moved_task = source_queue.pop()
                        target_queue.append(moved_task)

                    # Update load tracking
                    underloaded[underloaded_partition] += 1
                    if underloaded[underloaded_partition] >= avg_load:
                        del underloaded[underloaded_partition]

            self.logger.debug("Queue rebalancing completed")

        except Exception as e:
            self.logger.error(f"Failed to rebalance queues: {e}")
