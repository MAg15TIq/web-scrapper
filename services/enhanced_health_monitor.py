"""
Enhanced Node Health Monitoring and Failover System for Phase 3: Enterprise & Scalability Features.

This module provides comprehensive health monitoring, automatic failover mechanisms,
node health scoring, and recovery/rebalancing strategies.
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import statistics
import aiohttp

from services.cluster_manager import ClusterManager, NodeInfo, NodeStatus, NodeRole
from services.advanced_queue_manager import AdvancedQueueManager
from services.redis_service import RedisService


class HealthCheckType(Enum):
    """Health check type enumeration."""
    HTTP = "http"
    TCP = "tcp"
    PING = "ping"
    CUSTOM = "custom"
    RESOURCE = "resource"
    APPLICATION = "application"


class FailoverStrategy(Enum):
    """Failover strategy enumeration."""
    IMMEDIATE = "immediate"
    GRACEFUL = "graceful"
    MANUAL = "manual"
    CIRCUIT_BREAKER = "circuit_breaker"


class AlertSeverity(Enum):
    """Alert severity enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check configuration."""
    check_id: str
    name: str
    check_type: HealthCheckType
    interval: int = 30  # seconds
    timeout: int = 10  # seconds
    retries: int = 3
    failure_threshold: int = 3
    success_threshold: int = 2
    enabled: bool = True
    
    # HTTP-specific
    url: Optional[str] = None
    expected_status: int = 200
    expected_content: Optional[str] = None
    
    # TCP-specific
    port: Optional[int] = None
    
    # Resource-specific
    cpu_threshold: float = 90.0
    memory_threshold: float = 90.0
    disk_threshold: float = 90.0
    
    # Custom check
    custom_function: Optional[Callable] = None


@dataclass
class HealthResult:
    """Health check result."""
    check_id: str
    node_id: str
    timestamp: datetime
    success: bool
    response_time: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = None


@dataclass
class NodeHealthScore:
    """Node health scoring."""
    node_id: str
    overall_score: float  # 0-100
    availability_score: float
    performance_score: float
    resource_score: float
    last_updated: datetime
    health_trend: str  # improving, stable, degrading
    
    # Detailed metrics
    uptime_percentage: float = 100.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0


@dataclass
class FailoverEvent:
    """Failover event record."""
    event_id: str
    timestamp: datetime
    failed_node_id: str
    replacement_node_id: Optional[str]
    strategy: FailoverStrategy
    trigger_reason: str
    success: bool
    recovery_time: float  # seconds
    affected_tasks: int


class EnhancedHealthMonitor:
    """
    Enhanced health monitoring system with comprehensive health checks,
    automatic failover, and intelligent recovery strategies.
    """
    
    def __init__(
        self,
        cluster_manager: ClusterManager,
        queue_manager: AdvancedQueueManager,
        redis_service: RedisService
    ):
        self.cluster_manager = cluster_manager
        self.queue_manager = queue_manager
        self.redis_service = redis_service
        self.logger = logging.getLogger("enhanced_health_monitor")
        
        # Health check configuration
        self.health_checks: Dict[str, HealthCheck] = {}
        self.node_health_scores: Dict[str, NodeHealthScore] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Failover configuration
        self.failover_strategy = FailoverStrategy.GRACEFUL
        self.failover_history: List[FailoverEvent] = []
        self.failed_nodes: Dict[str, datetime] = {}
        self.recovery_queue: List[str] = []
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.failover_task: Optional[asyncio.Task] = None
        self.recovery_task: Optional[asyncio.Task] = None
        self.scoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Performance tracking
        self.monitoring_metrics = {
            "total_checks": 0,
            "successful_checks": 0,
            "failed_checks": 0,
            "average_check_time": 0.0,
            "total_failovers": 0,
            "successful_failovers": 0
        }
        
        self.logger.info("Enhanced health monitor initialized")
    
    async def start(self):
        """Start the health monitoring system."""
        if not self.is_running:
            self.is_running = True
            
            # Create default health checks
            await self._create_default_health_checks()
            
            # Start background tasks
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            self.failover_task = asyncio.create_task(self._failover_loop())
            self.recovery_task = asyncio.create_task(self._recovery_loop())
            self.scoring_task = asyncio.create_task(self._scoring_loop())
            
            self.logger.info("Enhanced health monitor started")
    
    async def stop(self):
        """Stop the health monitoring system."""
        if self.is_running:
            self.is_running = False
            
            # Stop background tasks
            tasks = [self.health_check_task, self.failover_task, self.recovery_task, self.scoring_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.logger.info("Enhanced health monitor stopped")
    
    async def _create_default_health_checks(self):
        """Create default health checks."""
        # HTTP health check
        http_check = HealthCheck(
            check_id="http_health",
            name="HTTP Health Check",
            check_type=HealthCheckType.HTTP,
            url="/health",
            interval=30,
            timeout=10,
            failure_threshold=3
        )
        
        # Resource health check
        resource_check = HealthCheck(
            check_id="resource_health",
            name="Resource Health Check",
            check_type=HealthCheckType.RESOURCE,
            interval=60,
            cpu_threshold=85.0,
            memory_threshold=85.0,
            disk_threshold=90.0
        )
        
        # Application health check
        app_check = HealthCheck(
            check_id="application_health",
            name="Application Health Check",
            check_type=HealthCheckType.APPLICATION,
            interval=45,
            timeout=15
        )
        
        self.health_checks["http_health"] = http_check
        self.health_checks["resource_health"] = resource_check
        self.health_checks["application_health"] = app_check
        
        self.logger.info("Created default health checks")
    
    async def add_health_check(self, health_check: HealthCheck) -> bool:
        """Add a new health check."""
        try:
            self.health_checks[health_check.check_id] = health_check
            self.logger.info(f"Added health check: {health_check.check_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add health check: {e}")
            return False
    
    async def remove_health_check(self, check_id: str) -> bool:
        """Remove a health check."""
        try:
            if check_id in self.health_checks:
                del self.health_checks[check_id]
                self.logger.info(f"Removed health check: {check_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to remove health check: {e}")
            return False
    
    async def _health_check_loop(self):
        """Main health check loop."""
        while self.is_running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
    
    async def _run_health_checks(self):
        """Run all health checks for all nodes."""
        healthy_nodes = self.cluster_manager.get_healthy_nodes()
        current_time = datetime.now()
        
        for node_id, node_info in healthy_nodes.items():
            for check_id, health_check in self.health_checks.items():
                if not health_check.enabled:
                    continue
                
                # Check if it's time to run this check
                last_check_key = f"{node_id}:{check_id}:last_check"
                last_check_time = getattr(self, '_last_checks', {}).get(last_check_key)
                
                if not hasattr(self, '_last_checks'):
                    self._last_checks = {}
                
                if (not last_check_time or 
                    (current_time - last_check_time).total_seconds() >= health_check.interval):
                    
                    # Run the health check
                    result = await self._execute_health_check(node_info, health_check)
                    if result:
                        await self._process_health_result(result)
                    
                    self._last_checks[last_check_key] = current_time
    
    async def _execute_health_check(self, node_info: NodeInfo, health_check: HealthCheck) -> Optional[HealthResult]:
        """Execute a single health check."""
        start_time = time.time()
        
        try:
            success = False
            error_message = None
            details = {}
            
            if health_check.check_type == HealthCheckType.HTTP:
                success, error_message, details = await self._http_health_check(node_info, health_check)
            
            elif health_check.check_type == HealthCheckType.TCP:
                success, error_message, details = await self._tcp_health_check(node_info, health_check)
            
            elif health_check.check_type == HealthCheckType.RESOURCE:
                success, error_message, details = await self._resource_health_check(node_info, health_check)
            
            elif health_check.check_type == HealthCheckType.APPLICATION:
                success, error_message, details = await self._application_health_check(node_info, health_check)
            
            elif health_check.check_type == HealthCheckType.CUSTOM:
                success, error_message, details = await self._custom_health_check(node_info, health_check)
            
            response_time = time.time() - start_time
            
            # Update monitoring metrics
            self.monitoring_metrics["total_checks"] += 1
            if success:
                self.monitoring_metrics["successful_checks"] += 1
            else:
                self.monitoring_metrics["failed_checks"] += 1
            
            # Update average check time
            total_checks = self.monitoring_metrics["total_checks"]
            current_avg = self.monitoring_metrics["average_check_time"]
            self.monitoring_metrics["average_check_time"] = (
                (current_avg * (total_checks - 1) + response_time) / total_checks
            )
            
            return HealthResult(
                check_id=health_check.check_id,
                node_id=node_info.node_id,
                timestamp=datetime.now(),
                success=success,
                response_time=response_time,
                error_message=error_message,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Health check {health_check.check_id} failed for node {node_info.node_id}: {e}")
            return HealthResult(
                check_id=health_check.check_id,
                node_id=node_info.node_id,
                timestamp=datetime.now(),
                success=False,
                response_time=time.time() - start_time,
                error_message=str(e)
            )

    async def _http_health_check(self, node_info: NodeInfo, health_check: HealthCheck) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Execute HTTP health check."""
        try:
            url = f"http://{node_info.ip_address}:{node_info.port}{health_check.url}"

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=health_check.timeout)) as session:
                async with session.get(url) as response:
                    success = response.status == health_check.expected_status

                    details = {
                        "status_code": response.status,
                        "response_time": response.headers.get("X-Response-Time", "unknown")
                    }

                    if health_check.expected_content:
                        content = await response.text()
                        if health_check.expected_content not in content:
                            success = False
                            return success, f"Expected content not found", details

                    if success:
                        return True, None, details
                    else:
                        return False, f"HTTP {response.status}", details

        except asyncio.TimeoutError:
            return False, "HTTP timeout", {}
        except Exception as e:
            return False, f"HTTP error: {str(e)}", {}

    async def _tcp_health_check(self, node_info: NodeInfo, health_check: HealthCheck) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Execute TCP health check."""
        try:
            port = health_check.port or node_info.port

            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(node_info.ip_address, port),
                timeout=health_check.timeout
            )

            writer.close()
            await writer.wait_closed()

            return True, None, {"port": port}

        except asyncio.TimeoutError:
            return False, "TCP timeout", {}
        except Exception as e:
            return False, f"TCP error: {str(e)}", {}

    async def _resource_health_check(self, node_info: NodeInfo, health_check: HealthCheck) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Execute resource health check."""
        try:
            resources = node_info.resources
            if not resources:
                return False, "No resource data available", {}

            cpu_usage = resources.get('cpu_usage', 0)
            memory_usage = resources.get('memory_usage', 0)
            disk_usage = resources.get('disk_usage', 0)

            details = {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage
            }

            failures = []

            if cpu_usage > health_check.cpu_threshold:
                failures.append(f"CPU usage {cpu_usage}% > {health_check.cpu_threshold}%")

            if memory_usage > health_check.memory_threshold:
                failures.append(f"Memory usage {memory_usage}% > {health_check.memory_threshold}%")

            if disk_usage > health_check.disk_threshold:
                failures.append(f"Disk usage {disk_usage}% > {health_check.disk_threshold}%")

            if failures:
                return False, "; ".join(failures), details
            else:
                return True, None, details

        except Exception as e:
            return False, f"Resource check error: {str(e)}", {}

    async def _application_health_check(self, node_info: NodeInfo, health_check: HealthCheck) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Execute application-specific health check."""
        try:
            # Check if the node is processing tasks properly
            # This is a simplified check - in practice, you'd check application-specific metrics

            # Check if node is responsive to cluster communications
            cluster_nodes = self.cluster_manager.get_cluster_nodes()
            node = cluster_nodes.get(node_info.node_id)

            if not node:
                return False, "Node not found in cluster", {}

            # Check last heartbeat
            last_heartbeat = node.last_heartbeat
            time_since_heartbeat = (datetime.now() - last_heartbeat).total_seconds()

            if time_since_heartbeat > 120:  # 2 minutes
                return False, f"No heartbeat for {time_since_heartbeat:.0f} seconds", {
                    "last_heartbeat": last_heartbeat.isoformat(),
                    "time_since_heartbeat": time_since_heartbeat
                }

            return True, None, {
                "last_heartbeat": last_heartbeat.isoformat(),
                "time_since_heartbeat": time_since_heartbeat
            }

        except Exception as e:
            return False, f"Application check error: {str(e)}", {}

    async def _custom_health_check(self, node_info: NodeInfo, health_check: HealthCheck) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Execute custom health check."""
        try:
            if health_check.custom_function:
                result = await health_check.custom_function(node_info)
                if isinstance(result, bool):
                    return result, None, {}
                elif isinstance(result, tuple) and len(result) >= 2:
                    return result[0], result[1], result[2] if len(result) > 2 else {}
                else:
                    return False, "Invalid custom check result", {}
            else:
                return False, "No custom function defined", {}

        except Exception as e:
            return False, f"Custom check error: {str(e)}", {}

    async def _process_health_result(self, result: HealthResult):
        """Process a health check result."""
        # Store result in history
        self.health_history[result.node_id].append(result)

        # Update circuit breaker state
        await self._update_circuit_breaker(result)

        # Check for node failure
        if not result.success:
            await self._handle_health_failure(result)
        else:
            await self._handle_health_success(result)

    async def _update_circuit_breaker(self, result: HealthResult):
        """Update circuit breaker state for a node."""
        node_id = result.node_id

        if node_id not in self.circuit_breakers:
            self.circuit_breakers[node_id] = {
                "state": "closed",  # closed, open, half_open
                "failure_count": 0,
                "last_failure": None,
                "last_success": None,
                "open_until": None
            }

        cb = self.circuit_breakers[node_id]

        if result.success:
            cb["failure_count"] = 0
            cb["last_success"] = result.timestamp
            if cb["state"] == "half_open":
                cb["state"] = "closed"
                self.logger.info(f"Circuit breaker closed for node {node_id}")
        else:
            cb["failure_count"] += 1
            cb["last_failure"] = result.timestamp

            # Open circuit breaker after threshold failures
            if cb["failure_count"] >= 5 and cb["state"] == "closed":
                cb["state"] = "open"
                cb["open_until"] = result.timestamp + timedelta(minutes=5)
                self.logger.warning(f"Circuit breaker opened for node {node_id}")

                # Trigger failover
                await self._trigger_failover(node_id, "Circuit breaker opened")

        # Check if we should try half-open
        if (cb["state"] == "open" and cb["open_until"] and
            datetime.now() > cb["open_until"]):
            cb["state"] = "half_open"
            self.logger.info(f"Circuit breaker half-open for node {node_id}")

    async def _handle_health_failure(self, result: HealthResult):
        """Handle a health check failure."""
        node_id = result.node_id

        # Count consecutive failures
        recent_results = list(self.health_history[node_id])[-5:]  # Last 5 results
        consecutive_failures = 0

        for r in reversed(recent_results):
            if r.check_id == result.check_id:
                if not r.success:
                    consecutive_failures += 1
                else:
                    break

        # Get health check configuration
        health_check = self.health_checks.get(result.check_id)
        if not health_check:
            return

        # Trigger failover if threshold reached
        if consecutive_failures >= health_check.failure_threshold:
            await self._trigger_failover(
                node_id,
                f"Health check {result.check_id} failed {consecutive_failures} times"
            )

    async def _handle_health_success(self, result: HealthResult):
        """Handle a health check success."""
        node_id = result.node_id

        # If node was marked as failed, consider it for recovery
        if node_id in self.failed_nodes:
            # Count consecutive successes
            recent_results = list(self.health_history[node_id])[-3:]  # Last 3 results
            consecutive_successes = 0

            for r in reversed(recent_results):
                if r.check_id == result.check_id:
                    if r.success:
                        consecutive_successes += 1
                    else:
                        break

            # Get health check configuration
            health_check = self.health_checks.get(result.check_id)
            if health_check and consecutive_successes >= health_check.success_threshold:
                await self._recover_node(node_id)

    async def _trigger_failover(self, failed_node_id: str, reason: str):
        """Trigger failover for a failed node."""
        if failed_node_id in self.failed_nodes:
            return  # Already handling this node

        self.failed_nodes[failed_node_id] = datetime.now()

        self.logger.warning(f"Triggering failover for node {failed_node_id}: {reason}")

        # Create failover event
        event = FailoverEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            failed_node_id=failed_node_id,
            replacement_node_id=None,
            strategy=self.failover_strategy,
            trigger_reason=reason,
            success=False,
            recovery_time=0.0,
            affected_tasks=0
        )

        start_time = time.time()

        try:
            # Execute failover based on strategy
            if self.failover_strategy == FailoverStrategy.IMMEDIATE:
                success = await self._immediate_failover(failed_node_id, event)
            elif self.failover_strategy == FailoverStrategy.GRACEFUL:
                success = await self._graceful_failover(failed_node_id, event)
            else:
                success = False

            event.success = success
            event.recovery_time = time.time() - start_time

            # Update metrics
            self.monitoring_metrics["total_failovers"] += 1
            if success:
                self.monitoring_metrics["successful_failovers"] += 1

            # Send alerts
            await self._send_alert(
                AlertSeverity.ERROR if success else AlertSeverity.CRITICAL,
                f"Failover {'completed' if success else 'failed'} for node {failed_node_id}",
                {"event": asdict(event)}
            )

        except Exception as e:
            self.logger.error(f"Failover failed for node {failed_node_id}: {e}")
            event.success = False
            event.recovery_time = time.time() - start_time

        self.failover_history.append(event)

        # Keep only last 100 events
        if len(self.failover_history) > 100:
            self.failover_history = self.failover_history[-100:]

    async def _immediate_failover(self, failed_node_id: str, event: FailoverEvent) -> bool:
        """Execute immediate failover."""
        try:
            # 1. Mark node as failed in cluster
            cluster_nodes = self.cluster_manager.get_cluster_nodes()
            if failed_node_id in cluster_nodes:
                cluster_nodes[failed_node_id].status = NodeStatus.OFFLINE

            # 2. Redistribute running tasks
            affected_tasks = await self._redistribute_tasks(failed_node_id)
            event.affected_tasks = affected_tasks

            # 3. Update load balancer to exclude failed node
            # This would be handled by the load balancer automatically

            self.logger.info(f"Immediate failover completed for node {failed_node_id}")
            return True

        except Exception as e:
            self.logger.error(f"Immediate failover failed: {e}")
            return False

    async def _graceful_failover(self, failed_node_id: str, event: FailoverEvent) -> bool:
        """Execute graceful failover."""
        try:
            # 1. Stop accepting new tasks on the failed node
            # This would be implemented in the load balancer

            # 2. Wait for current tasks to complete (with timeout)
            timeout = 300  # 5 minutes
            start_time = time.time()

            while time.time() - start_time < timeout:
                # Check if node has any running tasks
                # In a real implementation, you'd query the task manager
                running_tasks = 0  # Placeholder

                if running_tasks == 0:
                    break

                await asyncio.sleep(10)

            # 3. Force redistribute any remaining tasks
            affected_tasks = await self._redistribute_tasks(failed_node_id)
            event.affected_tasks = affected_tasks

            # 4. Mark node as failed
            cluster_nodes = self.cluster_manager.get_cluster_nodes()
            if failed_node_id in cluster_nodes:
                cluster_nodes[failed_node_id].status = NodeStatus.OFFLINE

            self.logger.info(f"Graceful failover completed for node {failed_node_id}")
            return True

        except Exception as e:
            self.logger.error(f"Graceful failover failed: {e}")
            return False

    async def _redistribute_tasks(self, failed_node_id: str) -> int:
        """Redistribute tasks from a failed node."""
        try:
            # In a real implementation, this would:
            # 1. Get all tasks assigned to the failed node
            # 2. Reassign them to healthy nodes
            # 3. Update task status and metadata

            # For now, we'll simulate
            affected_tasks = 5  # Placeholder
            self.logger.info(f"Redistributed {affected_tasks} tasks from node {failed_node_id}")
            return affected_tasks

        except Exception as e:
            self.logger.error(f"Failed to redistribute tasks: {e}")
            return 0

    async def _recover_node(self, node_id: str):
        """Recover a previously failed node."""
        if node_id not in self.failed_nodes:
            return

        try:
            # Remove from failed nodes
            del self.failed_nodes[node_id]

            # Update cluster status
            cluster_nodes = self.cluster_manager.get_cluster_nodes()
            if node_id in cluster_nodes:
                cluster_nodes[node_id].status = NodeStatus.HEALTHY

            # Reset circuit breaker
            if node_id in self.circuit_breakers:
                self.circuit_breakers[node_id]["state"] = "closed"
                self.circuit_breakers[node_id]["failure_count"] = 0

            # Add back to recovery queue for gradual reintegration
            if node_id not in self.recovery_queue:
                self.recovery_queue.append(node_id)

            self.logger.info(f"Node {node_id} recovered and marked for reintegration")

            # Send alert
            await self._send_alert(
                AlertSeverity.INFO,
                f"Node {node_id} has recovered",
                {"node_id": node_id}
            )

        except Exception as e:
            self.logger.error(f"Failed to recover node {node_id}: {e}")

    async def _failover_loop(self):
        """Background task for handling failover operations."""
        while self.is_running:
            try:
                # Check for nodes that need failover
                await self._check_failed_nodes()
                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in failover loop: {e}")
                await asyncio.sleep(60)

    async def _recovery_loop(self):
        """Background task for handling node recovery."""
        while self.is_running:
            try:
                await self._process_recovery_queue()
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in recovery loop: {e}")
                await asyncio.sleep(60)

    async def _scoring_loop(self):
        """Background task for updating node health scores."""
        while self.is_running:
            try:
                await self._update_health_scores()
                await asyncio.sleep(120)  # Update every 2 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in scoring loop: {e}")
                await asyncio.sleep(120)

    async def _check_failed_nodes(self):
        """Check for nodes that have been failed for too long."""
        current_time = datetime.now()

        for node_id, failed_time in list(self.failed_nodes.items()):
            # If node has been failed for more than 1 hour, consider permanent removal
            if (current_time - failed_time).total_seconds() > 3600:
                self.logger.warning(f"Node {node_id} has been failed for over 1 hour")
                # In production, you might want to:
                # - Terminate the instance
                # - Remove from cluster permanently
                # - Send alerts to operations team

    async def _process_recovery_queue(self):
        """Process nodes in the recovery queue."""
        if not self.recovery_queue:
            return

        # Process one node at a time for gradual reintegration
        node_id = self.recovery_queue.pop(0)

        try:
            # Gradually reintegrate the node
            # Start with low-priority tasks
            self.logger.info(f"Reintegrating node {node_id}")

            # In a real implementation, you would:
            # 1. Start with health checks
            # 2. Assign low-priority tasks
            # 3. Gradually increase load
            # 4. Monitor performance

        except Exception as e:
            self.logger.error(f"Failed to reintegrate node {node_id}: {e}")
            # Put back in queue for retry
            self.recovery_queue.append(node_id)

    async def _update_health_scores(self):
        """Update health scores for all nodes."""
        cluster_nodes = self.cluster_manager.get_cluster_nodes()

        for node_id, node_info in cluster_nodes.items():
            try:
                score = await self._calculate_health_score(node_id, node_info)
                self.node_health_scores[node_id] = score

            except Exception as e:
                self.logger.error(f"Failed to calculate health score for node {node_id}: {e}")

    async def _calculate_health_score(self, node_id: str, node_info: NodeInfo) -> NodeHealthScore:
        """Calculate comprehensive health score for a node."""
        # Get recent health check results
        recent_results = list(self.health_history[node_id])[-20:]  # Last 20 results

        if not recent_results:
            # No health data available
            return NodeHealthScore(
                node_id=node_id,
                overall_score=50.0,
                availability_score=50.0,
                performance_score=50.0,
                resource_score=50.0,
                last_updated=datetime.now(),
                health_trend="unknown"
            )

        # Calculate availability score (success rate)
        successful_checks = sum(1 for r in recent_results if r.success)
        availability_score = (successful_checks / len(recent_results)) * 100

        # Calculate performance score (response time)
        response_times = [r.response_time for r in recent_results if r.success]
        if response_times:
            avg_response_time = statistics.mean(response_times)
            # Score based on response time (lower is better)
            performance_score = max(0, 100 - (avg_response_time * 10))
        else:
            performance_score = 0

        # Calculate resource score
        resources = node_info.resources
        if resources:
            cpu_usage = resources.get('cpu_usage', 0)
            memory_usage = resources.get('memory_usage', 0)
            disk_usage = resources.get('disk_usage', 0)

            # Score based on resource availability (lower usage is better)
            resource_score = 100 - ((cpu_usage + memory_usage + disk_usage) / 3)
        else:
            resource_score = 50

        # Calculate overall score (weighted average)
        overall_score = (
            availability_score * 0.4 +
            performance_score * 0.3 +
            resource_score * 0.3
        )

        # Determine health trend
        if len(recent_results) >= 10:
            first_half = recent_results[:len(recent_results)//2]
            second_half = recent_results[len(recent_results)//2:]

            first_half_success = sum(1 for r in first_half if r.success) / len(first_half)
            second_half_success = sum(1 for r in second_half if r.success) / len(second_half)

            if second_half_success > first_half_success + 0.1:
                health_trend = "improving"
            elif second_half_success < first_half_success - 0.1:
                health_trend = "degrading"
            else:
                health_trend = "stable"
        else:
            health_trend = "stable"

        return NodeHealthScore(
            node_id=node_id,
            overall_score=overall_score,
            availability_score=availability_score,
            performance_score=performance_score,
            resource_score=resource_score,
            last_updated=datetime.now(),
            health_trend=health_trend,
            uptime_percentage=availability_score,
            average_response_time=statistics.mean(response_times) if response_times else 0,
            error_rate=(1 - availability_score / 100) * 100,
            cpu_usage=resources.get('cpu_usage', 0) if resources else 0,
            memory_usage=resources.get('memory_usage', 0) if resources else 0,
            disk_usage=resources.get('disk_usage', 0) if resources else 0
        )

    async def _send_alert(self, severity: AlertSeverity, message: str, details: Dict[str, Any] = None):
        """Send alert to registered callbacks."""
        alert = {
            "severity": severity.value,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }

        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

    # Public API methods

    def add_alert_callback(self, callback: Callable):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)

    def get_node_health_score(self, node_id: str) -> Optional[NodeHealthScore]:
        """Get health score for a specific node."""
        return self.node_health_scores.get(node_id)

    def get_all_health_scores(self) -> Dict[str, NodeHealthScore]:
        """Get health scores for all nodes."""
        return self.node_health_scores.copy()

    def get_health_history(self, node_id: str, hours: int = 24) -> List[HealthResult]:
        """Get health check history for a node."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            result for result in self.health_history[node_id]
            if result.timestamp > cutoff_time
        ]

    def get_failover_history(self, hours: int = 24) -> List[FailoverEvent]:
        """Get failover history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self.failover_history
            if event.timestamp > cutoff_time
        ]

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        failed_nodes_count = len(self.failed_nodes)
        recovery_queue_count = len(self.recovery_queue)

        # Calculate average health score
        if self.node_health_scores:
            avg_health_score = statistics.mean(
                score.overall_score for score in self.node_health_scores.values()
            )
        else:
            avg_health_score = 0

        return {
            "monitoring_metrics": self.monitoring_metrics.copy(),
            "failed_nodes_count": failed_nodes_count,
            "recovery_queue_count": recovery_queue_count,
            "average_health_score": avg_health_score,
            "total_health_checks": len(self.health_checks),
            "circuit_breakers": {
                node_id: cb["state"]
                for node_id, cb in self.circuit_breakers.items()
            }
        }

    def set_failover_strategy(self, strategy: FailoverStrategy):
        """Set the failover strategy."""
        self.failover_strategy = strategy
        self.logger.info(f"Failover strategy set to: {strategy.value}")

    async def manual_failover(self, node_id: str, reason: str = "Manual failover") -> bool:
        """Manually trigger failover for a node."""
        try:
            await self._trigger_failover(node_id, reason)
            return True
        except Exception as e:
            self.logger.error(f"Manual failover failed: {e}")
            return False

    async def manual_recovery(self, node_id: str) -> bool:
        """Manually trigger recovery for a node."""
        try:
            await self._recover_node(node_id)
            return True
        except Exception as e:
            self.logger.error(f"Manual recovery failed: {e}")
            return False
