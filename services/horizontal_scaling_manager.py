"""
Horizontal Scaling and Load Balancing System for Phase 3: Enterprise & Scalability Features.

This module provides auto-scaling mechanisms, load balancing algorithms,
dynamic resource allocation, and scaling policies with triggers.
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import statistics

from services.cluster_manager import ClusterManager, NodeInfo, NodeStatus, NodeRole
from services.advanced_queue_manager import AdvancedQueueManager
from services.monitoring_service import MonitoringService


class ScalingDirection(Enum):
    """Scaling direction enumeration."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithm enumeration."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_BASED = "resource_based"
    GEOGRAPHIC = "geographic"


@dataclass
class ScalingPolicy:
    """Scaling policy configuration."""
    policy_id: str
    name: str
    metric_name: str  # cpu_usage, memory_usage, queue_length, response_time
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    min_nodes: int = 1
    max_nodes: int = 10
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    evaluation_periods: int = 3
    enabled: bool = True


@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""
    algorithm: LoadBalancingAlgorithm
    health_check_interval: int = 30
    health_check_timeout: int = 10
    max_retries: int = 3
    sticky_sessions: bool = False
    session_timeout: int = 3600
    weights: Dict[str, float] = None


@dataclass
class ScalingEvent:
    """Scaling event record."""
    event_id: str
    timestamp: datetime
    direction: ScalingDirection
    trigger_metric: str
    trigger_value: float
    threshold: float
    nodes_before: int
    nodes_after: int
    policy_id: str
    success: bool
    reason: str


class HorizontalScalingManager:
    """
    Horizontal scaling and load balancing manager providing auto-scaling,
    intelligent load distribution, and dynamic resource allocation.
    """
    
    def __init__(
        self,
        cluster_manager: ClusterManager,
        queue_manager: AdvancedQueueManager,
        monitoring_service: MonitoringService
    ):
        self.cluster_manager = cluster_manager
        self.queue_manager = queue_manager
        self.monitoring_service = monitoring_service
        self.logger = logging.getLogger("horizontal_scaling_manager")
        
        # Scaling configuration
        self.scaling_policies: Dict[str, ScalingPolicy] = {}
        self.load_balancer_config = LoadBalancerConfig(
            algorithm=LoadBalancingAlgorithm.LEAST_LOADED
        )
        
        # Scaling state
        self.last_scaling_action: Dict[str, datetime] = {}
        self.scaling_history: List[ScalingEvent] = []
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Load balancing state
        self.node_connections: Dict[str, int] = defaultdict(int)
        self.node_weights: Dict[str, float] = {}
        self.round_robin_index = 0
        self.sticky_sessions: Dict[str, str] = {}  # session_id -> node_id
        
        # Background tasks
        self.scaling_task: Optional[asyncio.Task] = None
        self.load_balancing_task: Optional[asyncio.Task] = None
        self.metrics_collection_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "requests_per_second": 0.0
        }
        
        self.logger.info("Horizontal scaling manager initialized")
    
    async def start(self):
        """Start the horizontal scaling manager."""
        if not self.is_running:
            self.is_running = True
            
            # Create default scaling policies
            await self._create_default_policies()
            
            # Start background tasks
            self.scaling_task = asyncio.create_task(self._scaling_loop())
            self.load_balancing_task = asyncio.create_task(self._load_balancing_loop())
            self.metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
            
            self.logger.info("Horizontal scaling manager started")
    
    async def stop(self):
        """Stop the horizontal scaling manager."""
        if self.is_running:
            self.is_running = False
            
            # Stop background tasks
            tasks = [self.scaling_task, self.load_balancing_task, self.metrics_collection_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.logger.info("Horizontal scaling manager stopped")
    
    async def _create_default_policies(self):
        """Create default scaling policies."""
        # CPU-based scaling policy
        cpu_policy = ScalingPolicy(
            policy_id="cpu_scaling",
            name="CPU Usage Scaling",
            metric_name="cpu_usage",
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            scale_up_cooldown=300,
            scale_down_cooldown=600,
            min_nodes=2,
            max_nodes=20,
            evaluation_periods=3
        )
        
        # Memory-based scaling policy
        memory_policy = ScalingPolicy(
            policy_id="memory_scaling",
            name="Memory Usage Scaling",
            metric_name="memory_usage",
            scale_up_threshold=85.0,
            scale_down_threshold=40.0,
            scale_up_cooldown=300,
            scale_down_cooldown=600,
            min_nodes=2,
            max_nodes=15,
            evaluation_periods=3
        )
        
        # Queue length-based scaling policy
        queue_policy = ScalingPolicy(
            policy_id="queue_scaling",
            name="Queue Length Scaling",
            metric_name="queue_length",
            scale_up_threshold=50.0,
            scale_down_threshold=10.0,
            scale_up_cooldown=180,
            scale_down_cooldown=300,
            min_nodes=1,
            max_nodes=25,
            evaluation_periods=2
        )
        
        self.scaling_policies["cpu_scaling"] = cpu_policy
        self.scaling_policies["memory_scaling"] = memory_policy
        self.scaling_policies["queue_scaling"] = queue_policy
        
        self.logger.info("Created default scaling policies")
    
    async def add_scaling_policy(self, policy: ScalingPolicy) -> bool:
        """Add a new scaling policy."""
        try:
            self.scaling_policies[policy.policy_id] = policy
            self.logger.info(f"Added scaling policy: {policy.policy_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add scaling policy: {e}")
            return False
    
    async def remove_scaling_policy(self, policy_id: str) -> bool:
        """Remove a scaling policy."""
        try:
            if policy_id in self.scaling_policies:
                del self.scaling_policies[policy_id]
                self.logger.info(f"Removed scaling policy: {policy_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to remove scaling policy: {e}")
            return False
    
    async def _scaling_loop(self):
        """Main scaling evaluation loop."""
        while self.is_running:
            try:
                await self._evaluate_scaling_policies()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(60)
    
    async def _load_balancing_loop(self):
        """Load balancing maintenance loop."""
        while self.is_running:
            try:
                await self._update_node_weights()
                await self._cleanup_sticky_sessions()
                await asyncio.sleep(60)  # Update every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in load balancing loop: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collection_loop(self):
        """Metrics collection loop."""
        while self.is_running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(15)  # Collect every 15 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(30)
    
    async def _collect_metrics(self):
        """Collect metrics for scaling decisions."""
        try:
            # Get cluster nodes
            healthy_nodes = self.cluster_manager.get_healthy_nodes()
            
            if not healthy_nodes:
                return
            
            # Collect CPU usage
            cpu_values = []
            memory_values = []
            
            for node_id, node_info in healthy_nodes.items():
                resources = node_info.resources
                if resources:
                    cpu_usage = resources.get('cpu_usage', 0)
                    memory_usage = resources.get('memory_usage', 0)
                    
                    cpu_values.append(cpu_usage)
                    memory_values.append(memory_usage)
            
            # Calculate averages
            if cpu_values:
                avg_cpu = statistics.mean(cpu_values)
                self.metric_history['cpu_usage'].append(avg_cpu)
            
            if memory_values:
                avg_memory = statistics.mean(memory_values)
                self.metric_history['memory_usage'].append(avg_memory)
            
            # Get queue metrics
            queue_stats = self.queue_manager.get_queue_statistics()
            total_pending = queue_stats['global']['total_pending']
            self.metric_history['queue_length'].append(total_pending)
            
            # Calculate response time (mock for now)
            response_time = self.performance_metrics['average_response_time']
            self.metric_history['response_time'].append(response_time)
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")

    async def _evaluate_scaling_policies(self):
        """Evaluate all scaling policies and trigger scaling actions."""
        for policy_id, policy in self.scaling_policies.items():
            if not policy.enabled:
                continue

            try:
                await self._evaluate_single_policy(policy)
            except Exception as e:
                self.logger.error(f"Error evaluating policy {policy_id}: {e}")

    async def _evaluate_single_policy(self, policy: ScalingPolicy):
        """Evaluate a single scaling policy."""
        # Get recent metric values
        metric_values = list(self.metric_history.get(policy.metric_name, []))

        if len(metric_values) < policy.evaluation_periods:
            return  # Not enough data

        # Get the last N values for evaluation
        recent_values = metric_values[-policy.evaluation_periods:]
        avg_value = statistics.mean(recent_values)

        # Get current node count
        healthy_nodes = self.cluster_manager.get_healthy_nodes()
        current_nodes = len(healthy_nodes)

        # Check cooldown periods
        last_action_time = self.last_scaling_action.get(policy.policy_id)
        current_time = datetime.now()

        scaling_decision = None

        # Evaluate scale up
        if avg_value > policy.scale_up_threshold and current_nodes < policy.max_nodes:
            if not last_action_time or (current_time - last_action_time).total_seconds() > policy.scale_up_cooldown:
                scaling_decision = ScalingDirection.UP

        # Evaluate scale down
        elif avg_value < policy.scale_down_threshold and current_nodes > policy.min_nodes:
            if not last_action_time or (current_time - last_action_time).total_seconds() > policy.scale_down_cooldown:
                scaling_decision = ScalingDirection.DOWN

        # Execute scaling action
        if scaling_decision:
            await self._execute_scaling_action(policy, scaling_decision, avg_value, current_nodes)

    async def _execute_scaling_action(
        self,
        policy: ScalingPolicy,
        direction: ScalingDirection,
        trigger_value: float,
        current_nodes: int
    ):
        """Execute a scaling action."""
        try:
            success = False
            new_node_count = current_nodes
            reason = ""

            if direction == ScalingDirection.UP:
                # Scale up
                target_nodes = min(current_nodes + policy.scale_up_increment, policy.max_nodes)
                nodes_to_add = target_nodes - current_nodes

                if nodes_to_add > 0:
                    success = await self._scale_up(nodes_to_add)
                    new_node_count = target_nodes if success else current_nodes
                    reason = f"Added {nodes_to_add} nodes due to high {policy.metric_name}"

            elif direction == ScalingDirection.DOWN:
                # Scale down
                target_nodes = max(current_nodes - policy.scale_down_increment, policy.min_nodes)
                nodes_to_remove = current_nodes - target_nodes

                if nodes_to_remove > 0:
                    success = await self._scale_down(nodes_to_remove)
                    new_node_count = target_nodes if success else current_nodes
                    reason = f"Removed {nodes_to_remove} nodes due to low {policy.metric_name}"

            # Record scaling event
            event = ScalingEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                direction=direction,
                trigger_metric=policy.metric_name,
                trigger_value=trigger_value,
                threshold=policy.scale_up_threshold if direction == ScalingDirection.UP else policy.scale_down_threshold,
                nodes_before=current_nodes,
                nodes_after=new_node_count,
                policy_id=policy.policy_id,
                success=success,
                reason=reason
            )

            self.scaling_history.append(event)
            self.last_scaling_action[policy.policy_id] = datetime.now()

            # Keep only last 100 events
            if len(self.scaling_history) > 100:
                self.scaling_history = self.scaling_history[-100:]

            if success:
                self.logger.info(f"Scaling action completed: {reason}")
            else:
                self.logger.warning(f"Scaling action failed: {reason}")

        except Exception as e:
            self.logger.error(f"Failed to execute scaling action: {e}")

    async def _scale_up(self, nodes_to_add: int) -> bool:
        """Scale up by adding new nodes."""
        try:
            # In a real implementation, this would:
            # 1. Launch new instances (EC2, GKE, etc.)
            # 2. Configure and deploy the application
            # 3. Register nodes with the cluster

            # For now, we'll simulate by logging
            self.logger.info(f"Scaling up: adding {nodes_to_add} nodes")

            # Simulate successful scaling
            return True

        except Exception as e:
            self.logger.error(f"Failed to scale up: {e}")
            return False

    async def _scale_down(self, nodes_to_remove: int) -> bool:
        """Scale down by removing nodes."""
        try:
            # Get nodes that can be safely removed
            healthy_nodes = self.cluster_manager.get_healthy_nodes()
            worker_nodes = [
                node_id for node_id, node_info in healthy_nodes.items()
                if node_info.role == NodeRole.WORKER
            ]

            if len(worker_nodes) < nodes_to_remove:
                self.logger.warning("Not enough worker nodes to remove")
                return False

            # Select nodes to remove (prefer least loaded)
            nodes_to_remove_list = await self._select_nodes_for_removal(worker_nodes, nodes_to_remove)

            # In a real implementation, this would:
            # 1. Drain tasks from selected nodes
            # 2. Gracefully shutdown nodes
            # 3. Terminate instances
            # 4. Update cluster state

            self.logger.info(f"Scaling down: removing {len(nodes_to_remove_list)} nodes")

            # Simulate successful scaling
            return True

        except Exception as e:
            self.logger.error(f"Failed to scale down: {e}")
            return False

    async def _select_nodes_for_removal(self, candidate_nodes: List[str], count: int) -> List[str]:
        """Select nodes for removal based on load and other factors."""
        # Get node loads
        node_loads = {}
        for node_id in candidate_nodes:
            # Calculate load based on connections and resource usage
            connections = self.node_connections.get(node_id, 0)

            # Get resource usage from cluster manager
            cluster_nodes = self.cluster_manager.get_cluster_nodes()
            node_info = cluster_nodes.get(node_id)

            if node_info and node_info.resources:
                cpu_usage = node_info.resources.get('cpu_usage', 0)
                memory_usage = node_info.resources.get('memory_usage', 0)
                load_score = (cpu_usage + memory_usage) / 2 + connections * 10
            else:
                load_score = connections * 10

            node_loads[node_id] = load_score

        # Sort by load (ascending) and select least loaded nodes
        sorted_nodes = sorted(node_loads.items(), key=lambda x: x[1])
        return [node_id for node_id, _ in sorted_nodes[:count]]

    # Load Balancing Methods

    async def select_node(self, session_id: str = None, request_metadata: Dict[str, Any] = None) -> Optional[str]:
        """Select a node for load balancing."""
        healthy_nodes = self.cluster_manager.get_healthy_nodes()

        if not healthy_nodes:
            return None

        # Check sticky sessions
        if session_id and self.load_balancer_config.sticky_sessions:
            if session_id in self.sticky_sessions:
                node_id = self.sticky_sessions[session_id]
                if node_id in healthy_nodes:
                    return node_id
                else:
                    # Remove invalid session
                    del self.sticky_sessions[session_id]

        # Select node based on algorithm
        algorithm = self.load_balancer_config.algorithm

        if algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            selected_node = await self._round_robin_selection(list(healthy_nodes.keys()))

        elif algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            selected_node = await self._least_connections_selection(list(healthy_nodes.keys()))

        elif algorithm == LoadBalancingAlgorithm.LEAST_LOADED:
            selected_node = await self._least_loaded_selection(healthy_nodes)

        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            selected_node = await self._weighted_round_robin_selection(list(healthy_nodes.keys()))

        elif algorithm == LoadBalancingAlgorithm.RESOURCE_BASED:
            selected_node = await self._resource_based_selection(healthy_nodes)

        else:
            # Default to round robin
            selected_node = await self._round_robin_selection(list(healthy_nodes.keys()))

        # Update sticky session if enabled
        if selected_node and session_id and self.load_balancer_config.sticky_sessions:
            self.sticky_sessions[session_id] = selected_node

        # Update connection count
        if selected_node:
            self.node_connections[selected_node] += 1

        return selected_node

    async def _round_robin_selection(self, node_ids: List[str]) -> str:
        """Round robin node selection."""
        if not node_ids:
            return None

        selected_node = node_ids[self.round_robin_index % len(node_ids)]
        self.round_robin_index += 1
        return selected_node

    async def _least_connections_selection(self, node_ids: List[str]) -> str:
        """Least connections node selection."""
        if not node_ids:
            return None

        # Find node with least connections
        min_connections = float('inf')
        selected_node = node_ids[0]

        for node_id in node_ids:
            connections = self.node_connections.get(node_id, 0)
            if connections < min_connections:
                min_connections = connections
                selected_node = node_id

        return selected_node

    async def _least_loaded_selection(self, healthy_nodes: Dict[str, NodeInfo]) -> str:
        """Least loaded node selection based on resource usage."""
        if not healthy_nodes:
            return None

        min_load = float('inf')
        selected_node = list(healthy_nodes.keys())[0]

        for node_id, node_info in healthy_nodes.items():
            # Calculate load score
            resources = node_info.resources
            if resources:
                cpu_usage = resources.get('cpu_usage', 0)
                memory_usage = resources.get('memory_usage', 0)
                load_score = (cpu_usage + memory_usage) / 2
            else:
                load_score = 0

            # Add connection factor
            connections = self.node_connections.get(node_id, 0)
            load_score += connections * 5  # Weight connections

            if load_score < min_load:
                min_load = load_score
                selected_node = node_id

        return selected_node

    async def _weighted_round_robin_selection(self, node_ids: List[str]) -> str:
        """Weighted round robin node selection."""
        if not node_ids:
            return None

        # Use weights if available, otherwise equal weights
        weighted_nodes = []
        for node_id in node_ids:
            weight = self.node_weights.get(node_id, 1.0)
            weighted_nodes.extend([node_id] * int(weight * 10))  # Scale weights

        if not weighted_nodes:
            return node_ids[0]

        selected_node = weighted_nodes[self.round_robin_index % len(weighted_nodes)]
        self.round_robin_index += 1
        return selected_node

    async def _resource_based_selection(self, healthy_nodes: Dict[str, NodeInfo]) -> str:
        """Resource-based node selection considering available resources."""
        if not healthy_nodes:
            return None

        best_score = -1
        selected_node = list(healthy_nodes.keys())[0]

        for node_id, node_info in healthy_nodes.items():
            resources = node_info.resources
            if not resources:
                continue

            # Calculate available resource score
            cpu_available = 100 - resources.get('cpu_usage', 100)
            memory_available = 100 - resources.get('memory_usage', 100)

            # Consider node capabilities
            capability_bonus = len(node_info.capabilities) * 5

            # Calculate total score
            score = (cpu_available + memory_available) / 2 + capability_bonus

            if score > best_score:
                best_score = score
                selected_node = node_id

        return selected_node

    async def release_connection(self, node_id: str):
        """Release a connection from a node."""
        if node_id in self.node_connections:
            self.node_connections[node_id] = max(0, self.node_connections[node_id] - 1)

    async def _update_node_weights(self):
        """Update node weights based on performance."""
        healthy_nodes = self.cluster_manager.get_healthy_nodes()

        for node_id, node_info in healthy_nodes.items():
            # Calculate weight based on resources and performance
            resources = node_info.resources
            if resources:
                cpu_cores = resources.get('cpu_cores', 1)
                memory_total = resources.get('memory_total', 1024) / 1024  # Convert to GB

                # Base weight on hardware capacity
                weight = (cpu_cores + memory_total / 4) / 2

                # Adjust based on current load
                cpu_usage = resources.get('cpu_usage', 0)
                memory_usage = resources.get('memory_usage', 0)
                load_factor = 1 - ((cpu_usage + memory_usage) / 200)  # 0-1 scale

                weight *= max(0.1, load_factor)  # Minimum weight of 0.1
            else:
                weight = 1.0

            self.node_weights[node_id] = weight

    async def _cleanup_sticky_sessions(self):
        """Clean up expired sticky sessions."""
        if not self.load_balancer_config.sticky_sessions:
            return

        current_time = datetime.now()
        expired_sessions = []

        # In a real implementation, you'd track session timestamps
        # For now, we'll just clean up sessions for offline nodes
        healthy_nodes = self.cluster_manager.get_healthy_nodes()

        for session_id, node_id in self.sticky_sessions.items():
            if node_id not in healthy_nodes:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sticky_sessions[session_id]

    # Public API methods

    def get_scaling_policies(self) -> Dict[str, ScalingPolicy]:
        """Get all scaling policies."""
        return self.scaling_policies.copy()

    def get_scaling_history(self, hours: int = 24) -> List[ScalingEvent]:
        """Get scaling history for the specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self.scaling_history
            if event.timestamp > cutoff_time
        ]

    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        healthy_nodes = self.cluster_manager.get_healthy_nodes()

        node_stats = {}
        for node_id in healthy_nodes.keys():
            node_stats[node_id] = {
                "connections": self.node_connections.get(node_id, 0),
                "weight": self.node_weights.get(node_id, 1.0)
            }

        return {
            "algorithm": self.load_balancer_config.algorithm.value,
            "total_nodes": len(healthy_nodes),
            "total_connections": sum(self.node_connections.values()),
            "sticky_sessions": len(self.sticky_sessions),
            "node_stats": node_stats,
            "performance_metrics": self.performance_metrics.copy()
        }

    def get_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """Get scaling recommendations based on current metrics."""
        recommendations = []

        for policy_id, policy in self.scaling_policies.items():
            if not policy.enabled:
                continue

            metric_values = list(self.metric_history.get(policy.metric_name, []))
            if len(metric_values) < policy.evaluation_periods:
                continue

            recent_values = metric_values[-policy.evaluation_periods:]
            avg_value = statistics.mean(recent_values)

            healthy_nodes = self.cluster_manager.get_healthy_nodes()
            current_nodes = len(healthy_nodes)

            recommendation = {
                "policy_id": policy_id,
                "policy_name": policy.name,
                "metric_name": policy.metric_name,
                "current_value": avg_value,
                "scale_up_threshold": policy.scale_up_threshold,
                "scale_down_threshold": policy.scale_down_threshold,
                "current_nodes": current_nodes,
                "min_nodes": policy.min_nodes,
                "max_nodes": policy.max_nodes,
                "recommendation": "stable"
            }

            if avg_value > policy.scale_up_threshold and current_nodes < policy.max_nodes:
                recommendation["recommendation"] = "scale_up"
                recommendation["suggested_nodes"] = min(
                    current_nodes + policy.scale_up_increment,
                    policy.max_nodes
                )
            elif avg_value < policy.scale_down_threshold and current_nodes > policy.min_nodes:
                recommendation["recommendation"] = "scale_down"
                recommendation["suggested_nodes"] = max(
                    current_nodes - policy.scale_down_increment,
                    policy.min_nodes
                )
            else:
                recommendation["suggested_nodes"] = current_nodes

            recommendations.append(recommendation)

        return recommendations

    def update_load_balancer_config(self, config_updates: Dict[str, Any]):
        """Update load balancer configuration."""
        for key, value in config_updates.items():
            if hasattr(self.load_balancer_config, key):
                if key == 'algorithm':
                    self.load_balancer_config.algorithm = LoadBalancingAlgorithm(value)
                else:
                    setattr(self.load_balancer_config, key, value)

        self.logger.info(f"Updated load balancer configuration: {config_updates}")

    def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics."""
        self.performance_metrics.update(metrics)

    def force_scaling_action(self, direction: ScalingDirection, node_count: int) -> bool:
        """Force a scaling action (for manual scaling)."""
        try:
            current_nodes = len(self.cluster_manager.get_healthy_nodes())

            if direction == ScalingDirection.UP:
                nodes_to_add = node_count
                return asyncio.create_task(self._scale_up(nodes_to_add))
            elif direction == ScalingDirection.DOWN:
                nodes_to_remove = node_count
                return asyncio.create_task(self._scale_down(nodes_to_remove))

            return False

        except Exception as e:
            self.logger.error(f"Failed to force scaling action: {e}")
            return False
