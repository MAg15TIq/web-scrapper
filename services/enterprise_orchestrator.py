"""
Enterprise Orchestrator for Phase 3: Enterprise & Scalability Features.

This module integrates all Phase 3 components into a unified enterprise-grade
distributed scraping system with cluster management, advanced queuing,
multi-tenancy, horizontal scaling, and enhanced monitoring.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from services.cluster_manager import ClusterManager, ClusterConfig, NodeRole
from services.advanced_queue_manager import AdvancedQueueManager, QueueType, ResourceRequirement
from services.multi_tenant_manager import MultiTenantManager, BillingModel, ResourceQuota
from services.horizontal_scaling_manager import HorizontalScalingManager, LoadBalancingAlgorithm
from services.enhanced_health_monitor import EnhancedHealthMonitor, FailoverStrategy
from services.redis_service import RedisService
from services.database_service import DatabaseService
from services.monitoring_service import MonitoringService


@dataclass
class EnterpriseConfig:
    """Enterprise system configuration."""
    cluster_id: str
    cluster_name: str
    
    # Cluster settings
    max_nodes: int = 50
    auto_scaling: bool = True
    load_balancing: bool = True
    failover_enabled: bool = True
    
    # Multi-tenancy settings
    multi_tenant_enabled: bool = True
    default_billing_model: BillingModel = BillingModel.FREEMIUM
    
    # Scaling settings
    scaling_enabled: bool = True
    load_balancing_algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.LEAST_LOADED
    
    # Health monitoring settings
    health_monitoring_enabled: bool = True
    failover_strategy: FailoverStrategy = FailoverStrategy.GRACEFUL
    
    # Performance settings
    max_concurrent_jobs_per_node: int = 10
    default_job_timeout: int = 3600
    queue_cleanup_interval: int = 300


class EnterpriseOrchestrator:
    """
    Enterprise orchestrator that coordinates all Phase 3 components to provide
    a comprehensive distributed scraping platform with enterprise features.
    """
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.logger = logging.getLogger("enterprise_orchestrator")
        
        # Initialize core services
        self.redis_service = RedisService()
        self.database_service = DatabaseService()
        self.monitoring_service = MonitoringService()
        
        # Initialize cluster configuration
        cluster_config = ClusterConfig(
            cluster_id=config.cluster_id,
            name=config.cluster_name,
            discovery_method="redis",  # Use Redis for service discovery
            max_nodes=config.max_nodes,
            auto_scaling=config.auto_scaling,
            load_balancing=config.load_balancing,
            failover_enabled=config.failover_enabled
        )
        
        # Initialize Phase 3 components
        self.cluster_manager = ClusterManager(cluster_config)
        self.queue_manager = AdvancedQueueManager(self.redis_service)
        self.multi_tenant_manager = MultiTenantManager(self.redis_service, self.database_service)
        self.scaling_manager = HorizontalScalingManager(
            self.cluster_manager, 
            self.queue_manager, 
            self.monitoring_service
        )
        self.health_monitor = EnhancedHealthMonitor(
            self.cluster_manager,
            self.queue_manager,
            self.redis_service
        )
        
        # System state
        self.is_running = False
        self.startup_time: Optional[datetime] = None
        
        # Performance metrics
        self.system_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_tenants": 0,
            "active_nodes": 0,
            "total_jobs_processed": 0
        }
        
        # Event handlers
        self._setup_event_handlers()
        
        self.logger.info(f"Enterprise orchestrator initialized for cluster: {config.cluster_id}")
    
    def _setup_event_handlers(self):
        """Setup event handlers between components."""
        # Cluster events
        self.cluster_manager.on_node_joined(self._on_node_joined)
        self.cluster_manager.on_node_left(self._on_node_left)
        self.cluster_manager.on_leader_changed(self._on_leader_changed)
        
        # Health monitoring alerts
        self.health_monitor.add_alert_callback(self._on_health_alert)
    
    async def start(self) -> bool:
        """Start the enterprise orchestrator and all components."""
        try:
            if self.is_running:
                self.logger.warning("Enterprise orchestrator is already running")
                return True
            
            self.logger.info("Starting enterprise orchestrator...")
            self.startup_time = datetime.now()
            
            # Start core services
            await self.redis_service.connect()
            await self.database_service.connect()
            await self.monitoring_service.start()
            
            # Start Phase 3 components in order
            await self.queue_manager.start()
            
            if self.config.multi_tenant_enabled:
                await self.multi_tenant_manager.start()
            
            # Join cluster as coordinator node
            success = await self.cluster_manager.join_cluster(
                role=NodeRole.COORDINATOR,
                capabilities=["orchestration", "coordination", "monitoring"],
                metadata={"version": "3.0.0", "features": ["enterprise", "scaling", "multi-tenant"]}
            )
            
            if not success:
                raise Exception("Failed to join cluster")
            
            if self.config.scaling_enabled:
                await self.scaling_manager.start()
            
            if self.config.health_monitoring_enabled:
                await self.health_monitor.start()
            
            # Configure components
            await self._configure_components()
            
            self.is_running = True
            
            # Log startup summary
            startup_time = (datetime.now() - self.startup_time).total_seconds()
            self.logger.info(f"Enterprise orchestrator started successfully in {startup_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start enterprise orchestrator: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> bool:
        """Stop the enterprise orchestrator and all components."""
        try:
            if not self.is_running:
                return True
            
            self.logger.info("Stopping enterprise orchestrator...")
            
            # Stop components in reverse order
            if self.config.health_monitoring_enabled:
                await self.health_monitor.stop()
            
            if self.config.scaling_enabled:
                await self.scaling_manager.stop()
            
            await self.cluster_manager.leave_cluster()
            
            if self.config.multi_tenant_enabled:
                await self.multi_tenant_manager.stop()
            
            await self.queue_manager.stop()
            
            # Stop core services
            await self.monitoring_service.stop()
            await self.database_service.disconnect()
            await self.redis_service.disconnect()
            
            self.is_running = False
            self.logger.info("Enterprise orchestrator stopped successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping enterprise orchestrator: {e}")
            return False
    
    async def _configure_components(self):
        """Configure all components with enterprise settings."""
        # Configure queue manager
        await self.queue_manager.create_partition(
            "high_priority",
            "High Priority Queue",
            QueueType.PRIORITY,
            max_size=5000,
            priority_weight=2.0
        )
        
        await self.queue_manager.create_partition(
            "batch_processing",
            "Batch Processing Queue",
            QueueType.FIFO,
            max_size=10000,
            priority_weight=0.5
        )
        
        # Configure scaling manager
        self.scaling_manager.update_load_balancer_config({
            "algorithm": self.config.load_balancing_algorithm.value,
            "health_check_interval": 30,
            "sticky_sessions": True
        })
        
        # Configure health monitor
        self.health_monitor.set_failover_strategy(self.config.failover_strategy)
        
        self.logger.info("Component configuration completed")
    
    # Event handlers
    
    async def _on_node_joined(self, node_info):
        """Handle node joined event."""
        self.logger.info(f"Node joined cluster: {node_info.node_id} ({node_info.role.value})")
        self.system_metrics["active_nodes"] += 1
        
        # Update queue manager resources if it's a worker node
        if node_info.role == NodeRole.WORKER and node_info.resources:
            additional_resources = {
                "cpu_cores": node_info.resources.get("cpu_cores", 0),
                "memory_mb": node_info.resources.get("memory_total", 0) / (1024 * 1024),
                "storage_mb": node_info.resources.get("disk_total", 0) / (1024 * 1024)
            }
            self.queue_manager.update_available_resources(additional_resources)
    
    async def _on_node_left(self, node_info):
        """Handle node left event."""
        self.logger.warning(f"Node left cluster: {node_info.node_id} ({node_info.role.value})")
        self.system_metrics["active_nodes"] = max(0, self.system_metrics["active_nodes"] - 1)
    
    async def _on_leader_changed(self, new_leader_id: str, is_leader: bool):
        """Handle leader changed event."""
        if is_leader:
            self.logger.info(f"Became cluster leader: {new_leader_id}")
        else:
            self.logger.info(f"New cluster leader: {new_leader_id}")
    
    async def _on_health_alert(self, alert: Dict[str, Any]):
        """Handle health monitoring alerts."""
        severity = alert.get("severity", "info")
        message = alert.get("message", "")
        
        self.logger.log(
            logging.ERROR if severity in ["error", "critical"] else logging.WARNING,
            f"Health alert [{severity}]: {message}"
        )
        
        # In production, you might want to:
        # - Send notifications (email, Slack, etc.)
        # - Update monitoring dashboards
        # - Trigger automated responses
    
    # Public API methods
    
    async def submit_job(
        self,
        task_data: Dict[str, Any],
        tenant_id: str = "default",
        priority: int = 1,
        resource_requirements: ResourceRequirement = None,
        partition_id: str = "default"
    ) -> Optional[str]:
        """Submit a job to the enterprise system."""
        try:
            # Check tenant authorization and quotas
            if self.config.multi_tenant_enabled and tenant_id != "default":
                if not self.multi_tenant_manager.is_tenant_authorized(tenant_id):
                    self.logger.warning(f"Unauthorized tenant: {tenant_id}")
                    return None
                
                # Check resource quotas
                if not await self.multi_tenant_manager.check_quota(tenant_id, "daily_requests", 1):
                    self.logger.warning(f"Quota exceeded for tenant: {tenant_id}")
                    return None
            
            # Create task
            from models.task import Task, TaskType, TaskStatus
            task = Task(
                type=TaskType.FETCH_URL,
                parameters=task_data,
                priority=priority,
                status=TaskStatus.PENDING,
                metadata={"tenant_id": tenant_id, "submitted_at": datetime.now().isoformat()}
            )
            
            # Submit to queue manager
            success = await self.queue_manager.submit_task(
                task,
                resource_requirements,
                partition_id,
                priority_boost=0.0
            )
            
            if success:
                # Update tenant usage
                if self.config.multi_tenant_enabled and tenant_id != "default":
                    await self.multi_tenant_manager.allocate_resource(tenant_id, "daily_requests", 1)
                
                # Update metrics
                self.system_metrics["total_requests"] += 1
                
                self.logger.debug(f"Job submitted successfully: {task.id}")
                return task.id
            else:
                self.logger.warning(f"Failed to submit job for tenant: {tenant_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error submitting job: {e}")
            self.system_metrics["failed_requests"] += 1
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        cluster_stats = self.cluster_manager.get_cluster_stats()
        queue_stats = self.queue_manager.get_queue_statistics()
        
        status = {
            "system": {
                "running": self.is_running,
                "startup_time": self.startup_time.isoformat() if self.startup_time else None,
                "uptime_seconds": (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,
                "cluster_id": self.config.cluster_id,
                "cluster_name": self.config.cluster_name
            },
            "cluster": cluster_stats,
            "queues": queue_stats,
            "metrics": self.system_metrics.copy()
        }
        
        # Add multi-tenant stats if enabled
        if self.config.multi_tenant_enabled:
            status["tenants"] = self.multi_tenant_manager.get_tenant_stats()
        
        # Add scaling stats if enabled
        if self.config.scaling_enabled:
            status["scaling"] = self.scaling_manager.get_load_balancer_stats()
            status["scaling_recommendations"] = self.scaling_manager.get_scaling_recommendations()
        
        # Add health monitoring stats if enabled
        if self.config.health_monitoring_enabled:
            status["health"] = self.health_monitor.get_monitoring_stats()
        
        return status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        metrics = {
            "system_metrics": self.system_metrics.copy(),
            "timestamp": datetime.now().isoformat()
        }
        
        if self.config.multi_tenant_enabled:
            metrics["tenant_metrics"] = {
                tenant_id: asdict(usage) 
                for tenant_id, usage in self.multi_tenant_manager.tenant_usage.items()
            }
        
        if self.config.scaling_enabled:
            metrics["scaling_metrics"] = self.scaling_manager.get_load_balancer_stats()
        
        if self.config.health_monitoring_enabled:
            metrics["health_metrics"] = {
                node_id: asdict(score)
                for node_id, score in self.health_monitor.get_all_health_scores().items()
            }
        
        return metrics
    
    async def create_tenant(
        self,
        name: str,
        email: str,
        billing_model: BillingModel = None,
        custom_quota: ResourceQuota = None
    ) -> Optional[str]:
        """Create a new tenant."""
        if not self.config.multi_tenant_enabled:
            self.logger.warning("Multi-tenancy is not enabled")
            return None
        
        try:
            billing_model = billing_model or self.config.default_billing_model
            tenant_id = await self.multi_tenant_manager.create_tenant(
                name=name,
                email=email,
                billing_model=billing_model,
                quota=custom_quota
            )
            
            if tenant_id:
                self.system_metrics["total_tenants"] += 1
                self.logger.info(f"Created tenant: {tenant_id} ({name})")
            
            return tenant_id
            
        except Exception as e:
            self.logger.error(f"Failed to create tenant: {e}")
            return None
    
    async def scale_cluster(self, target_nodes: int) -> bool:
        """Manually scale the cluster to target number of nodes."""
        if not self.config.scaling_enabled:
            self.logger.warning("Scaling is not enabled")
            return False
        
        try:
            current_nodes = len(self.cluster_manager.get_healthy_nodes())
            
            if target_nodes > current_nodes:
                # Scale up
                from services.horizontal_scaling_manager import ScalingDirection
                return self.scaling_manager.force_scaling_action(
                    ScalingDirection.UP, 
                    target_nodes - current_nodes
                )
            elif target_nodes < current_nodes:
                # Scale down
                from services.horizontal_scaling_manager import ScalingDirection
                return self.scaling_manager.force_scaling_action(
                    ScalingDirection.DOWN, 
                    current_nodes - target_nodes
                )
            else:
                # Already at target
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to scale cluster: {e}")
            return False
