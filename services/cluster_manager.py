"""
Cluster Management System for Phase 3: Enterprise & Scalability Features.

This module provides distributed scraping architecture with multi-node clusters,
node discovery, registration, and cluster-wide coordination.
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import socket
import psutil

try:
    import consul
    CONSUL_AVAILABLE = True
except ImportError:
    CONSUL_AVAILABLE = False
    logging.warning("Consul not available. Install python-consul for service discovery.")

try:
    import etcd3
    ETCD_AVAILABLE = True
except ImportError:
    ETCD_AVAILABLE = False
    logging.warning("etcd3 not available. Install etcd3 for distributed configuration.")


class NodeStatus(Enum):
    """Node status enumeration."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class NodeRole(Enum):
    """Node role enumeration."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    HYBRID = "hybrid"
    MONITOR = "monitor"


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    role: NodeRole
    status: NodeStatus
    capabilities: List[str]
    resources: Dict[str, Any]
    metadata: Dict[str, Any]
    last_heartbeat: datetime
    joined_at: datetime
    version: str


@dataclass
class ClusterConfig:
    """Cluster configuration."""
    cluster_id: str
    name: str
    discovery_method: str  # consul, etcd, static
    heartbeat_interval: int = 30
    health_check_interval: int = 60
    node_timeout: int = 180
    max_nodes: int = 100
    auto_scaling: bool = True
    load_balancing: bool = True
    failover_enabled: bool = True


class ClusterManager:
    """
    Manages distributed scraping cluster with node discovery, health monitoring,
    and automatic failover capabilities.
    """
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.logger = logging.getLogger("cluster_manager")
        
        # Node information
        self.node_id = str(uuid.uuid4())
        self.node_info: Optional[NodeInfo] = None
        self.cluster_nodes: Dict[str, NodeInfo] = {}
        
        # Service discovery
        self.consul_client = None
        self.etcd_client = None
        self._init_service_discovery()
        
        # Cluster state
        self.is_leader = False
        self.leader_node_id: Optional[str] = None
        self.cluster_health = 1.0
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.leader_election_task: Optional[asyncio.Task] = None
        
        # Event callbacks
        self.node_joined_callbacks: List[callable] = []
        self.node_left_callbacks: List[callable] = []
        self.leader_changed_callbacks: List[callable] = []
        
        self.logger.info(f"Cluster manager initialized for cluster: {config.cluster_id}")
    
    def _init_service_discovery(self):
        """Initialize service discovery clients."""
        if self.config.discovery_method == "consul" and CONSUL_AVAILABLE:
            try:
                self.consul_client = consul.Consul()
                self.logger.info("Consul client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Consul client: {e}")
        
        elif self.config.discovery_method == "etcd" and ETCD_AVAILABLE:
            try:
                self.etcd_client = etcd3.client()
                self.logger.info("etcd client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize etcd client: {e}")
    
    async def join_cluster(
        self,
        role: NodeRole = NodeRole.WORKER,
        capabilities: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Join the cluster as a new node."""
        try:
            # Get node information
            hostname = socket.gethostname()
            ip_address = self._get_local_ip()
            port = self._get_available_port()
            
            # Create node info
            self.node_info = NodeInfo(
                node_id=self.node_id,
                hostname=hostname,
                ip_address=ip_address,
                port=port,
                role=role,
                status=NodeStatus.INITIALIZING,
                capabilities=capabilities or [],
                resources=self._get_node_resources(),
                metadata=metadata or {},
                last_heartbeat=datetime.now(),
                joined_at=datetime.now(),
                version="1.0.0"
            )
            
            # Register with service discovery
            success = await self._register_node()
            if success:
                self.node_info.status = NodeStatus.HEALTHY
                
                # Start background tasks
                await self._start_background_tasks()
                
                # Trigger callbacks
                for callback in self.node_joined_callbacks:
                    try:
                        await callback(self.node_info)
                    except Exception as e:
                        self.logger.error(f"Error in node joined callback: {e}")
                
                self.logger.info(f"Successfully joined cluster as {role.value}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to join cluster: {e}")
            return False
    
    async def leave_cluster(self) -> bool:
        """Leave the cluster gracefully."""
        try:
            if self.node_info:
                # Update status
                self.node_info.status = NodeStatus.OFFLINE
                
                # Stop background tasks
                await self._stop_background_tasks()
                
                # Deregister from service discovery
                await self._deregister_node()
                
                # Trigger callbacks
                for callback in self.node_left_callbacks:
                    try:
                        await callback(self.node_info)
                    except Exception as e:
                        self.logger.error(f"Error in node left callback: {e}")
                
                self.logger.info("Successfully left cluster")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to leave cluster: {e}")
            return False
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def _get_available_port(self, start_port: int = 8000) -> int:
        """Find an available port."""
        for port in range(start_port, start_port + 1000):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                continue
        return start_port
    
    def _get_node_resources(self) -> Dict[str, Any]:
        """Get current node resource information."""
        try:
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_cores": cpu_count,
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_total": memory.total,
                "memory_available": memory.available,
                "memory_usage": memory.percent,
                "disk_total": disk.total,
                "disk_free": disk.free,
                "disk_usage": (disk.used / disk.total) * 100,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            self.logger.error(f"Failed to get node resources: {e}")
            return {}
    
    async def _register_node(self) -> bool:
        """Register node with service discovery."""
        if not self.node_info:
            return False
        
        try:
            node_data = asdict(self.node_info)
            # Convert datetime objects to ISO strings
            node_data['last_heartbeat'] = self.node_info.last_heartbeat.isoformat()
            node_data['joined_at'] = self.node_info.joined_at.isoformat()
            
            if self.consul_client:
                # Register with Consul
                self.consul_client.agent.service.register(
                    name=f"webscraper-{self.config.cluster_id}",
                    service_id=self.node_id,
                    address=self.node_info.ip_address,
                    port=self.node_info.port,
                    tags=[self.node_info.role.value] + self.node_info.capabilities,
                    meta=node_data
                )
                return True
            
            elif self.etcd_client:
                # Register with etcd
                key = f"/webscraper/{self.config.cluster_id}/nodes/{self.node_id}"
                value = json.dumps(node_data)
                self.etcd_client.put(key, value)
                return True
            
            else:
                # Fallback to in-memory registration
                self.cluster_nodes[self.node_id] = self.node_info
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to register node: {e}")
            return False
    
    async def _deregister_node(self) -> bool:
        """Deregister node from service discovery."""
        try:
            if self.consul_client:
                self.consul_client.agent.service.deregister(self.node_id)
                return True
            
            elif self.etcd_client:
                key = f"/webscraper/{self.config.cluster_id}/nodes/{self.node_id}"
                self.etcd_client.delete(key)
                return True
            
            else:
                # Remove from in-memory storage
                self.cluster_nodes.pop(self.node_id, None)
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to deregister node: {e}")
            return False

    async def _start_background_tasks(self):
        """Start background tasks for cluster management."""
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.leader_election_task = asyncio.create_task(self._leader_election_loop())

    async def _stop_background_tasks(self):
        """Stop background tasks."""
        tasks = [self.heartbeat_task, self.health_check_task, self.leader_election_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to maintain cluster membership."""
        while True:
            try:
                if self.node_info:
                    self.node_info.last_heartbeat = datetime.now()
                    self.node_info.resources = self._get_node_resources()
                    await self._update_node_info()

                await asyncio.sleep(self.config.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)

    async def _health_check_loop(self):
        """Monitor cluster health and detect failed nodes."""
        while True:
            try:
                await self._discover_nodes()
                await self._check_node_health()
                await self._update_cluster_health()

                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)

    async def _leader_election_loop(self):
        """Handle leader election and monitoring."""
        while True:
            try:
                if not self.is_leader:
                    await self._attempt_leader_election()
                else:
                    await self._maintain_leadership()

                await asyncio.sleep(30)  # Check leadership every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in leader election loop: {e}")
                await asyncio.sleep(10)

    async def _update_node_info(self):
        """Update node information in service discovery."""
        if not self.node_info:
            return

        try:
            node_data = asdict(self.node_info)
            node_data['last_heartbeat'] = self.node_info.last_heartbeat.isoformat()
            node_data['joined_at'] = self.node_info.joined_at.isoformat()

            if self.consul_client:
                # Update Consul service
                self.consul_client.agent.service.register(
                    name=f"webscraper-{self.config.cluster_id}",
                    service_id=self.node_id,
                    address=self.node_info.ip_address,
                    port=self.node_info.port,
                    tags=[self.node_info.role.value] + self.node_info.capabilities,
                    meta=node_data
                )

            elif self.etcd_client:
                # Update etcd
                key = f"/webscraper/{self.config.cluster_id}/nodes/{self.node_id}"
                value = json.dumps(node_data)
                self.etcd_client.put(key, value)

            else:
                # Update in-memory storage
                self.cluster_nodes[self.node_id] = self.node_info

        except Exception as e:
            self.logger.error(f"Failed to update node info: {e}")

    async def _discover_nodes(self):
        """Discover other nodes in the cluster."""
        try:
            if self.consul_client:
                # Discover via Consul
                services = self.consul_client.health.service(
                    f"webscraper-{self.config.cluster_id}",
                    passing=True
                )[1]

                for service in services:
                    node_id = service['Service']['ID']
                    if node_id != self.node_id:
                        node_info = self._parse_consul_service(service)
                        if node_info:
                            self.cluster_nodes[node_id] = node_info

            elif self.etcd_client:
                # Discover via etcd
                prefix = f"/webscraper/{self.config.cluster_id}/nodes/"
                for value, metadata in self.etcd_client.get_prefix(prefix):
                    try:
                        node_data = json.loads(value.decode())
                        node_info = self._parse_node_data(node_data)
                        if node_info and node_info.node_id != self.node_id:
                            self.cluster_nodes[node_info.node_id] = node_info
                    except Exception as e:
                        self.logger.error(f"Failed to parse node data: {e}")

        except Exception as e:
            self.logger.error(f"Failed to discover nodes: {e}")

    def _parse_consul_service(self, service: Dict) -> Optional[NodeInfo]:
        """Parse Consul service data into NodeInfo."""
        try:
            meta = service['Service'].get('Meta', {})
            return NodeInfo(
                node_id=service['Service']['ID'],
                hostname=meta.get('hostname', ''),
                ip_address=service['Service']['Address'],
                port=service['Service']['Port'],
                role=NodeRole(meta.get('role', 'worker')),
                status=NodeStatus(meta.get('status', 'healthy')),
                capabilities=meta.get('capabilities', []),
                resources=json.loads(meta.get('resources', '{}')),
                metadata=json.loads(meta.get('metadata', '{}')),
                last_heartbeat=datetime.fromisoformat(meta.get('last_heartbeat', datetime.now().isoformat())),
                joined_at=datetime.fromisoformat(meta.get('joined_at', datetime.now().isoformat())),
                version=meta.get('version', '1.0.0')
            )
        except Exception as e:
            self.logger.error(f"Failed to parse Consul service: {e}")
            return None

    def _parse_node_data(self, data: Dict) -> Optional[NodeInfo]:
        """Parse node data dictionary into NodeInfo."""
        try:
            return NodeInfo(
                node_id=data['node_id'],
                hostname=data['hostname'],
                ip_address=data['ip_address'],
                port=data['port'],
                role=NodeRole(data['role']),
                status=NodeStatus(data['status']),
                capabilities=data['capabilities'],
                resources=data['resources'],
                metadata=data['metadata'],
                last_heartbeat=datetime.fromisoformat(data['last_heartbeat']),
                joined_at=datetime.fromisoformat(data['joined_at']),
                version=data['version']
            )
        except Exception as e:
            self.logger.error(f"Failed to parse node data: {e}")
            return None

    async def _check_node_health(self):
        """Check health of all nodes and remove unhealthy ones."""
        current_time = datetime.now()
        timeout_threshold = timedelta(seconds=self.config.node_timeout)

        unhealthy_nodes = []
        for node_id, node_info in list(self.cluster_nodes.items()):
            if current_time - node_info.last_heartbeat > timeout_threshold:
                unhealthy_nodes.append(node_id)
                node_info.status = NodeStatus.OFFLINE

        # Remove unhealthy nodes
        for node_id in unhealthy_nodes:
            self.logger.warning(f"Removing unhealthy node: {node_id}")
            removed_node = self.cluster_nodes.pop(node_id, None)

            # Trigger callbacks
            if removed_node:
                for callback in self.node_left_callbacks:
                    try:
                        await callback(removed_node)
                    except Exception as e:
                        self.logger.error(f"Error in node left callback: {e}")

    async def _update_cluster_health(self):
        """Update overall cluster health score."""
        if not self.cluster_nodes:
            self.cluster_health = 0.0
            return

        healthy_nodes = sum(1 for node in self.cluster_nodes.values()
                          if node.status == NodeStatus.HEALTHY)
        self.cluster_health = healthy_nodes / len(self.cluster_nodes)

        self.logger.debug(f"Cluster health: {self.cluster_health:.2f} "
                         f"({healthy_nodes}/{len(self.cluster_nodes)} nodes healthy)")

    async def _attempt_leader_election(self):
        """Attempt to become cluster leader."""
        try:
            if self.consul_client:
                # Use Consul's leader election
                session_id = self.consul_client.session.create(
                    ttl=60,
                    behavior='release'
                )

                leader_key = f"webscraper/{self.config.cluster_id}/leader"
                success = self.consul_client.kv.put(
                    leader_key,
                    self.node_id,
                    acquire=session_id
                )

                if success:
                    self.is_leader = True
                    self.leader_node_id = self.node_id
                    self.logger.info("Became cluster leader")

                    # Trigger callbacks
                    for callback in self.leader_changed_callbacks:
                        try:
                            await callback(self.node_id, True)
                        except Exception as e:
                            self.logger.error(f"Error in leader changed callback: {e}")

            elif self.etcd_client:
                # Use etcd's leader election
                leader_key = f"/webscraper/{self.config.cluster_id}/leader"
                lease = self.etcd_client.lease(60)

                success = self.etcd_client.transaction(
                    compare=[self.etcd_client.transactions.create(leader_key) == 0],
                    success=[self.etcd_client.transactions.put(leader_key, self.node_id, lease)],
                    failure=[]
                )

                if success:
                    self.is_leader = True
                    self.leader_node_id = self.node_id
                    self.logger.info("Became cluster leader")

            else:
                # Simple leader election based on node ID
                if not self.leader_node_id or self.leader_node_id not in self.cluster_nodes:
                    # Become leader if no current leader or leader is offline
                    self.is_leader = True
                    self.leader_node_id = self.node_id
                    self.logger.info("Became cluster leader (fallback)")

        except Exception as e:
            self.logger.error(f"Failed leader election: {e}")

    async def _maintain_leadership(self):
        """Maintain leadership by renewing locks."""
        try:
            if self.consul_client:
                # Renew Consul session
                leader_key = f"webscraper/{self.config.cluster_id}/leader"
                current_leader = self.consul_client.kv.get(leader_key)[1]

                if not current_leader or current_leader['Value'].decode() != self.node_id:
                    self.is_leader = False
                    self.logger.warning("Lost cluster leadership")

            elif self.etcd_client:
                # Check etcd leadership
                leader_key = f"/webscraper/{self.config.cluster_id}/leader"
                value, metadata = self.etcd_client.get(leader_key)

                if not value or value.decode() != self.node_id:
                    self.is_leader = False
                    self.logger.warning("Lost cluster leadership")

        except Exception as e:
            self.logger.error(f"Failed to maintain leadership: {e}")
            self.is_leader = False

    # Public API methods

    def get_cluster_nodes(self) -> Dict[str, NodeInfo]:
        """Get all nodes in the cluster."""
        return self.cluster_nodes.copy()

    def get_healthy_nodes(self) -> Dict[str, NodeInfo]:
        """Get only healthy nodes in the cluster."""
        return {
            node_id: node_info
            for node_id, node_info in self.cluster_nodes.items()
            if node_info.status == NodeStatus.HEALTHY
        }

    def get_nodes_by_role(self, role: NodeRole) -> Dict[str, NodeInfo]:
        """Get nodes by their role."""
        return {
            node_id: node_info
            for node_id, node_info in self.cluster_nodes.items()
            if node_info.role == role
        }

    def get_nodes_by_capability(self, capability: str) -> Dict[str, NodeInfo]:
        """Get nodes that have a specific capability."""
        return {
            node_id: node_info
            for node_id, node_info in self.cluster_nodes.items()
            if capability in node_info.capabilities
        }

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        total_nodes = len(self.cluster_nodes)
        healthy_nodes = len(self.get_healthy_nodes())

        roles = {}
        for node in self.cluster_nodes.values():
            role = node.role.value
            roles[role] = roles.get(role, 0) + 1

        total_cpu_cores = sum(node.resources.get('cpu_cores', 0)
                             for node in self.cluster_nodes.values())
        total_memory = sum(node.resources.get('memory_total', 0)
                          for node in self.cluster_nodes.values())

        return {
            "cluster_id": self.config.cluster_id,
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
            "cluster_health": self.cluster_health,
            "leader_node_id": self.leader_node_id,
            "is_leader": self.is_leader,
            "roles": roles,
            "total_cpu_cores": total_cpu_cores,
            "total_memory_gb": total_memory / (1024**3) if total_memory else 0,
            "discovery_method": self.config.discovery_method
        }

    # Event callback registration

    def on_node_joined(self, callback: callable):
        """Register callback for node joined events."""
        self.node_joined_callbacks.append(callback)

    def on_node_left(self, callback: callable):
        """Register callback for node left events."""
        self.node_left_callbacks.append(callback)

    def on_leader_changed(self, callback: callable):
        """Register callback for leader changed events."""
        self.leader_changed_callbacks.append(callback)
