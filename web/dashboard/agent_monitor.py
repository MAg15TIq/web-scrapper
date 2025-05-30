"""
Agent Monitor
Real-time monitoring and metrics collection for agents.
"""

import asyncio
import logging
import psutil
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from cli.agent_communication import AgentCommunicationLayer


# Configure logging
logger = logging.getLogger("agent_monitor")


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    agent_id: str
    agent_type: str
    status: str
    cpu_usage: float
    memory_usage: float
    tasks_completed: int
    tasks_failed: int
    tasks_active: int
    average_response_time: float
    last_activity: datetime
    uptime: float
    error_rate: float
    throughput: float


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_agents: int
    total_tasks: int
    system_load: float


class MetricsCollector:
    """Collects and stores metrics data."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.agent_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.system_metrics_history: deque = deque(maxlen=max_history)
        self.alerts: List[Dict[str, Any]] = []
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "error_rate": 5.0,
            "response_time": 5000.0  # milliseconds
        }
    
    def add_agent_metrics(self, metrics: AgentMetrics):
        """Add agent metrics to history."""
        self.agent_metrics_history[metrics.agent_id].append(metrics)
        self._check_agent_alerts(metrics)
    
    def add_system_metrics(self, metrics: SystemMetrics):
        """Add system metrics to history."""
        self.system_metrics_history.append(metrics)
        self._check_system_alerts(metrics)
    
    def get_agent_metrics(self, agent_id: str, limit: int = 100) -> List[AgentMetrics]:
        """Get agent metrics history."""
        history = self.agent_metrics_history.get(agent_id, deque())
        return list(history)[-limit:]
    
    def get_system_metrics(self, limit: int = 100) -> List[SystemMetrics]:
        """Get system metrics history."""
        return list(self.system_metrics_history)[-limit:]
    
    def get_latest_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get latest metrics for an agent."""
        history = self.agent_metrics_history.get(agent_id)
        return history[-1] if history else None
    
    def get_latest_system_metrics(self) -> Optional[SystemMetrics]:
        """Get latest system metrics."""
        return self.system_metrics_history[-1] if self.system_metrics_history else None
    
    def _check_agent_alerts(self, metrics: AgentMetrics):
        """Check for agent-related alerts."""
        alerts = []
        
        if metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
            alerts.append({
                "type": "agent_high_cpu",
                "agent_id": metrics.agent_id,
                "value": metrics.cpu_usage,
                "threshold": self.alert_thresholds["cpu_usage"],
                "timestamp": datetime.now()
            })
        
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            alerts.append({
                "type": "agent_high_memory",
                "agent_id": metrics.agent_id,
                "value": metrics.memory_usage,
                "threshold": self.alert_thresholds["memory_usage"],
                "timestamp": datetime.now()
            })
        
        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            alerts.append({
                "type": "agent_high_error_rate",
                "agent_id": metrics.agent_id,
                "value": metrics.error_rate,
                "threshold": self.alert_thresholds["error_rate"],
                "timestamp": datetime.now()
            })
        
        if metrics.average_response_time > self.alert_thresholds["response_time"]:
            alerts.append({
                "type": "agent_slow_response",
                "agent_id": metrics.agent_id,
                "value": metrics.average_response_time,
                "threshold": self.alert_thresholds["response_time"],
                "timestamp": datetime.now()
            })
        
        self.alerts.extend(alerts)
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system-related alerts."""
        alerts = []
        
        if metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
            alerts.append({
                "type": "system_high_cpu",
                "value": metrics.cpu_usage,
                "threshold": self.alert_thresholds["cpu_usage"],
                "timestamp": datetime.now()
            })
        
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            alerts.append({
                "type": "system_high_memory",
                "value": metrics.memory_usage,
                "threshold": self.alert_thresholds["memory_usage"],
                "timestamp": datetime.now()
            })
        
        self.alerts.extend(alerts)
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return self.alerts[-limit:]
    
    def clear_old_alerts(self, hours: int = 24):
        """Clear alerts older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        self.alerts = [alert for alert in self.alerts if alert["timestamp"] > cutoff_time]


class AgentMonitor:
    """Real-time agent monitoring system."""
    
    def __init__(self):
        self.logger = logging.getLogger("agent_monitor")
        self.agent_comm = AgentCommunicationLayer()
        self.metrics_collector = MetricsCollector()
        
        # Monitoring state
        self.is_running = False
        self.monitoring_task = None
        self.update_interval = 5  # seconds
        
        # Agent tracking
        self.agent_start_times: Dict[str, datetime] = {}
        self.agent_task_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "completed": 0,
            "failed": 0,
            "active": 0
        })
        self.agent_response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        self.logger.info("Agent monitor initialized")
    
    async def start(self):
        """Start the agent monitoring system."""
        if self.is_running:
            self.logger.warning("Agent monitor is already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Initialize agent communication
        await self.agent_comm.initialize_agents()
        
        self.logger.info("Agent monitor started")
    
    async def stop(self):
        """Stop the agent monitoring system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Agent monitor stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect agent metrics
                await self._collect_agent_metrics()
                
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _collect_agent_metrics(self):
        """Collect metrics for all agents."""
        try:
            # Get agent statuses
            agent_statuses = await self.agent_comm.get_agent_status()
            
            for agent_id, status_data in agent_statuses.items():
                # Calculate metrics
                metrics = await self._calculate_agent_metrics(agent_id, status_data)
                
                # Store metrics
                self.metrics_collector.add_agent_metrics(metrics)
                
        except Exception as e:
            self.logger.error(f"Error collecting agent metrics: {e}")
    
    async def _calculate_agent_metrics(self, agent_id: str, status_data: Dict[str, Any]) -> AgentMetrics:
        """Calculate metrics for a specific agent."""
        # Get or initialize agent start time
        if agent_id not in self.agent_start_times:
            self.agent_start_times[agent_id] = datetime.now()
        
        # Calculate uptime
        uptime = (datetime.now() - self.agent_start_times[agent_id]).total_seconds()
        
        # Get task counts
        task_counts = self.agent_task_counts[agent_id]
        
        # Calculate error rate
        total_tasks = task_counts["completed"] + task_counts["failed"]
        error_rate = (task_counts["failed"] / total_tasks * 100) if total_tasks > 0 else 0.0
        
        # Calculate average response time
        response_times = self.agent_response_times[agent_id]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Calculate throughput (tasks per minute)
        throughput = (task_counts["completed"] / (uptime / 60)) if uptime > 0 else 0.0
        
        # Mock CPU and memory usage (in a real implementation, you'd get actual values)
        cpu_usage = 25.0 + (hash(agent_id) % 30)  # Mock value between 25-55%
        memory_usage = 40.0 + (hash(agent_id) % 25)  # Mock value between 40-65%
        
        return AgentMetrics(
            agent_id=agent_id,
            agent_type=status_data.get("agent_type", "unknown"),
            status=status_data.get("status", "unknown"),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            tasks_completed=task_counts["completed"],
            tasks_failed=task_counts["failed"],
            tasks_active=task_counts["active"],
            average_response_time=avg_response_time,
            last_activity=status_data.get("last_activity", datetime.now()),
            uptime=uptime,
            error_rate=error_rate,
            throughput=throughput
        )
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics."""
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Get agent count
            agent_statuses = await self.agent_comm.get_agent_status()
            active_agents = len([s for s in agent_statuses.values() if s.get("status") == "active"])
            
            # Calculate total tasks
            total_tasks = sum(
                counts["completed"] + counts["failed"] + counts["active"]
                for counts in self.agent_task_counts.values()
            )
            
            # Get system load
            system_load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_io={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv
                },
                active_agents=active_agents,
                total_tasks=total_tasks,
                system_load=system_load
            )
            
            self.metrics_collector.add_system_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        # Clean up old alerts
        self.metrics_collector.clear_old_alerts(hours=24)
        
        # Clean up old response times
        for agent_id in list(self.agent_response_times.keys()):
            if len(self.agent_response_times[agent_id]) == 0:
                del self.agent_response_times[agent_id]
    
    async def get_agent_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get current metrics for all agents."""
        metrics = {}
        
        for agent_id in self.agent_task_counts.keys():
            latest_metrics = self.metrics_collector.get_latest_agent_metrics(agent_id)
            if latest_metrics:
                metrics[agent_id] = asdict(latest_metrics)
        
        return metrics
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        latest_metrics = self.metrics_collector.get_latest_system_metrics()
        return asdict(latest_metrics) if latest_metrics else {}
    
    def update_agent_task_count(self, agent_id: str, task_type: str, increment: int = 1):
        """Update task count for an agent."""
        if task_type in ["completed", "failed", "active"]:
            self.agent_task_counts[agent_id][task_type] += increment
    
    def add_agent_response_time(self, agent_id: str, response_time: float):
        """Add response time measurement for an agent."""
        self.agent_response_times[agent_id].append(response_time)
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return self.metrics_collector.get_recent_alerts()
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring system statistics."""
        return {
            "is_running": self.is_running,
            "update_interval": self.update_interval,
            "tracked_agents": len(self.agent_task_counts),
            "total_metrics_points": sum(
                len(history) for history in self.metrics_collector.agent_metrics_history.values()
            ),
            "system_metrics_points": len(self.metrics_collector.system_metrics_history),
            "active_alerts": len(self.metrics_collector.get_recent_alerts())
        }
