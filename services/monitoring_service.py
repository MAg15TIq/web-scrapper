"""
Monitoring service for system observability using Prometheus metrics.
Provides real-time metrics collection and health monitoring.
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
    from prometheus_client.core import CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Install prometheus_client for monitoring.")

from config.langchain_config import get_config


class MetricsCollector:
    """Collector for system metrics with Prometheus integration."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.logger = logging.getLogger("monitoring.metrics")
        self.config = get_config().monitoring
        
        # Metrics registry
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        
        # System metrics
        self.metrics = {}
        self._initialize_metrics()
        
        # Internal tracking
        self.start_time = time.time()
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Mock storage for when Prometheus is not available
        self._mock_metrics = defaultdict(float)
        
    def _initialize_metrics(self):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Using mock metrics - Prometheus not available")
            return
        
        # Workflow metrics
        self.metrics['workflows_total'] = Counter(
            'workflows_total',
            'Total number of workflows executed',
            ['status', 'workflow_type'],
            registry=self.registry
        )
        
        self.metrics['workflow_duration'] = Histogram(
            'workflow_duration_seconds',
            'Workflow execution duration',
            ['workflow_type'],
            registry=self.registry
        )
        
        self.metrics['workflow_success_rate'] = Gauge(
            'workflow_success_rate',
            'Current workflow success rate',
            registry=self.registry
        )
        
        # Agent metrics
        self.metrics['agent_tasks_total'] = Counter(
            'agent_tasks_total',
            'Total number of agent tasks',
            ['agent_type', 'agent_id', 'status'],
            registry=self.registry
        )
        
        self.metrics['agent_execution_time'] = Histogram(
            'agent_execution_time_seconds',
            'Agent task execution time',
            ['agent_type', 'agent_id'],
            registry=self.registry
        )
        
        self.metrics['active_agents'] = Gauge(
            'active_agents_count',
            'Number of currently active agents',
            ['agent_type'],
            registry=self.registry
        )
        
        # Data quality metrics
        self.metrics['data_quality_score'] = Gauge(
            'data_quality_score',
            'Current data quality score',
            ['source'],
            registry=self.registry
        )
        
        self.metrics['data_records_processed'] = Counter(
            'data_records_processed_total',
            'Total number of data records processed',
            ['source', 'status'],
            registry=self.registry
        )
        
        # System health metrics
        self.metrics['system_health_score'] = Gauge(
            'system_health_score',
            'Overall system health score',
            registry=self.registry
        )
        
        self.metrics['error_rate'] = Gauge(
            'error_rate',
            'Current system error rate',
            registry=self.registry
        )
        
        self.metrics['memory_usage'] = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        # Performance metrics
        self.metrics['response_time'] = Histogram(
            'response_time_seconds',
            'Response time for operations',
            ['operation'],
            registry=self.registry
        )
        
        self.metrics['throughput'] = Gauge(
            'throughput_operations_per_second',
            'Operations per second',
            ['operation_type'],
            registry=self.registry
        )
        
        # System info
        self.metrics['system_info'] = Info(
            'system_info',
            'System information',
            registry=self.registry
        )
        
        self.logger.info("Prometheus metrics initialized")
    
    def record_workflow_start(self, workflow_id: str, workflow_type: str = "scraping"):
        """Record workflow start."""
        if PROMETHEUS_AVAILABLE:
            self.metrics['workflows_total'].labels(
                status='started',
                workflow_type=workflow_type
            ).inc()
        else:
            self._mock_metrics[f'workflows_started_{workflow_type}'] += 1
        
        # Store start time for duration calculation
        self.metric_history[f'workflow_start_{workflow_id}'].append(time.time())
    
    def record_workflow_completion(
        self,
        workflow_id: str,
        status: str,
        workflow_type: str = "scraping",
        execution_time: Optional[float] = None
    ):
        """Record workflow completion."""
        if PROMETHEUS_AVAILABLE:
            self.metrics['workflows_total'].labels(
                status=status,
                workflow_type=workflow_type
            ).inc()
            
            if execution_time:
                self.metrics['workflow_duration'].labels(
                    workflow_type=workflow_type
                ).observe(execution_time)
        else:
            self._mock_metrics[f'workflows_{status}_{workflow_type}'] += 1
            if execution_time:
                self._mock_metrics[f'workflow_duration_{workflow_type}'] = execution_time
        
        self.logger.debug(f"Recorded workflow completion: {workflow_id} - {status}")
    
    def record_agent_task(
        self,
        agent_type: str,
        agent_id: str,
        status: str,
        execution_time: Optional[float] = None
    ):
        """Record agent task execution."""
        if PROMETHEUS_AVAILABLE:
            self.metrics['agent_tasks_total'].labels(
                agent_type=agent_type,
                agent_id=agent_id,
                status=status
            ).inc()
            
            if execution_time:
                self.metrics['agent_execution_time'].labels(
                    agent_type=agent_type,
                    agent_id=agent_id
                ).observe(execution_time)
        else:
            self._mock_metrics[f'agent_tasks_{agent_type}_{status}'] += 1
            if execution_time:
                self._mock_metrics[f'agent_execution_time_{agent_type}'] = execution_time
    
    def update_active_agents(self, agent_type: str, count: int):
        """Update active agent count."""
        if PROMETHEUS_AVAILABLE:
            self.metrics['active_agents'].labels(agent_type=agent_type).set(count)
        else:
            self._mock_metrics[f'active_agents_{agent_type}'] = count
    
    def record_data_quality(self, source: str, quality_score: float):
        """Record data quality metrics."""
        if PROMETHEUS_AVAILABLE:
            self.metrics['data_quality_score'].labels(source=source).set(quality_score)
        else:
            self._mock_metrics[f'data_quality_{source}'] = quality_score
    
    def record_data_processing(self, source: str, status: str, count: int = 1):
        """Record data processing metrics."""
        if PROMETHEUS_AVAILABLE:
            self.metrics['data_records_processed'].labels(
                source=source,
                status=status
            ).inc(count)
        else:
            self._mock_metrics[f'data_processed_{source}_{status}'] += count
    
    def update_system_health(self, health_score: float):
        """Update system health score."""
        if PROMETHEUS_AVAILABLE:
            self.metrics['system_health_score'].set(health_score)
        else:
            self._mock_metrics['system_health_score'] = health_score
    
    def update_error_rate(self, error_rate: float):
        """Update system error rate."""
        if PROMETHEUS_AVAILABLE:
            self.metrics['error_rate'].set(error_rate)
        else:
            self._mock_metrics['error_rate'] = error_rate
    
    def record_memory_usage(self, component: str, bytes_used: int):
        """Record memory usage."""
        if PROMETHEUS_AVAILABLE:
            self.metrics['memory_usage'].labels(component=component).set(bytes_used)
        else:
            self._mock_metrics[f'memory_usage_{component}'] = bytes_used
    
    def record_response_time(self, operation: str, response_time: float):
        """Record operation response time."""
        if PROMETHEUS_AVAILABLE:
            self.metrics['response_time'].labels(operation=operation).observe(response_time)
        else:
            self._mock_metrics[f'response_time_{operation}'] = response_time
    
    def update_throughput(self, operation_type: str, ops_per_second: float):
        """Update throughput metrics."""
        if PROMETHEUS_AVAILABLE:
            self.metrics['throughput'].labels(operation_type=operation_type).set(ops_per_second)
        else:
            self._mock_metrics[f'throughput_{operation_type}'] = ops_per_second
    
    def set_system_info(self, info: Dict[str, str]):
        """Set system information."""
        if PROMETHEUS_AVAILABLE:
            self.metrics['system_info'].info(info)
        else:
            self._mock_metrics['system_info'] = str(info)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        if PROMETHEUS_AVAILABLE:
            # In a real implementation, you would collect metrics from Prometheus
            # For now, return a summary
            return {
                "prometheus_enabled": True,
                "metrics_count": len(self.metrics),
                "uptime_seconds": time.time() - self.start_time
            }
        else:
            return {
                "prometheus_enabled": False,
                "mock_metrics": dict(self._mock_metrics),
                "uptime_seconds": time.time() - self.start_time
            }


class MonitoringService:
    """
    Monitoring service for system observability and health tracking.
    Provides metrics collection, alerting, and performance monitoring.
    """
    
    def __init__(self):
        """Initialize monitoring service."""
        self.logger = logging.getLogger("monitoring.service")
        self.config = get_config().monitoring
        
        # Metrics collector
        self.metrics = MetricsCollector()
        
        # Health tracking
        self.health_checks = {}
        self.alert_thresholds = {
            "error_rate": 0.1,  # 10%
            "response_time": 30.0,  # 30 seconds
            "memory_usage": 0.9,  # 90%
            "disk_usage": 0.85,  # 85%
        }
        
        # Prometheus HTTP server
        self.prometheus_server_started = False
        
    async def start(self) -> bool:
        """Start monitoring service."""
        try:
            # Start Prometheus HTTP server if enabled
            if self.config.prometheus_enabled and PROMETHEUS_AVAILABLE:
                await self._start_prometheus_server()
            
            # Set system information
            self.metrics.set_system_info({
                "version": "2.0.0",
                "environment": "development",
                "start_time": datetime.now().isoformat()
            })
            
            self.logger.info("Monitoring service started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring service: {e}")
            return False
    
    async def _start_prometheus_server(self):
        """Start Prometheus HTTP server."""
        try:
            if not self.prometheus_server_started:
                start_http_server(
                    self.config.prometheus_port,
                    addr=self.config.prometheus_host,
                    registry=self.metrics.registry
                )
                self.prometheus_server_started = True
                self.logger.info(f"Prometheus server started on {self.config.prometheus_host}:{self.config.prometheus_port}")
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus server: {e}")
    
    async def stop(self):
        """Stop monitoring service."""
        self.logger.info("Monitoring service stopped")
    
    def register_health_check(self, name: str, check_func: callable):
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.logger.debug(f"Registered health check: {name}")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_health = True
        
        for name, check_func in self.health_checks.items():
            try:
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "details": result
                }
                if not result:
                    overall_health = False
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e)
                }
                overall_health = False
        
        # Update system health metric
        health_score = len([r for r in results.values() if r["status"] == "healthy"]) / len(results) if results else 1.0
        self.metrics.update_system_health(health_score)
        
        return {
            "overall_health": "healthy" if overall_health else "unhealthy",
            "checks": results,
            "health_score": health_score,
            "timestamp": datetime.now().isoformat()
        }
    
    def check_alerts(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        
        for metric, threshold in self.alert_thresholds.items():
            if metric in current_metrics:
                value = current_metrics[metric]
                if isinstance(value, (int, float)) and value > threshold:
                    alerts.append({
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "severity": "warning" if value < threshold * 1.2 else "critical",
                        "timestamp": datetime.now().isoformat()
                    })
        
        return alerts
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        current_metrics = self.metrics.get_current_metrics()
        
        return {
            "monitoring_service": {
                "status": "running",
                "prometheus_enabled": self.config.prometheus_enabled,
                "prometheus_server_started": self.prometheus_server_started,
                "health_checks_registered": len(self.health_checks)
            },
            "current_metrics": current_metrics,
            "uptime": current_metrics.get("uptime_seconds", 0),
            "timestamp": datetime.now().isoformat()
        }


# Global monitoring service instance
_monitoring_service: Optional[MonitoringService] = None


async def get_monitoring_service() -> MonitoringService:
    """Get the global monitoring service instance."""
    global _monitoring_service
    
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
        await _monitoring_service.start()
    
    return _monitoring_service


async def cleanup_monitoring_service():
    """Cleanup the global monitoring service."""
    global _monitoring_service
    
    if _monitoring_service:
        await _monitoring_service.stop()
        _monitoring_service = None


# Convenience functions for common metrics
async def record_workflow_metrics(workflow_id: str, status: str, execution_time: float):
    """Record workflow completion metrics."""
    monitoring = await get_monitoring_service()
    monitoring.metrics.record_workflow_completion(workflow_id, status, execution_time=execution_time)


async def record_agent_metrics(agent_type: str, agent_id: str, status: str, execution_time: float):
    """Record agent task metrics."""
    monitoring = await get_monitoring_service()
    monitoring.metrics.record_agent_task(agent_type, agent_id, status, execution_time)
