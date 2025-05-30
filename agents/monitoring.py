"""
Monitoring agent for the web scraping system.
"""
import asyncio
import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict

from agents.base import Agent
from models.message import Message, StatusMessage, AlertMessage, MetricMessage


class MonitoringAgent(Agent):
    """
    Agent responsible for monitoring system health and performance.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new monitoring agent.
        
        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent.
        """
        super().__init__(agent_id=agent_id, agent_type="monitoring")
        self.coordinator_id = coordinator_id
        
        # Monitoring thresholds
        self.thresholds = {
            "cpu_usage": 80.0,  # Percentage
            "memory_usage": 80.0,  # Percentage
            "disk_usage": 80.0,  # Percentage
            "task_success_rate": 0.9,  # 90%
            "task_execution_time": 300.0,  # 5 minutes
            "error_rate": 0.1,  # 10%
            "queue_size": 1000,  # Number of tasks
            "agent_health": 0.8  # 80% healthy agents
        }
        
        # Alert configuration
        self.alert_levels = {
            "info": 0,
            "warning": 1,
            "error": 2,
            "critical": 3
        }
        
        # Alert channels
        self.alert_channels = {
            "log": True,
            "email": False,
            "slack": False,
            "webhook": False
        }
        
        # Alert history
        self.alert_history: List[Dict[str, Any]] = []
        self.max_alert_history = 1000
        
        # Metrics storage
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.max_metrics_history = 10000
        
        # Register message handlers
        self.register_handler("status", self._handle_status_message)
        self.register_handler("metric", self._handle_metric_message)
        
        # Start periodic tasks
        self._start_periodic_tasks()
    
    def _start_periodic_tasks(self) -> None:
        """Start periodic monitoring tasks."""
        asyncio.create_task(self._periodic_health_check())
        asyncio.create_task(self._periodic_metrics_analysis())
        asyncio.create_task(self._periodic_alert_cleanup())
    
    async def _periodic_health_check(self) -> None:
        """Periodically check system health."""
        while self.running:
            try:
                # Check health every minute
                await asyncio.sleep(60)
                if not self.running:
                    break
                
                self.logger.info("Running periodic health check")
                await self._check_system_health()
            except Exception as e:
                self.logger.error(f"Error in periodic health check: {str(e)}", exc_info=True)
    
    async def _periodic_metrics_analysis(self) -> None:
        """Periodically analyze system metrics."""
        while self.running:
            try:
                # Analyze metrics every 5 minutes
                await asyncio.sleep(300)
                if not self.running:
                    break
                
                self.logger.info("Running periodic metrics analysis")
                await self._analyze_metrics()
            except Exception as e:
                self.logger.error(f"Error in periodic metrics analysis: {str(e)}", exc_info=True)
    
    async def _periodic_alert_cleanup(self) -> None:
        """Periodically clean up old alerts."""
        while self.running:
            try:
                # Clean up alerts every hour
                await asyncio.sleep(3600)
                if not self.running:
                    break
                
                self.logger.info("Running periodic alert cleanup")
                self._cleanup_old_alerts()
            except Exception as e:
                self.logger.error(f"Error in periodic alert cleanup: {str(e)}", exc_info=True)
    
    async def _check_system_health(self) -> None:
        """Check the health of the entire system."""
        # Get system metrics
        system_metrics = await self._get_system_metrics()
        
        # Check CPU usage
        if system_metrics["cpu_usage"] > self.thresholds["cpu_usage"]:
            await self._create_alert(
                "High CPU Usage",
                f"CPU usage is {system_metrics['cpu_usage']}%",
                "warning"
            )
        
        # Check memory usage
        if system_metrics["memory_usage"] > self.thresholds["memory_usage"]:
            await self._create_alert(
                "High Memory Usage",
                f"Memory usage is {system_metrics['memory_usage']}%",
                "warning"
            )
        
        # Check disk usage
        if system_metrics["disk_usage"] > self.thresholds["disk_usage"]:
            await self._create_alert(
                "High Disk Usage",
                f"Disk usage is {system_metrics['disk_usage']}%",
                "warning"
            )
        
        # Check task success rate
        if system_metrics["task_success_rate"] < self.thresholds["task_success_rate"]:
            await self._create_alert(
                "Low Task Success Rate",
                f"Task success rate is {system_metrics['task_success_rate']*100}%",
                "error"
            )
        
        # Check error rate
        if system_metrics["error_rate"] > self.thresholds["error_rate"]:
            await self._create_alert(
                "High Error Rate",
                f"Error rate is {system_metrics['error_rate']*100}%",
                "error"
            )
        
        # Check queue size
        if system_metrics["queue_size"] > self.thresholds["queue_size"]:
            await self._create_alert(
                "Large Task Queue",
                f"Queue size is {system_metrics['queue_size']} tasks",
                "warning"
            )
        
        # Check agent health
        if system_metrics["agent_health"] < self.thresholds["agent_health"]:
            await self._create_alert(
                "Low Agent Health",
                f"Agent health is {system_metrics['agent_health']*100}%",
                "error"
            )
    
    async def _get_system_metrics(self) -> Dict[str, float]:
        """
        Get current system metrics.
        
        Returns:
            A dictionary of system metrics.
        """
        # This is a placeholder for actual system metrics collection
        # In a real implementation, you would collect actual system metrics
        return {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "task_success_rate": 1.0,
            "task_execution_time": 0.0,
            "error_rate": 0.0,
            "queue_size": 0,
            "agent_health": 1.0
        }
    
    async def _analyze_metrics(self) -> None:
        """Analyze system metrics for trends and anomalies."""
        # Analyze each metric type
        for metric_type, metric_data in self.metrics.items():
            if not metric_data:
                continue
            
            # Calculate basic statistics
            values = [m["value"] for m in metric_data]
            avg_value = sum(values) / len(values)
            max_value = max(values)
            min_value = min(values)
            
            # Detect anomalies (values more than 2 standard deviations from mean)
            std_dev = (sum((x - avg_value) ** 2 for x in values) / len(values)) ** 0.5
            threshold = 2 * std_dev
            
            anomalies = [m for m in metric_data if abs(m["value"] - avg_value) > threshold]
            
            if anomalies:
                await self._create_alert(
                    f"Anomaly Detected in {metric_type}",
                    f"Found {len(anomalies)} anomalies in {metric_type}",
                    "warning"
                )
    
    async def _create_alert(self, title: str, message: str, level: str) -> None:
        """
        Create and send an alert.
        
        Args:
            title: The alert title.
            message: The alert message.
            level: The alert level (info, warning, error, critical).
        """
        if level not in self.alert_levels:
            self.logger.warning(f"Invalid alert level: {level}")
            return
        
        alert = {
            "timestamp": time.time(),
            "title": title,
            "message": message,
            "level": level,
            "level_value": self.alert_levels[level]
        }
        
        # Add to alert history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_alert_history:
            self.alert_history = self.alert_history[-self.max_alert_history:]
        
        # Send alert through configured channels
        if self.alert_channels["log"]:
            self.logger.warning(f"ALERT [{level.upper()}] {title}: {message}")
        
        if self.alert_channels["email"]:
            # TODO: Implement email alerts
            pass
        
        if self.alert_channels["slack"]:
            # TODO: Implement Slack alerts
            pass
        
        if self.alert_channels["webhook"]:
            # TODO: Implement webhook alerts
            pass
    
    def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts from history."""
        # Remove alerts older than 7 days
        cutoff_time = time.time() - (7 * 24 * 3600)
        self.alert_history = [a for a in self.alert_history if a["timestamp"] > cutoff_time]
    
    async def _handle_status_message(self, message: StatusMessage) -> None:
        """
        Handle a status message from an agent.
        
        Args:
            message: The status message to handle.
        """
        self.logger.debug(f"Received status update from {message.sender_id}: {message.status}")
        
        # Check for unhealthy status
        if message.status == "error" or message.status == "failed":
            await self._create_alert(
                f"Agent {message.sender_id} Status",
                f"Agent reported {message.status} status",
                "error"
            )
    
    async def _handle_metric_message(self, message: MetricMessage) -> None:
        """
        Handle a metric message from an agent.
        
        Args:
            message: The metric message to handle.
        """
        self.logger.debug(f"Received metric from {message.sender_id}: {message.metric_type}")
        
        # Store metric
        metric_data = {
            "timestamp": time.time(),
            "value": message.value,
            "tags": message.tags
        }
        
        self.metrics[message.metric_type].append(metric_data)
        
        # Trim metrics history if needed
        if len(self.metrics[message.metric_type]) > self.max_metrics_history:
            self.metrics[message.metric_type] = self.metrics[message.metric_type][-self.max_metrics_history:]
    
    def get_alert_history(self, level: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get alert history.
        
        Args:
            level: Filter alerts by level.
            limit: Maximum number of alerts to return.
            
        Returns:
            A list of alerts.
        """
        alerts = self.alert_history
        
        if level:
            alerts = [a for a in alerts if a["level"] == level]
        
        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_metrics(self, metric_type: str, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get metrics for a specific type.
        
        Args:
            metric_type: The type of metrics to get.
            start_time: Start time for filtering.
            end_time: End time for filtering.
            
        Returns:
            A list of metrics.
        """
        if metric_type not in self.metrics:
            return []
        
        metrics = self.metrics[metric_type]
        
        if start_time:
            metrics = [m for m in metrics if m["timestamp"] >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m["timestamp"] <= end_time]
        
        return sorted(metrics, key=lambda x: x["timestamp"])
