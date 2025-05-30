"""
Performance Optimization Agent for the self-aware web scraping system.

This agent continuously optimizes system performance.
"""
import asyncio
import logging
import time
import json
import os
import psutil
import platform
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import uuid

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority
from models.task import Task, TaskStatus, TaskType
from models.intelligence import AgentCapability


class PerformanceOptimizationAgent(Agent):
    """
    Performance Optimization Agent that continuously optimizes system performance.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new Performance Optimization Agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent. Used for message routing.
        """
        super().__init__(agent_id=agent_id, agent_type="performance_optimization", coordinator_id=coordinator_id)
        
        # Performance metrics
        self.metrics: Dict[str, Dict[str, Any]] = {
            "system": {},
            "agents": {},
            "tasks": {}
        }
        
        # Optimization settings
        self.settings: Dict[str, Any] = {
            "cpu_threshold": 0.8,  # 80% CPU usage threshold
            "memory_threshold": 0.8,  # 80% memory usage threshold
            "task_timeout": 300,  # 5 minutes task timeout
            "max_concurrent_tasks": 10,
            "max_retries": 3,
            "backoff_factor": 2.0,
            "optimization_interval": 60  # 1 minute optimization interval
        }
        
        # Register message handlers
        self.register_handler("monitor_performance", self._handle_monitor_performance)
        self.register_handler("optimize_performance", self._handle_optimize_performance)
        self.register_handler("update_metrics", self._handle_update_metrics)
        self.register_handler("get_recommendations", self._handle_get_recommendations)
        
        # Start periodic tasks
        self._start_periodic_tasks()
    
    def _start_periodic_tasks(self) -> None:
        """Start periodic tasks for the Performance Optimization Agent."""
        asyncio.create_task(self._periodic_performance_monitoring())
        asyncio.create_task(self._periodic_optimization())
    
    async def _periodic_performance_monitoring(self) -> None:
        """Periodically monitor system performance."""
        while self.running:
            self.logger.debug("Running periodic performance monitoring")
            
            # Monitor system performance
            await self.monitor_system_performance()
            
            # Sleep for 10 seconds
            await asyncio.sleep(10)
    
    async def _periodic_optimization(self) -> None:
        """Periodically optimize system performance."""
        while self.running:
            self.logger.debug("Running periodic optimization")
            
            # Optimize performance
            await self.optimize_performance()
            
            # Sleep for optimization interval
            await asyncio.sleep(self.settings["optimization_interval"])
    
    async def monitor_system_performance(self) -> Dict[str, Any]:
        """
        Monitor system performance.

        Returns:
            A dictionary containing system performance metrics.
        """
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Update system metrics
        self.metrics["system"] = {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available": memory.available,
            "memory_total": memory.total,
            "disk_percent": disk.percent,
            "disk_free": disk.free,
            "disk_total": disk.total,
            "platform": platform.system(),
            "platform_release": platform.release(),
            "python_version": platform.python_version()
        }
        
        return self.metrics["system"]
    
    async def monitor_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """
        Monitor agent performance.

        Args:
            agent_id: The ID of the agent to monitor.

        Returns:
            A dictionary containing agent performance metrics.
        """
        # Get agent metrics
        # In a real implementation, you would get metrics from the agent
        # For now, we'll use placeholder metrics
        
        # Check if we have existing metrics for this agent
        if agent_id not in self.metrics["agents"]:
            self.metrics["agents"][agent_id] = {
                "timestamp": time.time(),
                "task_count": 0,
                "success_count": 0,
                "error_count": 0,
                "average_execution_time": 0.0,
                "cpu_usage": 0.0,
                "memory_usage": 0.0
            }
        
        return self.metrics["agents"][agent_id]
    
    async def monitor_task_performance(self, task_id: str) -> Dict[str, Any]:
        """
        Monitor task performance.

        Args:
            task_id: The ID of the task to monitor.

        Returns:
            A dictionary containing task performance metrics.
        """
        # Get task metrics
        # In a real implementation, you would get metrics from the task
        # For now, we'll use placeholder metrics
        
        # Check if we have existing metrics for this task
        if task_id not in self.metrics["tasks"]:
            self.metrics["tasks"][task_id] = {
                "timestamp": time.time(),
                "start_time": time.time(),
                "end_time": None,
                "execution_time": None,
                "status": "running",
                "agent_id": None,
                "cpu_usage": 0.0,
                "memory_usage": 0.0
            }
        
        return self.metrics["tasks"][task_id]
    
    async def update_agent_metrics(self, agent_id: str, metrics: Dict[str, Any]) -> None:
        """
        Update agent metrics.

        Args:
            agent_id: The ID of the agent.
            metrics: The metrics to update.
        """
        # Update agent metrics
        if agent_id not in self.metrics["agents"]:
            self.metrics["agents"][agent_id] = {}
        
        self.metrics["agents"][agent_id].update(metrics)
        self.metrics["agents"][agent_id]["timestamp"] = time.time()
    
    async def update_task_metrics(self, task_id: str, metrics: Dict[str, Any]) -> None:
        """
        Update task metrics.

        Args:
            task_id: The ID of the task.
            metrics: The metrics to update.
        """
        # Update task metrics
        if task_id not in self.metrics["tasks"]:
            self.metrics["tasks"][task_id] = {}
        
        self.metrics["tasks"][task_id].update(metrics)
        self.metrics["tasks"][task_id]["timestamp"] = time.time()
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """
        Optimize system performance.

        Returns:
            A dictionary containing optimization results.
        """
        self.logger.info("Optimizing performance")
        
        # Check system performance
        system_metrics = self.metrics["system"]
        
        # Initialize optimization results
        optimization_results = {
            "timestamp": time.time(),
            "optimizations_applied": [],
            "recommendations": []
        }
        
        # Check CPU usage
        if system_metrics.get("cpu_percent", 0) > self.settings["cpu_threshold"] * 100:
            self.logger.warning(f"High CPU usage: {system_metrics.get('cpu_percent')}%")
            
            # Add recommendation to reduce concurrent tasks
            optimization_results["recommendations"].append({
                "type": "reduce_concurrent_tasks",
                "reason": f"High CPU usage: {system_metrics.get('cpu_percent')}%",
                "current_value": self.settings["max_concurrent_tasks"],
                "recommended_value": max(1, self.settings["max_concurrent_tasks"] - 2)
            })
            
            # Apply optimization if CPU usage is very high
            if system_metrics.get("cpu_percent", 0) > 90:
                old_value = self.settings["max_concurrent_tasks"]
                self.settings["max_concurrent_tasks"] = max(1, self.settings["max_concurrent_tasks"] - 2)
                
                optimization_results["optimizations_applied"].append({
                    "type": "reduce_concurrent_tasks",
                    "old_value": old_value,
                    "new_value": self.settings["max_concurrent_tasks"]
                })
        
        # Check memory usage
        if system_metrics.get("memory_percent", 0) > self.settings["memory_threshold"] * 100:
            self.logger.warning(f"High memory usage: {system_metrics.get('memory_percent')}%")
            
            # Add recommendation to reduce memory usage
            optimization_results["recommendations"].append({
                "type": "reduce_memory_usage",
                "reason": f"High memory usage: {system_metrics.get('memory_percent')}%",
                "current_value": "N/A",
                "recommended_value": "Optimize memory-intensive operations"
            })
        
        # Check agent performance
        for agent_id, agent_metrics in self.metrics["agents"].items():
            # Check execution time
            if agent_metrics.get("average_execution_time", 0) > 10.0:
                self.logger.warning(f"High execution time for agent {agent_id}: {agent_metrics.get('average_execution_time')}s")
                
                # Add recommendation to optimize agent
                optimization_results["recommendations"].append({
                    "type": "optimize_agent",
                    "agent_id": agent_id,
                    "reason": f"High execution time: {agent_metrics.get('average_execution_time')}s",
                    "recommended_value": "Optimize agent implementation"
                })
            
            # Check error rate
            if agent_metrics.get("task_count", 0) > 0:
                error_rate = agent_metrics.get("error_count", 0) / agent_metrics.get("task_count", 1)
                if error_rate > 0.2:  # 20% error rate
                    self.logger.warning(f"High error rate for agent {agent_id}: {error_rate:.2%}")
                    
                    # Add recommendation to improve error handling
                    optimization_results["recommendations"].append({
                        "type": "improve_error_handling",
                        "agent_id": agent_id,
                        "reason": f"High error rate: {error_rate:.2%}",
                        "recommended_value": "Improve error handling in agent"
                    })
        
        # Check task performance
        running_tasks = 0
        long_running_tasks = []
        
        for task_id, task_metrics in self.metrics["tasks"].items():
            # Count running tasks
            if task_metrics.get("status") == "running":
                running_tasks += 1
                
                # Check for long-running tasks
                if task_metrics.get("start_time") and time.time() - task_metrics.get("start_time", 0) > self.settings["task_timeout"]:
                    long_running_tasks.append(task_id)
        
        # Add recommendation for long-running tasks
        if long_running_tasks:
            self.logger.warning(f"Long-running tasks detected: {long_running_tasks}")
            
            optimization_results["recommendations"].append({
                "type": "timeout_long_running_tasks",
                "reason": f"Long-running tasks detected: {len(long_running_tasks)}",
                "task_ids": long_running_tasks,
                "recommended_value": "Implement task timeouts"
            })
        
        # Update optimization settings based on system load
        if running_tasks > 0:
            # Adjust max_concurrent_tasks based on CPU and memory usage
            cpu_factor = 1.0 - (system_metrics.get("cpu_percent", 0) / 100)
            memory_factor = 1.0 - (system_metrics.get("memory_percent", 0) / 100)
            load_factor = min(cpu_factor, memory_factor)
            
            # Calculate new max_concurrent_tasks
            new_max_tasks = max(1, int(20 * load_factor))
            
            # Only change if the difference is significant
            if abs(new_max_tasks - self.settings["max_concurrent_tasks"]) >= 2:
                old_value = self.settings["max_concurrent_tasks"]
                self.settings["max_concurrent_tasks"] = new_max_tasks
                
                optimization_results["optimizations_applied"].append({
                    "type": "adjust_concurrent_tasks",
                    "old_value": old_value,
                    "new_value": self.settings["max_concurrent_tasks"],
                    "reason": f"System load factor: {load_factor:.2f}"
                })
        
        self.logger.info(f"Performance optimization complete: {len(optimization_results['optimizations_applied'])} optimizations applied, {len(optimization_results['recommendations'])} recommendations")
        
        return optimization_results
    
    async def get_performance_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get performance optimization recommendations.

        Returns:
            A list of performance optimization recommendations.
        """
        # Optimize performance to generate recommendations
        optimization_results = await self.optimize_performance()
        
        # Return recommendations
        return optimization_results["recommendations"]
    
    async def _handle_monitor_performance(self, message: Message) -> None:
        """
        Handle a monitor performance message.

        Args:
            message: The message to handle.
        """
        try:
            # Monitor system performance
            system_metrics = await self.monitor_system_performance()
            
            # Monitor agent performance if specified
            agent_metrics = {}
            if hasattr(message, "agent_id"):
                agent_metrics = await self.monitor_agent_performance(message.agent_id)
            
            # Monitor task performance if specified
            task_metrics = {}
            if hasattr(message, "task_id"):
                task_metrics = await self.monitor_task_performance(message.task_id)
            
            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={
                    "system": system_metrics,
                    "agent": agent_metrics,
                    "task": task_metrics
                }
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error monitoring performance: {str(e)}")
            
            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)
    
    async def _handle_optimize_performance(self, message: Message) -> None:
        """
        Handle an optimize performance message.

        Args:
            message: The message to handle.
        """
        try:
            # Optimize performance
            optimization_results = await self.optimize_performance()
            
            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=optimization_results
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error optimizing performance: {str(e)}")
            
            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)
    
    async def _handle_update_metrics(self, message: Message) -> None:
        """
        Handle an update metrics message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "metrics") or not isinstance(message.metrics, dict):
            self.logger.warning("Received update_metrics message without valid metrics")
            return
        
        try:
            # Update agent metrics if specified
            if hasattr(message, "agent_id"):
                await self.update_agent_metrics(message.agent_id, message.metrics)
            
            # Update task metrics if specified
            if hasattr(message, "task_id"):
                await self.update_task_metrics(message.task_id, message.metrics)
            
            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"success": True}
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
            
            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)
    
    async def _handle_get_recommendations(self, message: Message) -> None:
        """
        Handle a get recommendations message.

        Args:
            message: The message to handle.
        """
        try:
            # Get recommendations
            recommendations = await self.get_performance_recommendations()
            
            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"recommendations": recommendations}
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {str(e)}")
            
            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        self.logger.info(f"Executing task: {task.type}")
        
        if task.type == TaskType.OPTIMIZE_PERFORMANCE:
            # Optimize performance
            target_components = task.parameters.get("target_components", ["system", "agents", "tasks"])
            optimization_goals = task.parameters.get("optimization_goals", ["speed", "memory", "accuracy"])
            
            # Update settings if provided
            if "max_resource_usage" in task.parameters:
                resource_usage = task.parameters["max_resource_usage"]
                if "cpu" in resource_usage:
                    self.settings["cpu_threshold"] = resource_usage["cpu"]
                if "memory" in resource_usage:
                    self.settings["memory_threshold"] = resource_usage["memory"]
            
            # Optimize performance
            return await self.optimize_performance()
        
        elif task.type == TaskType.MONITOR_SYSTEM:
            # Monitor system performance
            return await self.monitor_system_performance()
        
        elif task.type == TaskType.TRACK_PERFORMANCE:
            # Track performance of a specific agent or task
            agent_id = task.parameters.get("agent_id")
            task_id = task.parameters.get("task_id")
            
            result = {
                "system": await self.monitor_system_performance()
            }
            
            if agent_id:
                result["agent"] = await self.monitor_agent_performance(agent_id)
            
            if task_id:
                result["task"] = await self.monitor_task_performance(task_id)
            
            return result
        
        else:
            raise ValueError(f"Unsupported task type: {task.type}")
