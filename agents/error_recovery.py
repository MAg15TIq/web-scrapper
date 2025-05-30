"""
Error Recovery agent for the web scraping system.
"""
import asyncio
import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import re
import traceback
import sys
from pathlib import Path
import hashlib
import shutil
import gzip
import pickle
import signal
import psutil
import aiohttp
import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage
from models.task import Task, TaskStatus, TaskType


class ErrorRecoveryAgent(Agent):
    """
    Agent responsible for error recovery and system resilience.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new error recovery agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent.
        """
        super().__init__(agent_id=agent_id, agent_type="error_recovery")
        self.coordinator_id = coordinator_id

        # Error tracking
        self.error_history: List[Dict[str, Any]] = []
        self.max_error_history = 1000
        self.error_patterns: Dict[str, Dict[str, Any]] = {}

        # System monitoring
        self.system_metrics: Dict[str, Dict[str, Any]] = {
            "cpu": {"threshold": 80, "current": 0},
            "memory": {"threshold": 85, "current": 0},
            "disk": {"threshold": 90, "current": 0},
            "network": {"threshold": 70, "current": 0}
        }

        # Recovery strategies
        self.recovery_strategies: Dict[str, Dict[str, Any]] = {
            "system": {
                "restart": {"priority": 1, "max_attempts": 3},
                "cleanup": {"priority": 2, "max_attempts": 5},
                "scale": {"priority": 3, "max_attempts": 2}
            },
            "process": {
                "restart": {"priority": 1, "max_attempts": 3},
                "reload": {"priority": 2, "max_attempts": 5},
                "reset": {"priority": 3, "max_attempts": 2}
            },
            "service": {
                "restart": {"priority": 1, "max_attempts": 3},
                "reconfigure": {"priority": 2, "max_attempts": 5},
                "fallback": {"priority": 3, "max_attempts": 2}
            }
        }

        # Recovery history
        self.recovery_history: List[Dict[str, Any]] = []
        self.max_recovery_history = 100

        # Register message handlers
        self.register_handler("handle_error", self._handle_error)
        self.register_handler("check_system", self._handle_check_system)
        self.register_handler("recover_system", self._handle_recover_system)
        self.register_handler("update_metrics", self._handle_update_metrics)

        # Start periodic tasks
        self._start_periodic_tasks()

    def _start_periodic_tasks(self) -> None:
        """Start periodic tasks for the error recovery agent."""
        asyncio.create_task(self._periodic_error_analysis())
        asyncio.create_task(self._periodic_system_check())
        asyncio.create_task(self._periodic_cleanup())

    async def _periodic_error_analysis(self) -> None:
        """Periodically analyze error patterns."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                if not self.running:
                    break

                await self._analyze_error_patterns()
            except Exception as e:
                self.logger.error(f"Error in error analysis: {str(e)}", exc_info=True)

    async def _periodic_system_check(self) -> None:
        """Periodically check system health."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Every minute
                if not self.running:
                    break

                await self._check_system_health()
            except Exception as e:
                self.logger.error(f"Error in system check: {str(e)}", exc_info=True)

    async def _periodic_cleanup(self) -> None:
        """Periodically clean up old data."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Every hour
                if not self.running:
                    break

                await self._cleanup_old_data()
            except Exception as e:
                self.logger.error(f"Error in cleanup: {str(e)}", exc_info=True)

    async def _analyze_error_patterns(self) -> None:
        """Analyze error patterns and update recovery strategies."""
        # Group errors by type
        error_groups: Dict[str, List[Dict[str, Any]]] = {}
        for error in self.error_history:
            error_type = error.get("type", "unknown")
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(error)

        # Analyze each group
        for error_type, errors in error_groups.items():
            # Calculate frequency
            frequency = len(errors) / len(self.error_history) if self.error_history else 0

            # Calculate average impact
            impact = sum(error.get("impact", 0) for error in errors) / len(errors)

            # Update pattern
            self.error_patterns[error_type] = {
                "frequency": frequency,
                "impact": impact,
                "last_occurrence": max(error["timestamp"] for error in errors),
                "recovery_strategy": self._determine_recovery_strategy(error_type, frequency, impact)
            }

    async def _check_system_health(self) -> None:
        """Check system health and trigger recovery if needed."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()

            # Update metrics
            self.system_metrics["cpu"]["current"] = cpu_percent
            self.system_metrics["memory"]["current"] = memory.percent
            self.system_metrics["disk"]["current"] = disk.percent
            self.system_metrics["network"]["current"] = (
                (network.bytes_sent + network.bytes_recv) / 1024 / 1024
            )  # MB/s

            # Check for issues
            issues = []
            for metric, data in self.system_metrics.items():
                if data["current"] > data["threshold"]:
                    issues.append({
                        "type": metric,
                        "current": data["current"],
                        "threshold": data["threshold"]
                    })

            # Trigger recovery if needed
            if issues:
                await self._trigger_recovery("system", issues)

        except Exception as e:
            self.logger.error(f"Error checking system health: {str(e)}", exc_info=True)

    async def _cleanup_old_data(self) -> None:
        """Clean up old error and recovery history."""
        current_time = time.time()

        # Clean up error history
        self.error_history = [
            error for error in self.error_history
            if current_time - error["timestamp"] < 86400  # 24 hours
        ]

        # Clean up recovery history
        self.recovery_history = [
            recovery for recovery in self.recovery_history
            if current_time - recovery["timestamp"] < 86400  # 24 hours
        ]

    async def _handle_error(self, message: Message) -> None:
        """Handle error message."""
        try:
            error_data = message.data
            error_type = error_data.get("type", "unknown")

            # Add to history
            self.error_history.append({
                "type": error_type,
                "message": error_data.get("message", ""),
                "timestamp": time.time(),
                "impact": error_data.get("impact", 1),
                "context": error_data.get("context", {})
            })

            # Trim history if needed
            if len(self.error_history) > self.max_error_history:
                self.error_history = self.error_history[-self.max_error_history:]

            # Determine recovery strategy
            strategy = self._determine_recovery_strategy(
                error_type,
                self.error_patterns.get(error_type, {}).get("frequency", 0),
                self.error_patterns.get(error_type, {}).get("impact", 1)
            )

            # Trigger recovery
            await self._trigger_recovery(error_type, error_data, strategy)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"status": "recovery_triggered", "strategy": strategy}
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error handling error message: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="error_handling_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_check_system(self, message: Message) -> None:
        """Handle system check request."""
        try:
            await self._check_system_health()

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"metrics": self.system_metrics}
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error checking system: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="system_check_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_recover_system(self, message: Message) -> None:
        """Handle system recovery request."""
        try:
            recovery_type = message.data.get("type", "system")
            options = message.data.get("options", {})

            result = await self._recover_system(recovery_type, options)

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error recovering system: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="system_recovery_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_update_metrics(self, message: Message) -> None:
        """Handle metrics update request."""
        try:
            metrics = message.data.get("metrics", {})

            # Update metrics
            for metric, value in metrics.items():
                if metric in self.system_metrics:
                    self.system_metrics[metric]["current"] = value

            # Check for issues
            await self._check_system_health()

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result={"status": "success", "metrics": self.system_metrics}
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="metrics_update_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    def _determine_recovery_strategy(self, error_type: str, frequency: float, impact: float) -> str:
        """Determine the best recovery strategy based on error characteristics."""
        if error_type == "system":
            if impact > 0.8 or frequency > 0.5:
                return "restart"
            elif impact > 0.5 or frequency > 0.3:
                return "cleanup"
            else:
                return "scale"
        elif error_type == "process":
            if impact > 0.8 or frequency > 0.5:
                return "restart"
            elif impact > 0.5 or frequency > 0.3:
                return "reload"
            else:
                return "reset"
        elif error_type == "service":
            if impact > 0.8 or frequency > 0.5:
                return "restart"
            elif impact > 0.5 or frequency > 0.3:
                return "reconfigure"
            else:
                return "fallback"
        else:
            return "restart"

    async def _trigger_recovery(self, error_type: str, context: Dict[str, Any], strategy: Optional[str] = None) -> None:
        """Trigger recovery based on error type and context."""
        if not strategy:
            strategy = self._determine_recovery_strategy(
                error_type,
                self.error_patterns.get(error_type, {}).get("frequency", 0),
                self.error_patterns.get(error_type, {}).get("impact", 1)
            )

        # Get recovery options
        options = self.recovery_strategies.get(error_type, {}).get(strategy, {})

        # Check attempt limits
        attempts = sum(
            1 for recovery in self.recovery_history
            if recovery["type"] == error_type and recovery["strategy"] == strategy
        )

        if attempts >= options.get("max_attempts", 3):
            self.logger.warning(f"Max recovery attempts reached for {error_type} using {strategy}")
            return

        # Execute recovery
        result = await self._recover_system(error_type, {
            "strategy": strategy,
            "context": context,
            "attempt": attempts + 1
        })

        # Record recovery attempt
        self.recovery_history.append({
            "type": error_type,
            "strategy": strategy,
            "timestamp": time.time(),
            "result": result
        })

        # Trim history if needed
        if len(self.recovery_history) > self.max_recovery_history:
            self.recovery_history = self.recovery_history[-self.max_recovery_history:]

    async def _recover_system(self, recovery_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute system recovery based on type and options."""
        strategy = options.get("strategy", "restart")
        context = options.get("context", {})
        attempt = options.get("attempt", 1)

        try:
            if recovery_type == "system":
                if strategy == "restart":
                    # Restart system components
                    await self._restart_system_components()
                elif strategy == "cleanup":
                    # Clean up system resources
                    await self._cleanup_system_resources()
                elif strategy == "scale":
                    # Scale system resources
                    await self._scale_system_resources()

            elif recovery_type == "process":
                if strategy == "restart":
                    # Restart process
                    await self._restart_process()
                elif strategy == "reload":
                    # Reload process configuration
                    await self._reload_process_config()
                elif strategy == "reset":
                    # Reset process state
                    await self._reset_process_state()

            elif recovery_type == "service":
                if strategy == "restart":
                    # Restart service
                    await self._restart_service()
                elif strategy == "reconfigure":
                    # Reconfigure service
                    await self._reconfigure_service()
                elif strategy == "fallback":
                    # Fall back to backup service
                    await self._fallback_service()

            return {
                "status": "success",
                "type": recovery_type,
                "strategy": strategy,
                "attempt": attempt
            }

        except Exception as e:
            self.logger.error(f"Error in system recovery: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "type": recovery_type,
                "strategy": strategy,
                "attempt": attempt,
                "error": str(e)
            }

    async def _restart_system_components(self) -> None:
        """Restart system components."""
        try:
            # Get all Python processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'python' in proc.info['name'].lower():
                    # Skip the current process
                    if proc.info['pid'] == os.getpid():
                        continue

                    # Restart the process
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        proc.kill()

            # Restart the current process
            os.execv(sys.executable, [sys.executable] + sys.argv)

        except Exception as e:
            self.logger.error(f"Error restarting system components: {str(e)}", exc_info=True)
            raise

    async def _cleanup_system_resources(self) -> None:
        """Clean up system resources."""
        try:
            # Clean up temporary files
            temp_dir = Path(os.getenv('TEMP', '/tmp'))
            for file in temp_dir.glob('web_scraper_*'):
                try:
                    if file.is_file():
                        file.unlink()
                except Exception as e:
                    self.logger.warning(f"Error deleting temp file {file}: {str(e)}")

            # Clean up old log files
            log_dir = Path('logs')
            if log_dir.exists():
                for file in log_dir.glob('*.log'):
                    try:
                        if file.stat().st_mtime < time.time() - 7 * 24 * 3600:  # 7 days
                            file.unlink()
                    except Exception as e:
                        self.logger.warning(f"Error deleting log file {file}: {str(e)}")

            # Clean up cache directories
            cache_dir = Path('cache')
            if cache_dir.exists():
                for file in cache_dir.glob('*'):
                    try:
                        if file.stat().st_mtime < time.time() - 24 * 3600:  # 24 hours
                            if file.is_file():
                                file.unlink()
                            elif file.is_dir():
                                shutil.rmtree(file)
                    except Exception as e:
                        self.logger.warning(f"Error deleting cache file {file}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error cleaning up system resources: {str(e)}", exc_info=True)
            raise

    async def _scale_system_resources(self) -> None:
        """Scale system resources."""
        try:
            # Get current system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Adjust process priority based on system load
            current_process = psutil.Process(os.getpid())

            if cpu_percent > 90 or memory.percent > 90:
                # High load - lower priority
                current_process.nice(10)
            elif cpu_percent < 50 and memory.percent < 50:
                # Low load - higher priority
                current_process.nice(-10)
            else:
                # Normal load - normal priority
                current_process.nice(0)

            # Adjust memory limits
            if memory.percent > 80:
                # Reduce memory usage
                import gc
                gc.collect()

                # Clear caches if they exist
                if hasattr(self, 'extraction_cache'):
                    self.extraction_cache.clear()
                if hasattr(self, 'error_history'):
                    self.error_history = self.error_history[-100:]  # Keep only last 100 errors

        except Exception as e:
            self.logger.error(f"Error scaling system resources: {str(e)}", exc_info=True)
            raise

    async def _restart_process(self) -> None:
        """Restart a process."""
        try:
            # Get the current process
            current_process = psutil.Process(os.getpid())

            # Save current state
            state_file = Path('process_state.pkl')
            with open(state_file, 'wb') as f:
                pickle.dump({
                    'error_history': self.error_history,
                    'recovery_history': self.recovery_history,
                    'system_metrics': self.system_metrics
                }, f)

            # Restart the process
            os.execv(sys.executable, [sys.executable] + sys.argv)

        except Exception as e:
            self.logger.error(f"Error restarting process: {str(e)}", exc_info=True)
            raise

    async def _reload_process_config(self) -> None:
        """Reload process configuration."""
        try:
            # Load configuration from file
            config_file = Path('config.json')
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)

                # Update agent configuration
                if 'error_recovery' in config:
                    recovery_config = config['error_recovery']

                    # Update thresholds
                    if 'thresholds' in recovery_config:
                        for metric, threshold in recovery_config['thresholds'].items():
                            if metric in self.system_metrics:
                                self.system_metrics[metric]['threshold'] = threshold

                    # Update recovery strategies
                    if 'strategies' in recovery_config:
                        self.recovery_strategies.update(recovery_config['strategies'])

                    # Update history limits
                    if 'max_error_history' in recovery_config:
                        self.max_error_history = recovery_config['max_error_history']
                    if 'max_recovery_history' in recovery_config:
                        self.max_recovery_history = recovery_config['max_recovery_history']

        except Exception as e:
            self.logger.error(f"Error reloading process config: {str(e)}", exc_info=True)
            raise

    async def _reset_process_state(self) -> None:
        """Reset process state."""
        try:
            # Clear error history
            self.error_history = []

            # Clear recovery history
            self.recovery_history = []

            # Reset system metrics
            for metric in self.system_metrics:
                self.system_metrics[metric]['current'] = 0

            # Clear error patterns
            self.error_patterns = {}

            # Reset recovery strategies to defaults
            self.recovery_strategies = {
                "system": {
                    "restart": {"priority": 1, "max_attempts": 3},
                    "cleanup": {"priority": 2, "max_attempts": 5},
                    "scale": {"priority": 3, "max_attempts": 2}
                },
                "process": {
                    "restart": {"priority": 1, "max_attempts": 3},
                    "reload": {"priority": 2, "max_attempts": 5},
                    "reset": {"priority": 3, "max_attempts": 2}
                },
                "service": {
                    "restart": {"priority": 1, "max_attempts": 3},
                    "reconfigure": {"priority": 2, "max_attempts": 5},
                    "fallback": {"priority": 3, "max_attempts": 2}
                }
            }

        except Exception as e:
            self.logger.error(f"Error resetting process state: {str(e)}", exc_info=True)
            raise

    async def _restart_service(self) -> None:
        """Restart a service."""
        try:
            # Get service name from context
            service_name = self.context.get('service_name', 'web_scraper')

            # Check if running as a service
            if os.name == 'nt':  # Windows
                import win32serviceutil
                win32serviceutil.RestartService(service_name)
            else:  # Linux/Unix
                import subprocess
                subprocess.run(['systemctl', 'restart', service_name], check=True)

        except Exception as e:
            self.logger.error(f"Error restarting service: {str(e)}", exc_info=True)
            raise

    async def _reconfigure_service(self) -> None:
        """Reconfigure a service."""
        try:
            # Get service name from context
            service_name = self.context.get('service_name', 'web_scraper')

            # Load new configuration
            config_file = Path('service_config.json')
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)

                # Update service configuration
                if os.name == 'nt':  # Windows
                    import win32serviceutil
                    win32serviceutil.ChangeServiceConfig(
                        service_name,
                        win32serviceutil.SERVICE_NO_CHANGE,
                        win32serviceutil.SERVICE_NO_CHANGE,
                        win32serviceutil.SERVICE_NO_CHANGE,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None
                    )
                else:  # Linux/Unix
                    import subprocess
                    subprocess.run(['systemctl', 'daemon-reload'], check=True)
                    subprocess.run(['systemctl', 'restart', service_name], check=True)

        except Exception as e:
            self.logger.error(f"Error reconfiguring service: {str(e)}", exc_info=True)
            raise

    async def _fallback_service(self) -> None:
        """Fall back to a backup service."""
        try:
            # Get service name from context
            service_name = self.context.get('service_name', 'web_scraper')
            backup_service = self.context.get('backup_service', f'{service_name}_backup')

            # Stop current service
            if os.name == 'nt':  # Windows
                import win32serviceutil
                win32serviceutil.StopService(service_name)
            else:  # Linux/Unix
                import subprocess
                subprocess.run(['systemctl', 'stop', service_name], check=True)

            # Start backup service
            if os.name == 'nt':  # Windows
                win32serviceutil.StartService(backup_service)
            else:  # Linux/Unix
                subprocess.run(['systemctl', 'start', backup_service], check=True)

        except Exception as e:
            self.logger.error(f"Error falling back to backup service: {str(e)}", exc_info=True)
            raise

    async def _send_message(self, message: Message) -> None:
        """
        Send a message to another agent.

        Args:
            message: The message to send.
        """
        # Add message to outbox
        self.outbox.put(message)

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        if task.type == TaskType.CHECK_SYSTEM:
            await self._check_system_health()
            return {"metrics": self.system_metrics}

        elif task.type == TaskType.RECOVER_SYSTEM:
            recovery_type = task.parameters.get("type", "system")
            options = task.parameters.get("options", {})
            return await self._recover_system(recovery_type, options)

        elif task.type == TaskType.HANDLE_ERROR:
            error_data = task.parameters
            error_type = error_data.get("type", "unknown")

            # Add to history
            self.error_history.append({
                "type": error_type,
                "message": error_data.get("message", ""),
                "timestamp": time.time(),
                "impact": error_data.get("impact", 1),
                "context": error_data.get("context", {})
            })

            # Trim history if needed
            if len(self.error_history) > self.max_error_history:
                self.error_history = self.error_history[-self.max_error_history:]

            # Determine recovery strategy
            strategy = self._determine_recovery_strategy(
                error_type,
                self.error_patterns.get(error_type, {}).get("frequency", 0),
                self.error_patterns.get(error_type, {}).get("impact", 1)
            )

            # Trigger recovery
            await self._trigger_recovery(error_type, error_data, strategy)

            return {"status": "recovery_triggered", "strategy": strategy}

        else:
            raise ValueError(f"Unsupported task type for error recovery agent: {task.type}")