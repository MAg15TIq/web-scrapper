"""
Enhanced Coordinator agent for the web scraping system with advanced features.
"""
import asyncio
import logging
import time
import json
import os
import math
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from queue import PriorityQueue
from datetime import datetime, timedelta
from collections import defaultdict
import random

from agents.coordinator import CoordinatorAgent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, ControlMessage, Priority
from models.task import Task, TaskStatus, TaskType


class EnhancedCoordinatorAgent(CoordinatorAgent):
    """
    Enhanced Coordinator agent with adaptive scheduling, failure recovery, performance analytics,
    load balancing, and workflow templates.
    """
    def __init__(self, agent_id: str = "coordinator"):
        """
        Initialize a new enhanced coordinator agent.

        Args:
            agent_id: Unique identifier for the agent.
        """
        super().__init__(agent_id=agent_id)
        
        # Performance tracking
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        self.task_performance: Dict[str, Dict[str, Any]] = {}
        
        # Failure recovery
        self.retry_policies: Dict[str, Dict[str, Any]] = {
            "default": {
                "max_retries": 3,
                "backoff_factor": 2.0,
                "max_backoff": 300,  # 5 minutes
                "jitter": True
            }
        }
        self.failed_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Load balancing
        self.load_metrics: Dict[str, Dict[str, Any]] = {}
        self.load_balancing_strategy = "performance_based"  # or "round_robin", "least_busy"
        
        # Workflow templates
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        self.templates_dir = os.path.join("config", "workflow_templates")
        os.makedirs(self.templates_dir, exist_ok=True)
        self._load_workflow_templates()
        
        # Adaptive scheduling
        self.task_priorities: Dict[str, Dict[str, float]] = {}
        self.priority_factors = {
            "time_sensitivity": 0.4,
            "value": 0.3,
            "difficulty": 0.2,
            "dependencies": 0.1
        }
        
        # Register additional message handlers
        self.register_handler("performance", self._handle_performance_message)
        
        # Start periodic tasks
        self._start_periodic_tasks()
    
    def _start_periodic_tasks(self) -> None:
        """Start periodic tasks for the coordinator."""
        # Schedule periodic tasks
        asyncio.create_task(self._periodic_performance_analysis())
        asyncio.create_task(self._periodic_load_balancing())
        asyncio.create_task(self._periodic_retry_failed_tasks())
    
    async def _periodic_performance_analysis(self) -> None:
        """Periodically analyze agent performance."""
        while self.running:
            try:
                # Analyze performance every 5 minutes
                await asyncio.sleep(300)
                if not self.running:
                    break
                
                self.logger.info("Running periodic performance analysis")
                await self._analyze_agent_performance()
            except Exception as e:
                self.logger.error(f"Error in periodic performance analysis: {str(e)}", exc_info=True)
    
    async def _periodic_load_balancing(self) -> None:
        """Periodically balance load across agents."""
        while self.running:
            try:
                # Balance load every 2 minutes
                await asyncio.sleep(120)
                if not self.running:
                    break
                
                self.logger.info("Running periodic load balancing")
                await self._balance_agent_load()
            except Exception as e:
                self.logger.error(f"Error in periodic load balancing: {str(e)}", exc_info=True)
    
    async def _periodic_retry_failed_tasks(self) -> None:
        """Periodically retry failed tasks."""
        while self.running:
            try:
                # Check for failed tasks every 30 seconds
                await asyncio.sleep(30)
                if not self.running:
                    break
                
                self.logger.info("Checking for failed tasks to retry")
                await self._retry_eligible_failed_tasks()
            except Exception as e:
                self.logger.error(f"Error in periodic retry of failed tasks: {str(e)}", exc_info=True)
    
    async def _analyze_agent_performance(self) -> Dict[str, Any]:
        """
        Analyze the performance of all agents.
        
        Returns:
            A dictionary containing performance metrics.
        """
        performance_data = {
            "timestamp": time.time(),
            "agents": {},
            "task_types": {},
            "overall": {
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "throughput": 0.0
            }
        }
        
        # Calculate agent-specific metrics
        for agent_id, agent_info in self.agents.items():
            if agent_id not in self.agent_performance:
                continue
                
            perf = self.agent_performance[agent_id]
            
            # Calculate success rate
            total_tasks = perf.get("completed_tasks", 0) + perf.get("failed_tasks", 0)
            success_rate = perf.get("completed_tasks", 0) / max(total_tasks, 1)
            
            # Calculate average execution time
            avg_execution_time = perf.get("total_execution_time", 0) / max(perf.get("completed_tasks", 1), 1)
            
            # Store metrics
            performance_data["agents"][agent_id] = {
                "success_rate": success_rate,
                "avg_execution_time": avg_execution_time,
                "throughput": perf.get("completed_tasks", 0) / max(perf.get("uptime", 1), 1),
                "task_count": total_tasks
            }
        
        # Calculate task type metrics
        for task_type, task_perf in self.task_performance.items():
            total_tasks = task_perf.get("completed", 0) + task_perf.get("failed", 0)
            success_rate = task_perf.get("completed", 0) / max(total_tasks, 1)
            avg_execution_time = task_perf.get("total_execution_time", 0) / max(task_perf.get("completed", 1), 1)
            
            performance_data["task_types"][task_type] = {
                "success_rate": success_rate,
                "avg_execution_time": avg_execution_time,
                "count": total_tasks
            }
        
        # Calculate overall metrics
        total_completed = sum(perf.get("completed_tasks", 0) for perf in self.agent_performance.values())
        total_failed = sum(perf.get("failed_tasks", 0) for perf in self.agent_performance.values())
        total_execution_time = sum(perf.get("total_execution_time", 0) for perf in self.agent_performance.values())
        
        total_tasks = total_completed + total_failed
        overall_success_rate = total_completed / max(total_tasks, 1)
        overall_avg_execution_time = total_execution_time / max(total_completed, 1)
        
        performance_data["overall"] = {
            "success_rate": overall_success_rate,
            "avg_execution_time": overall_avg_execution_time,
            "total_tasks": total_tasks,
            "completed_tasks": total_completed,
            "failed_tasks": total_failed
        }
        
        self.logger.info(f"Performance analysis completed: {json.dumps(performance_data['overall'])}")
        return performance_data
    
    async def _handle_performance_message(self, message: Message) -> None:
        """
        Handle a performance message from an agent.
        
        Args:
            message: The performance message to handle.
        """
        self.logger.debug(f"Received performance update from {message.sender_id}")
        
        # Extract performance data
        agent_id = message.sender_id
        performance_data = message.payload
        
        # Update agent performance data
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                "completed_tasks": 0,
                "failed_tasks": 0,
                "total_execution_time": 0.0,
                "uptime": 0.0,
                "task_type_stats": {}
            }
        
        # Update with new data
        for key, value in performance_data.items():
            if key == "task_type_stats":
                # Merge task type statistics
                for task_type, stats in value.items():
                    if task_type not in self.agent_performance[agent_id]["task_type_stats"]:
                        self.agent_performance[agent_id]["task_type_stats"][task_type] = stats
                    else:
                        for stat_key, stat_value in stats.items():
                            self.agent_performance[agent_id]["task_type_stats"][task_type][stat_key] = stat_value
            else:
                # Update simple metrics
                self.agent_performance[agent_id][key] = value
    
    async def _balance_agent_load(self) -> None:
        """Balance load across agents based on performance metrics."""
        self.logger.info("Balancing agent load")
        
        # Group agents by type
        agents_by_type = defaultdict(list)
        for agent_id, agent_info in self.agents.items():
            agents_by_type[agent_info["type"]].append(agent_id)
        
        # Check each agent type for load imbalance
        for agent_type, agent_ids in agents_by_type.items():
            if len(agent_ids) <= 1:
                continue  # No need to balance with only one agent
            
            # Calculate load metrics for each agent
            load_metrics = {}
            for agent_id in agent_ids:
                if agent_id in self.agents:
                    # Use task count as a simple load metric
                    load_metrics[agent_id] = self.agents[agent_id]["tasks"]
            
            # Check for imbalance
            if not load_metrics:
                continue
                
            avg_load = sum(load_metrics.values()) / len(load_metrics)
            max_load = max(load_metrics.values())
            min_load = min(load_metrics.values())
            
            # If imbalance exceeds threshold (20%), rebalance
            if max_load > 0 and (max_load - min_load) / max_load > 0.2:
                self.logger.info(f"Load imbalance detected for {agent_type} agents: max={max_load}, min={min_load}, avg={avg_load:.2f}")
                
                # Identify overloaded and underloaded agents
                overloaded = [agent_id for agent_id, load in load_metrics.items() if load > avg_load * 1.1]
                underloaded = [agent_id for agent_id, load in load_metrics.items() if load < avg_load * 0.9]
                
                if overloaded and underloaded:
                    self.logger.info(f"Rebalancing load from {overloaded} to {underloaded}")
                    
                    # TODO: Implement task reassignment logic
                    # This would involve:
                    # 1. Identifying tasks that can be reassigned
                    # 2. Sending control messages to agents to pause/cancel tasks
                    # 3. Reassigning tasks to underloaded agents
    
    async def _retry_eligible_failed_tasks(self) -> None:
        """Retry failed tasks that are eligible for retry."""
        current_time = time.time()
        tasks_to_retry = []
        
        # Find tasks eligible for retry
        for task_id, failure_info in list(self.failed_tasks.items()):
            if current_time >= failure_info.get("next_retry_time", 0):
                tasks_to_retry.append(task_id)
                
                # Update retry count
                failure_info["retry_count"] += 1
                
                # If max retries reached, remove from failed tasks
                if failure_info["retry_count"] >= failure_info["max_retries"]:
                    self.logger.warning(f"Task {task_id} has reached max retries ({failure_info['max_retries']}), giving up")
                    del self.failed_tasks[task_id]
                else:
                    # Calculate next retry time with exponential backoff
                    backoff_factor = failure_info["backoff_factor"]
                    max_backoff = failure_info["max_backoff"]
                    retry_count = failure_info["retry_count"]
                    
                    # Calculate delay with exponential backoff
                    delay = min(max_backoff, (backoff_factor ** retry_count))
                    
                    # Add jitter if enabled
                    if failure_info.get("jitter", True):
                        delay = delay * (0.5 + random.random())
                    
                    failure_info["next_retry_time"] = current_time + delay
                    self.logger.info(f"Scheduled next retry for task {task_id} in {delay:.2f} seconds (retry {retry_count}/{failure_info['max_retries']})")
        
        # Retry the eligible tasks
        for task_id in tasks_to_retry:
            if task_id in self.all_tasks:
                task = self.all_tasks[task_id]
                
                # Reset task status to pending
                task.status = TaskStatus.PENDING
                
                # Resubmit the task
                self.logger.info(f"Retrying failed task {task_id} (type: {task.type})")
                await self.submit_task(task)
    
    async def _handle_error_message(self, message: ErrorMessage) -> None:
        """
        Handle an error message from an agent with enhanced failure recovery.
        
        Args:
            message: The error message to handle.
        """
        self.logger.warning(f"Received error for task {message.task_id} from {message.sender_id}: {message.error_type} - {message.error_message}")
        
        # Update task status
        if message.task_id in self.all_tasks:
            task = self.all_tasks[message.task_id]
            task.fail(message.error_type, message.error_message, message.traceback)
            
            # Update agent information
            if task.assigned_to in self.agents:
                self.agents[task.assigned_to]["tasks"] -= 1
                self.agents[task.assigned_to]["status"] = "idle"
            
            # Add to failed tasks for potential retry
            self._schedule_task_retry(task, message.error_type)
        else:
            self.logger.warning(f"Received error for unknown task: {message.task_id}")
    
    def _schedule_task_retry(self, task: Task, error_type: str) -> None:
        """
        Schedule a failed task for retry with exponential backoff.
        
        Args:
            task: The failed task.
            error_type: The type of error that occurred.
        """
        # Skip retry for certain error types or if task already has too many retries
        if hasattr(task, "retry_count") and getattr(task, "retry_count", 0) >= 5:
            self.logger.warning(f"Task {task.id} has already been retried too many times, giving up")
            return
        
        # Get retry policy based on task type or error type
        retry_policy = self.retry_policies.get(task.type, self.retry_policies["default"])
        
        # Initialize retry count
        retry_count = getattr(task, "retry_count", 0)
        
        # Store failed task info
        self.failed_tasks[task.id] = {
            "task": task,
            "error_type": error_type,
            "retry_count": retry_count,
            "max_retries": retry_policy["max_retries"],
            "backoff_factor": retry_policy["backoff_factor"],
            "max_backoff": retry_policy["max_backoff"],
            "jitter": retry_policy["jitter"],
            "next_retry_time": time.time() + retry_policy["backoff_factor"] ** retry_count
        }
        
        # Update task's retry count for future reference
        setattr(task, "retry_count", retry_count + 1)
        
        self.logger.info(f"Scheduled retry for failed task {task.id} (retry {retry_count + 1}/{retry_policy['max_retries']})")
    
    async def _handle_result_message(self, message: ResultMessage) -> None:
        """
        Handle a result message from an agent with performance tracking.
        
        Args:
            message: The result message to handle.
        """
        self.logger.info(f"Received result for task {message.task_id} from {message.sender_id}")
        
        # Update task status
        if message.task_id in self.all_tasks:
            task = self.all_tasks[message.task_id]
            task.complete(message.result)
            
            # Update agent information
            if task.assigned_to in self.agents:
                self.agents[task.assigned_to]["tasks"] -= 1
                self.agents[task.assigned_to]["status"] = "idle"
            
            # Update performance metrics
            self._update_performance_metrics(task, message.sender_id, message.execution_time)
            
            # Check if any waiting tasks can now be scheduled
            await self._check_waiting_tasks()
        else:
            self.logger.warning(f"Received result for unknown task: {message.task_id}")
    
    def _update_performance_metrics(self, task: Task, agent_id: str, execution_time: float) -> None:
        """
        Update performance metrics for a completed task.
        
        Args:
            task: The completed task.
            agent_id: The ID of the agent that completed the task.
            execution_time: The time taken to execute the task.
        """
        # Initialize agent performance data if not exists
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                "completed_tasks": 0,
                "failed_tasks": 0,
                "total_execution_time": 0.0,
                "uptime": 0.0,
                "task_type_stats": {}
            }
        
        # Initialize task type performance data if not exists
        if task.type not in self.task_performance:
            self.task_performance[task.type] = {
                "completed": 0,
                "failed": 0,
                "total_execution_time": 0.0,
                "avg_execution_time": 0.0
            }
        
        # Update agent performance metrics
        self.agent_performance[agent_id]["completed_tasks"] += 1
        self.agent_performance[agent_id]["total_execution_time"] += execution_time
        
        # Update task type stats for this agent
        if task.type not in self.agent_performance[agent_id]["task_type_stats"]:
            self.agent_performance[agent_id]["task_type_stats"][task.type] = {
                "completed": 0,
                "failed": 0,
                "total_execution_time": 0.0
            }
        
        self.agent_performance[agent_id]["task_type_stats"][task.type]["completed"] += 1
        self.agent_performance[agent_id]["task_type_stats"][task.type]["total_execution_time"] += execution_time
        
        # Update task type performance metrics
        self.task_performance[task.type]["completed"] += 1
        self.task_performance[task.type]["total_execution_time"] += execution_time
        self.task_performance[task.type]["avg_execution_time"] = (
            self.task_performance[task.type]["total_execution_time"] / 
            self.task_performance[task.type]["completed"]
        )
    
    def _load_workflow_templates(self) -> None:
        """Load workflow templates from the templates directory."""
        if not os.path.exists(self.templates_dir):
            return
            
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.templates_dir, filename), 'r') as f:
                        template = json.load(f)
                        template_name = os.path.splitext(filename)[0]
                        self.workflow_templates[template_name] = template
                        self.logger.info(f"Loaded workflow template: {template_name}")
                except Exception as e:
                    self.logger.error(f"Error loading workflow template {filename}: {str(e)}")
    
    def save_workflow_template(self, template_name: str, template: Dict[str, Any]) -> bool:
        """
        Save a workflow template.
        
        Args:
            template_name: The name of the template.
            template: The template data.
            
        Returns:
            True if the template was saved successfully, False otherwise.
        """
        try:
            # Validate template
            if "workflow_steps" not in template or not isinstance(template["workflow_steps"], list):
                self.logger.error(f"Invalid workflow template: {template_name} - missing or invalid workflow_steps")
                return False
            
            # Save to memory
            self.workflow_templates[template_name] = template
            
            # Save to file
            filename = os.path.join(self.templates_dir, f"{template_name}.json")
            with open(filename, 'w') as f:
                json.dump(template, f, indent=2)
            
            self.logger.info(f"Saved workflow template: {template_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving workflow template {template_name}: {str(e)}")
            return False
    
    async def execute_workflow_template(self, template_name: str, variables: Dict[str, Any] = None) -> List[str]:
        """
        Execute a workflow template.
        
        Args:
            template_name: The name of the template to execute.
            variables: Variables to substitute in the template.
            
        Returns:
            A list of task IDs created from the workflow.
        """
        if template_name not in self.workflow_templates:
            raise ValueError(f"Workflow template not found: {template_name}")
        
        template = self.workflow_templates[template_name]
        variables = variables or {}
        
        self.logger.info(f"Executing workflow template: {template_name}")
        
        # Create tasks from workflow steps
        task_ids = []
        tasks_by_index = {}
        
        for i, step in enumerate(template["workflow_steps"]):
            # Substitute variables in parameters
            parameters = self._substitute_variables(step.get("parameters", {}), variables)
            
            # Create task
            task = Task(
                type=step["task_type"],
                parameters=parameters,
                priority=step.get("priority", 1)
            )
            
            # Add dependencies
            if "depends_on" in step:
                depends_on = step["depends_on"]
                if isinstance(depends_on, int):
                    # Single dependency by index
                    if depends_on in tasks_by_index:
                        task.dependencies.append(tasks_by_index[depends_on].id)
                elif isinstance(depends_on, list):
                    # Multiple dependencies by index
                    for dep_idx in depends_on:
                        if dep_idx in tasks_by_index:
                            task.dependencies.append(tasks_by_index[dep_idx].id)
            
            # Submit task
            task_id = await self.submit_task(task)
            task_ids.append(task_id)
            tasks_by_index[i] = task
        
        return task_ids
    
    def _substitute_variables(self, obj: Any, variables: Dict[str, Any]) -> Any:
        """
        Substitute variables in an object.
        
        Args:
            obj: The object to substitute variables in.
            variables: The variables to substitute.
            
        Returns:
            The object with variables substituted.
        """
        if isinstance(obj, str):
            # Substitute variables in string
            for var_name, var_value in variables.items():
                obj = obj.replace(f"{{{var_name}}}", str(var_value))
            return obj
        elif isinstance(obj, dict):
            # Substitute variables in dictionary
            return {k: self._substitute_variables(v, variables) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Substitute variables in list
            return [self._substitute_variables(item, variables) for item in obj]
        else:
            # Return as is
            return obj
    
    async def submit_task(self, task: Task) -> str:
        """
        Submit a new task to the system with adaptive priority.
        
        Args:
            task: The task to submit.
            
        Returns:
            The ID of the submitted task.
        """
        # Calculate adaptive priority if not explicitly set
        if task.priority == 1 and hasattr(self, "calculate_task_priority"):
            task.priority = self.calculate_task_priority(task)
        
        self.logger.info(f"Submitting task: {task.id} (type: {task.type}, priority: {task.priority})")
        
        # Store the task
        self.all_tasks[task.id] = task
        
        # Check if the task has dependencies
        if task.dependencies:
            # Check if all dependencies are completed
            all_completed = True
            for dep_id in task.dependencies:
                if dep_id not in self.all_tasks or self.all_tasks[dep_id].status != TaskStatus.COMPLETED:
                    all_completed = False
                    break
            
            if not all_completed:
                # Add to waiting tasks
                self.waiting_tasks.add(task.id)
                self.logger.info(f"Task {task.id} waiting for dependencies")
                return task.id
        
        # Add to task queue
        self.task_queue.put((task.priority, task))
        
        # Trigger task assignment
        asyncio.create_task(self._assign_tasks())
        
        return task.id
    
    def calculate_task_priority(self, task: Task) -> int:
        """
        Calculate the priority of a task based on various factors.
        
        Args:
            task: The task to calculate priority for.
            
        Returns:
            The calculated priority (higher is more important).
        """
        # Base priority
        priority = 1
        
        # Factor 1: Time sensitivity (tasks that need to be done quickly)
        time_sensitivity = task.parameters.get("time_sensitivity", 0)
        
        # Factor 2: Value (importance of the task)
        value = task.parameters.get("value", 0)
        
        # Factor 3: Difficulty (complex tasks may need more time)
        difficulty = task.parameters.get("difficulty", 0)
        
        # Factor 4: Dependencies (tasks with many dependents should be prioritized)
        dependency_count = 0
        for other_task in self.all_tasks.values():
            if task.id in other_task.dependencies:
                dependency_count += 1
        
        # Calculate weighted priority
        weighted_priority = (
            self.priority_factors["time_sensitivity"] * time_sensitivity +
            self.priority_factors["value"] * value +
            self.priority_factors["difficulty"] * difficulty +
            self.priority_factors["dependencies"] * min(dependency_count, 10) / 10
        )
        
        # Convert to integer priority (1-10)
        priority = max(1, min(10, int(weighted_priority * 10)))
        
        return priority
    
    async def _find_agent_for_task(self, task: Task) -> Optional[str]:
        """
        Find an appropriate agent for a task using performance-based selection.
        
        Args:
            task: The task to assign.
            
        Returns:
            The ID of the selected agent, or None if no suitable agent is found.
        """
        # Determine which agent types can handle this task
        agent_types = self._get_agent_types_for_task(task.type)
        
        if not agent_types:
            self.logger.warning(f"No agent type can handle task type: {task.type}")
            return None
        
        # Get all available agents of the appropriate types
        available_agents = []
        for agent_type in agent_types:
            if agent_type not in self.agents_by_type:
                continue
            
            for agent_id in self.agents_by_type[agent_type]:
                agent_info = self.agents[agent_id]
                
                # Skip agents that are not idle
                if agent_info["status"] != "idle":
                    continue
                
                available_agents.append((agent_id, agent_info))
        
        if not available_agents:
            return None
        
        # Select agent based on strategy
        if self.load_balancing_strategy == "performance_based":
            # Use performance metrics if available
            best_agent_id = None
            best_score = float('-inf')
            
            for agent_id, agent_info in available_agents:
                # Calculate performance score
                if agent_id in self.agent_performance:
                    perf = self.agent_performance[agent_id]
                    
                    # Calculate success rate for this task type
                    task_type_stats = perf.get("task_type_stats", {}).get(task.type, {})
                    completed = task_type_stats.get("completed", 0)
                    failed = task_type_stats.get("failed", 0)
                    total = completed + failed
                    
                    if total > 0:
                        success_rate = completed / total
                    else:
                        success_rate = 0.5  # Neutral if no data
                    
                    # Calculate execution time score (lower is better)
                    avg_time = task_type_stats.get("total_execution_time", 0) / max(completed, 1)
                    time_score = 1.0 / max(avg_time, 0.1)  # Avoid division by zero
                    
                    # Calculate load score (lower load is better)
                    load = agent_info["tasks"]
                    load_score = 1.0 / (load + 1)  # +1 to avoid division by zero
                    
                    # Combined score (higher is better)
                    score = (success_rate * 0.5) + (time_score * 0.3) + (load_score * 0.2)
                    
                    if score > best_score:
                        best_score = score
                        best_agent_id = agent_id
            
            # If we found a good agent, return it
            if best_agent_id:
                return best_agent_id
        
        # Fallback to least busy strategy
        return min(available_agents, key=lambda x: x[1]["tasks"])[0]
